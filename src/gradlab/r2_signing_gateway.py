"""Trusted-side SigV4 gateway behind a native Agent Vault Basic service.

Install this stdlib-only module in the protected broker runtime. It never loads
source credentials: the native proxy supplies one request's Basic credential.
Ordinary GradLab clients use placeholders and never import this as a key helper.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import re
import ssl
import stat
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http.client import HTTPSConnection
from urllib.parse import parse_qsl, quote, urlencode, urlsplit


MAX_BODY_BYTES = 256 * 1024 * 1024
METHODS = frozenset({"GET", "HEAD", "PUT", "DELETE"})
HOP_HEADERS = frozenset({
    "authorization", "proxy-authorization", "proxy-connection", "connection",
    "keep-alive", "transfer-encoding", "te", "trailer", "upgrade", "expect",
    "host", "user-agent", "accept-encoding", "x-amz-date", "x-amz-content-sha256",
})
ERROR_CODES = {400: "InvalidRequest", 403: "AccessDenied", 404: "NoSuchKey",
               409: "OperationAborted", 412: "PreconditionFailed", 429: "SlowDown"}


def validate_target(method: str, target: str, bucket: str) -> None:
    parsed = urlsplit(target)
    path = parsed.path
    if (method not in METHODS or parsed.scheme or parsed.netloc or parsed.fragment
        or not path.isascii() or not target.isascii() or "\\" in target
        or any(ord(c) < 32 or ord(c) == 127 for c in target)
        or re.search(r"%(?![0-9a-fA-F]{2})|%(?:2f|5c|00)", path, re.I)
        or any(segment in {".", ".."} for segment in path.split("/"))
        or re.search(r"%(?:2e)", path, re.I)
        or not (path == "/" + bucket or path.startswith("/" + bucket + "/"))
        or (method in {"PUT", "DELETE"} and path.rstrip("/") == "/" + bucket)):
        raise ValueError("Unreviewed R2 operation")


def basic_credential(value: str) -> tuple[str, str]:
    if not value.startswith("Basic ") or len(value) > 2048:
        raise ValueError("Expected brokered Basic credential")
    try:
        decoded = base64.b64decode(value[6:], validate=True).decode("ascii")
        access, secret = decoded.split(":", 1)
    except (ValueError, UnicodeError):
        raise ValueError("Invalid brokered credential") from None
    if (not re.fullmatch(r"[A-Za-z0-9]{8,128}", access)
        or not re.fullmatch(r"[A-Za-z0-9/+=]{16,256}", secret)
        or "placeholder" in secret.lower() or "placeholder" in access.lower()):
        raise ValueError("Expected real broker-side credential")
    return access, secret


def signed_headers(
    method: str, target: str, headers: dict[str, str], body: bytes, *,
    access: str, secret: str, host: str, region: str = "auto", now: datetime | None = None,
) -> dict[str, str]:
    """Sign an already URL-encoded S3 request; do not normalize its object path."""
    now = now or datetime.now(UTC)
    stamp, day = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
    result = {
        name.lower(): " ".join(value.split())
        for name, value in headers.items()
        if name.lower() not in HOP_HEADERS
    }
    if "x-amz-security-token" in result:
        raise ValueError("Temporary upstream credentials need separate review")
    payload_hash = hashlib.sha256(body).hexdigest()
    result.update({"host": host, "x-amz-date": stamp, "x-amz-content-sha256": payload_hash})
    parsed = urlsplit(target)
    pairs = [part.partition("=") for part in parsed.query.split("&")] if parsed.query else []
    query = "&".join(f"{key}={value}" for key, _, value in sorted(pairs))
    names = sorted(result)
    canonical = "\n".join([
        method, parsed.path or "/", query,
        "".join(f"{name}:{result[name]}\n" for name in names), ";".join(names), payload_hash,
    ])
    scope = f"{day}/{region}/s3/aws4_request"
    to_sign = f"AWS4-HMAC-SHA256\n{stamp}\n{scope}\n{hashlib.sha256(canonical.encode()).hexdigest()}"
    key = ("AWS4" + secret).encode()
    for value in (day, region, "s3", "aws4_request"):
        key = hmac.new(key, value.encode(), hashlib.sha256).digest()
    signature = hmac.new(key, to_sign.encode(), hashlib.sha256).hexdigest()
    result["authorization"] = (
        f"AWS4-HMAC-SHA256 Credential={access}/{scope}, "
        f"SignedHeaders={';'.join(names)}, Signature={signature}"
    )
    return result


def presigned_url(method, path, *, access, secret, host, expires, content_type=None, now=None):
    if method not in {"GET", "PUT"} or not 1 <= expires <= 604800:
        raise ValueError("Invalid signing lifetime or method")
    now = now or datetime.now(UTC)
    stamp, day = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
    scope = f"{day}/auto/s3/aws4_request"
    headers = {"host": host}
    if content_type is not None:
        if method != "PUT" or len(content_type) > 256 or any(ord(c) < 32 for c in content_type):
            raise ValueError("Invalid signing content type")
        headers["content-type"] = content_type
    names = sorted(headers)
    params = {"X-Amz-Algorithm": "AWS4-HMAC-SHA256", "X-Amz-Credential": f"{access}/{scope}",
              "X-Amz-Date": stamp, "X-Amz-Expires": str(expires), "X-Amz-SignedHeaders": ";".join(names)}
    query = urlencode(sorted(params.items()), quote_via=quote, safe="~")
    canonical = "\n".join([method, path, query,
                            "".join(f"{name}:{headers[name]}\n" for name in names),
                            ";".join(names), "UNSIGNED-PAYLOAD"])
    to_sign = f"AWS4-HMAC-SHA256\n{stamp}\n{scope}\n{hashlib.sha256(canonical.encode()).hexdigest()}"
    key = ("AWS4" + secret).encode()
    for value in (day, "auto", "s3", "aws4_request"):
        key = hmac.new(key, value.encode(), hashlib.sha256).digest()
    signature = hmac.new(key, to_sign.encode(), hashlib.sha256).hexdigest()
    return f"https://{host}{path}?{query}&X-Amz-Signature={signature}"


class GatewayServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 16

    def __init__(self, address, *, bucket: str, endpoint: str):
        parsed = urlsplit(endpoint)
        if (parsed.scheme != "https" or parsed.path not in {"", "/"}
            or parsed.query or parsed.fragment or parsed.username or parsed.password
            or not parsed.hostname or parsed.port not in {None, 443}
            or not re.fullmatch(r"[a-f0-9]{32}\.r2\.cloudflarestorage\.com", parsed.hostname)
            or not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,62}", bucket)):
            raise ValueError("Expected one fixed R2 account and bucket")
        self.bucket, self.upstream = bucket, parsed.hostname
        self.slots = threading.BoundedSemaphore(2)
        self.tls = ssl.create_default_context()
        super().__init__(address, GatewayHandler)


class GatewayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "gradlab-r2-gateway/1"
    sys_version = ""

    def log_message(self, format, *args):
        pass

    def _error(self, status: int) -> None:
        # Upstream error bodies may reflect headers/credentials; never relay them.
        code = ERROR_CODES.get(status, "InternalError")
        body = f"<Error><Code>{code}</Code><Message>R2 gateway request failed</Message></Error>".encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/xml")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)
        self.close_connection = True

    def _request(self) -> None:
        connection = None
        if not self.server.slots.acquire(blocking=False):
            self._error(503)
            return
        try:
            self.connection.settimeout(35)
            validate_target(self.command, self.path, self.server.bucket)
            if (self.headers.get("Transfer-Encoding") is not None
                or len(self.headers.get_all("Authorization", [])) != 1
                or len(self.headers.get_all("Content-Length", [])) > 1):
                raise ValueError("Unreviewed framing")
            access, secret = basic_credential(self.headers.get("Authorization", ""))
            parsed = urlsplit(self.path)
            query = dict(parse_qsl(parsed.query))
            if "__gradlab_presign" in query:
                if (self.command != "GET" or len(query) != len(parse_qsl(parsed.query))
                    or set(query) - {"__gradlab_presign", "expires", "content_type"}):
                    raise ValueError("Unreviewed signing request")
                url = presigned_url(
                    query["__gradlab_presign"], parsed.path, access=access, secret=secret,
                    host=self.server.upstream, expires=int(query["expires"]), content_type=query.get("content_type"),
                )
                body = json.dumps({"url": url}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)
                self.close_connection = True
                return
            size = int(self.headers.get("Content-Length", "0"))
            if size < 0 or size > MAX_BODY_BYTES or (size and self.command != "PUT"):
                raise ValueError("Unreviewed payload size")
            body = self.rfile.read(size)
            if len(body) != size:
                raise ValueError("Incomplete request")
            copy_source = self.headers.get("x-amz-copy-source")
            if copy_source is not None:
                if self.command != "PUT":
                    raise ValueError("Unreviewed copy operation")
                validate_target("GET", "/" + copy_source.lstrip("/"), self.server.bucket)
            headers = signed_headers(
                self.command, self.path, dict(self.headers.items()), body,
                access=access, secret=secret, host=self.server.upstream,
            )
            del access, secret
            connection = HTTPSConnection(self.server.upstream, timeout=30, context=self.server.tls)
            connection.request(self.command, self.path, body=body if size else None, headers=headers)
            response = connection.getresponse()
            if not 200 <= response.status < 300:
                self._error(response.status if response.status in ERROR_CODES else 502)
                return
            self.send_response(response.status)
            for name, value in response.getheaders():
                lower = name.lower()
                if (lower in {"content-length", "content-type", "etag", "last-modified", "cache-control"}
                    or lower.startswith("x-amz-meta-")):
                    self.send_header(name, value)
            self.send_header("Connection", "close")
            self.end_headers()
            if self.command != "HEAD":
                while chunk := response.read(64 * 1024):
                    self.wfile.write(chunk)
            self.close_connection = True
        except (ValueError, UnicodeError):
            self._error(403)
        except Exception:
            # Do not print request, credentials, provider errors or tracebacks.
            self._error(502)
        finally:
            if connection is not None:
                connection.close()
            self.server.slots.release()

    do_GET = do_HEAD = do_PUT = do_DELETE = _request


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    from pathlib import Path
    source = Path(args.config)
    info = source.lstat()
    if (source.is_symlink() or not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.geteuid() or info.st_mode & 0o022):
        raise ValueError("Gateway metadata must be protected")
    with open(args.config, encoding="utf-8") as handle:
        config = json.load(handle)
    if set(config) != {"bucket", "endpoint", "port"} or not 1024 <= config["port"] <= 65535:
        raise ValueError("Invalid gateway configuration")
    with GatewayServer(("127.0.0.1", config["port"]), bucket=config["bucket"], endpoint=config["endpoint"]) as server:
        server.serve_forever()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise SystemExit("R2 signing gateway stopped; details withheld") from None
