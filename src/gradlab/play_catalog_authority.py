from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import threading
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from gradlab.catalog_errors import CatalogIntegrityError, CatalogUnavailable
from gradlab.operator_credentials import OPERATOR_CONFIG_ENV, PROTECTED_ENV_NAMES


CONTROL_R2_ENV_NAMES = frozenset(
    {
        "GRADLAB_CONTROL_R2_URI",
        "GRADLAB_CONTROL_R2_ENDPOINT_URL",
        "GRADLAB_CONTROL_R2_REGION",
        "GRADLAB_CONTROL_R2_ACCESS_KEY_ID",
        "GRADLAB_CONTROL_R2_SECRET_ACCESS_KEY",
    }
)
MAX_REQUEST_BYTES = 16 * 1024
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_BULK_READ_KEYS = 256
MAX_BULK_REQUEST_KEYS = 64
MAX_BULK_READ_WORKERS = 16

_ALLOWED_CONTROL_KEYS = (
    re.compile(r"goal-catalog/v1/goals/[0-9a-f]{64}/current\.json"),
    re.compile(
        r"goal-catalog/v1/goals/[0-9a-f]{64}/generations/[0-9a-f]{64}\.json"
    ),
    re.compile(r"goal-catalog/v1/goals/[0-9a-f]{64}/pages/[0-9a-f]{64}\.json"),
    re.compile(r"runs/gradlab-[0-9a-f]{32}/manifest\.json"),
    re.compile(r"recipes/v2/sha256/[0-9a-f]{2}/[0-9a-f]{64}\.json"),
)


def _allowed_control_key(key: object) -> str:
    value = str(key or "").strip("/")
    if not any(pattern.fullmatch(value) for pattern in _ALLOWED_CONTROL_KEYS):
        raise ValueError("catalog helper rejected an unsupported control object")
    return value


def _send_json(sock: socket.socket, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    if len(encoded) > MAX_RESPONSE_BYTES:
        raise ValueError("catalog helper response exceeds the size limit")
    sock.sendall(len(encoded).to_bytes(8, "big") + encoded)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = int(size)
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("catalog helper connection closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_json(sock: socket.socket, *, limit: int) -> dict[str, Any]:
    size = int.from_bytes(_recv_exact(sock, 8), "big")
    if size < 2 or size > int(limit):
        raise ValueError("catalog helper message has an invalid size")
    value = json.loads(_recv_exact(sock, size).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("catalog helper message must be a JSON object")
    return value


class CatalogAuthorityClient:
    """Read-only control-catalog client backed by an isolated helper process."""

    def __init__(
        self,
        sock: socket.socket,
        *,
        process: subprocess.Popen[bytes] | None = None,
    ) -> None:
        self._socket = sock
        self._process = process
        self._lock = threading.Lock()
        self._closed = False
        self.authority_identity = ""

    def _request(self, operation: str, **payload: Any) -> Any:
        with self._lock:
            if self._closed:
                raise CatalogUnavailable("catalog authority helper is closed")
            try:
                _send_json(self._socket, {"operation": operation, **payload})
                response = _recv_json(self._socket, limit=MAX_RESPONSE_BYTES)
            except (EOFError, OSError, ValueError, json.JSONDecodeError) as exc:
                raise CatalogUnavailable(
                    f"catalog authority helper failed: {exc}",
                    code="catalog_transient",
                    retryable=True,
                ) from exc
        if response.get("ok") is not True:
            code = str(response.get("code") or "catalog_unavailable")
            message = str(response.get("message") or "catalog authority is unavailable")
            if code == "catalog_integrity":
                raise CatalogIntegrityError(message, source="control-catalog")
            raise CatalogUnavailable(
                message,
                code=code,
                retryable=bool(response.get("retryable")),
            )
        return response.get("value")

    def ready(self) -> None:
        value = self._request("ready")
        if not isinstance(value, Mapping):
            raise CatalogIntegrityError(
                "catalog helper readiness response is malformed",
                source="control-catalog",
            )
        identity = str(value.get("authority_identity") or "")
        if re.fullmatch(r"[0-9a-f]{32}", identity) is None:
            raise CatalogIntegrityError(
                "catalog helper authority identity is malformed",
                source="control-catalog",
            )
        self.authority_identity = identity

    def get_json_optional(self, key: str) -> dict[str, Any] | None:
        try:
            validated_key = _allowed_control_key(key)
        except ValueError as exc:
            raise CatalogIntegrityError(
                str(exc),
                source="control-catalog",
            ) from exc
        value = self._request("get_json_optional", key=validated_key)
        if value is None:
            return None
        if not isinstance(value, dict):
            raise CatalogIntegrityError(
                "control catalog returned a non-object document",
                source="control-catalog",
            )
        return value

    def get_json_many_optional(
        self,
        keys: Iterable[str],
    ) -> dict[str, dict[str, Any] | None]:
        try:
            validated_keys = tuple(_allowed_control_key(key) for key in keys)
        except ValueError as exc:
            raise CatalogIntegrityError(
                str(exc),
                source="control-catalog",
            ) from exc
        if len(validated_keys) > MAX_BULK_READ_KEYS:
            raise CatalogIntegrityError(
                "catalog helper bulk read exceeds the key limit",
                source="control-catalog",
            )
        if len(validated_keys) != len(set(validated_keys)):
            raise CatalogIntegrityError(
                "catalog helper bulk read contains duplicate keys",
                source="control-catalog",
            )
        if not validated_keys:
            return {}
        documents: dict[str, dict[str, Any] | None] = {}
        for offset in range(0, len(validated_keys), MAX_BULK_REQUEST_KEYS):
            chunk = validated_keys[offset : offset + MAX_BULK_REQUEST_KEYS]
            value = self._request("get_json_many_optional", keys=list(chunk))
            if not isinstance(value, Mapping) or set(value) != set(chunk):
                raise CatalogIntegrityError(
                    "control catalog returned a malformed bulk response",
                    source="control-catalog",
                )
            for key in chunk:
                document = value[key]
                if document is not None and not isinstance(document, dict):
                    raise CatalogIntegrityError(
                        "control catalog returned a non-object document",
                        source="control-catalog",
                    )
                documents[key] = document
        return documents

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                _send_json(self._socket, {"operation": "close"})
            except OSError:
                pass
            self._closed = True
            try:
                self._socket.close()
            except OSError:
                pass
        if self._process is not None:
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=2)


def _helper_environment() -> dict[str, str]:
    allowed = {
        "HOME",
        "LANG",
        "LC_ALL",
        "PATH",
        "PYTHONPATH",
        "PYTHONHOME",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "TMPDIR",
        "XDG_CONFIG_HOME",
        OPERATOR_CONFIG_ENV,
        *CONTROL_R2_ENV_NAMES,
    }
    return {
        name: value
        for name, value in os.environ.items()
        if name in allowed and str(value).strip()
    }


def start_catalog_authority_helper(repo_root: Path | str) -> CatalogAuthorityClient:
    parent, child = socket.socketpair()
    try:
        child.set_inheritable(True)
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "gradlab.play_catalog_authority",
                "--serve-fd",
                str(child.fileno()),
                "--repo-root",
                str(Path(repo_root).resolve()),
            ],
            close_fds=True,
            pass_fds=(child.fileno(),),
            env=_helper_environment(),
            start_new_session=True,
        )
    except Exception:
        parent.close()
        child.close()
        raise
    child.close()
    client = CatalogAuthorityClient(parent, process=process)
    try:
        client.ready()
    except Exception:
        client.close()
        raise
    return client


def scrub_protected_environment() -> None:
    for name in PROTECTED_ENV_NAMES:
        os.environ.pop(name, None)


def _serve(sock: socket.socket, repo_root: Path) -> int:
    from gradlab.operator_environment import load_repository_operator_environment
    from gradlab.r2_store import BucketConfig, R2Bucket

    bucket = None
    startup_error: tuple[str, str] | None = None
    try:
        load_repository_operator_environment(
            repo_root,
            requested_names=CONTROL_R2_ENV_NAMES,
        )
        bucket = R2Bucket(BucketConfig.from_env("GRADLAB_CONTROL_R2"))
    except Exception as exc:
        startup_error = ("catalog_configuration", str(exc))

    while True:
        try:
            request = _recv_json(sock, limit=MAX_REQUEST_BYTES)
        except EOFError:
            return 0
        except Exception as exc:
            _send_json(
                sock,
                {
                    "ok": False,
                    "code": "catalog_integrity",
                    "message": f"invalid helper request: {exc}",
                },
            )
            continue
        operation = str(request.get("operation") or "")
        if operation == "close":
            return 0
        if startup_error is not None:
            _send_json(
                sock,
                {
                    "ok": False,
                    "code": startup_error[0],
                    "message": startup_error[1],
                    "retryable": False,
                },
            )
            continue
        if operation == "ready":
            assert bucket is not None
            authority_identity = hashlib.sha256(
                (
                    f"{bucket.config.uri}\0{bucket.config.endpoint_url}\0"
                    f"{bucket.config.region}"
                ).encode("utf-8")
            ).hexdigest()[:32]
            _send_json(
                sock,
                {
                    "ok": True,
                    "value": {"authority_identity": authority_identity},
                },
            )
            continue
        if operation not in {"get_json_optional", "get_json_many_optional"}:
            _send_json(
                sock,
                {
                    "ok": False,
                    "code": "catalog_integrity",
                    "message": "catalog helper operation is not allowed",
                },
            )
            continue
        try:
            assert bucket is not None
            if operation == "get_json_optional":
                key = _allowed_control_key(request.get("key"))
                value = bucket.get_json_optional(key)
            else:
                raw_keys = request.get("keys")
                if not isinstance(raw_keys, list):
                    raise ValueError("catalog helper bulk keys must be a list")
                keys = tuple(_allowed_control_key(key) for key in raw_keys)
                if len(keys) > MAX_BULK_REQUEST_KEYS:
                    raise ValueError("catalog helper bulk read exceeds the key limit")
                if len(keys) != len(set(keys)):
                    raise ValueError("catalog helper bulk read contains duplicate keys")
                if not keys:
                    value = {}
                else:
                    with ThreadPoolExecutor(
                        max_workers=min(MAX_BULK_READ_WORKERS, len(keys)),
                        thread_name_prefix="gradlab-catalog-read",
                    ) as pool:
                        documents = tuple(pool.map(bucket.get_json_optional, keys))
                    value = dict(zip(keys, documents, strict=True))
        except ValueError as exc:
            _send_json(
                sock,
                {
                    "ok": False,
                    "code": "catalog_integrity",
                    "message": str(exc),
                },
            )
        except Exception as exc:
            _send_json(
                sock,
                {
                    "ok": False,
                    "code": "catalog_transient",
                    "message": f"control catalog read failed: {exc}",
                    "retryable": True,
                },
            )
        else:
            _send_json(sock, {"ok": True, "value": value})


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--serve-fd", type=int, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    sock = socket.socket(fileno=int(args.serve_fd))
    try:
        return _serve(sock, args.repo_root.resolve())
    finally:
        sock.close()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONTROL_R2_ENV_NAMES",
    "CatalogAuthorityClient",
    "scrub_protected_environment",
    "start_catalog_authority_helper",
]
