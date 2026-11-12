"""Manual native Agent Vault proof using disposable catalog services and fake keys."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import threading
import uuid
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
OUT = ROOT / ".internal/evidence/training-vault-migration-20261112"
HOST = "0" * 32 + ".r2.cloudflarestorage.com"
ROLES = [("CONTROL", "control-private", 19401), ("EVAL", "eval-private", 19402), ("MODELS", "models-public", 19403)]
FAKES = {bucket: ("SYNTHETIC" + role + "ACCESS", "Synthetic" + role + "SigningSecret1234567890")
         for role, bucket, _ in ROLES}
PHASE = "initialization"
FAKE_DSTACK = "SyntheticDstackBearerForNativeProof"


class FakeServices(BaseHTTPRequestHandler):
    requests = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path.startswith("/repo.git/info/refs"):
            body = ("a" * 40 + "\trefs/heads/main\n").encode()
        elif self.path == "/repo.git/HEAD":
            body = b"ref: refs/heads/main\n"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        authorized = self.headers.get("Authorization") == "Bearer " + FAKE_DSTACK
        self.requests.append((self.path, authorized))
        if self.path == "/api/runs/list" and authorized:
            payload, status = [], 200
        elif self.path == "/api/2.0/mlflow/experiments/search":
            payload, status = {"experiments": []}, 200
        else:
            payload, status = {"error": "unreviewed fake request"}, 403
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class Response:
    def __init__(self, status, body=b"", headers=None):
        self.status, self.body, self.headers = status, io.BytesIO(body), headers or {}
        self.headers.setdefault("Content-Length", str(len(body)))

    def getheaders(self):
        return list(self.headers.items())

    def read(self, size=-1):
        return self.body.read(size)


class FakeUpstream:
    objects = {}
    requests = 0

    def __init__(self, host, **kwargs):
        assert host == HOST

    def request(self, method, target, body, headers):
        from botocore.auth import S3SigV4Auth
        from botocore.awsrequest import AWSRequest
        from botocore.credentials import Credentials
        from unittest.mock import patch

        parsed = urlsplit(target)
        bucket = parsed.path.split("/")[1]
        access, secret = FAKES[bucket]
        assert headers["authorization"].startswith("AWS4-HMAC-SHA256 Credential=" + access)
        moment = datetime.strptime(headers["x-amz-date"], "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        reference = AWSRequest(method=method, url="https://" + HOST + target, data=body or b"",
                               headers={k: v for k, v in headers.items() if k != "authorization"})
        with patch("botocore.auth.get_current_datetime", return_value=moment):
            S3SigV4Auth(Credentials(access, secret), "s3", "auto").add_auth(reference)
        assert headers["authorization"] == reference.headers["Authorization"]
        FakeUpstream.requests += 1
        key = parsed.path
        existing = self.objects.get(key)
        if key.endswith("/reflected-error"):
            self.response = Response(403, secret.encode())
        elif method == "PUT" and headers.get("x-amz-copy-source"):
            source = self.objects["/" + headers["x-amz-copy-source"].lstrip("/")]
            assert headers.get("x-amz-copy-source-if-match", "").strip('"') == source[1]
            self.objects[key] = source
            body = ('<CopyObjectResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                    '<ETag>"' + source[1] + '"</ETag></CopyObjectResult>').encode()
            self.response = Response(200, body, {"Content-Type": "application/xml"})
        elif method == "PUT":
            if (headers.get("if-none-match") == "*" and existing is not None
                or headers.get("if-match") and (existing is None or headers["if-match"].strip('"') != existing[1])):
                self.response = Response(412)
            else:
                body = body or b""
                etag = hashlib.sha256(body).hexdigest()
                metadata = {k: v for k, v in headers.items() if k.startswith("x-amz-meta-")}
                self.objects[key] = (body, etag, metadata)
                self.response = Response(200, headers={"ETag": '"' + etag + '"'})
        elif method == "GET" and "list-type" in parse_qs(parsed.query):
            rows = []
            for path, (data, _, _) in self.objects.items():
                if path.startswith("/" + bucket + "/"):
                    rows.append(f"<Contents><Key>{escape(path[len(bucket)+2:])}</Key><Size>{len(data)}</Size></Contents>")
            document = '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><IsTruncated>false</IsTruncated>' + "".join(rows) + "</ListBucketResult>"
            self.response = Response(200, document.encode(), {"Content-Type": "application/xml"})
        elif method == "DELETE":
            if headers.get("if-match") and (existing is None or headers["if-match"].strip('"') != existing[1]):
                self.response = Response(412)
            else:
                self.objects.pop(key, None)
                self.response = Response(204)
        elif existing is None:
            self.response = Response(404)
        else:
            data, etag, metadata = existing
            self.response = Response(200, data if method == "GET" else b"",
                                     {"ETag": '"' + etag + '"', "Content-Length": str(len(data)), **metadata})

    def getresponse(self):
        return self.response

    def close(self):
        pass


def child():
    global PHASE
    import training_vault as launcher
    import urllib.error
    import urllib.request
    from gradlab import operator_credentials
    from gradlab.r2_store import BucketConfig, R2Bucket

    def forbidden(*args, **kwargs):
        raise AssertionError("Source credential retrieval attempted")
    operator_credentials.read_secret = forbidden
    operator_credentials._keychain_lookup = forbidden
    PHASE = "storage-roundtrip"
    launcher.check_storage()
    PHASE = "dstack-actual-client"
    from gradlab.dstack_backend import DstackBackend
    DstackBackend(project="main", server_url="http://127.0.0.1:19404",
                  token="gradlab_dstack_vault_placeholder", environment=dict(os.environ)).preflight()
    PHASE = "mlflow-actual-client"
    from mlflow.tracking import MlflowClient
    assert not MlflowClient(tracking_uri="http://127.0.0.1:19405").search_experiments()
    PHASE = "git-actual-client"
    git = subprocess.run(["git", "-c", "credential.helper=", "ls-remote", "http://127.0.0.1:19406/repo.git"],
                         capture_output=True, text=True, timeout=20)
    assert git.returncode == 0 and "refs/heads/main" in git.stdout
    PHASE = "error-redaction"
    bucket = R2Bucket(BucketConfig.from_env("GRADLAB_MODELS_R2", public=True))
    try:
        bucket.get_bytes("reflected-error")
    except Exception as exc:
        assert not any(secret in str(exc) for _, secret in FAKES.values())
    else:
        raise AssertionError("Expected reflected upstream error")
    PHASE = "native-policy-denial"
    for url in ("http://127.0.0.1:19401/eval-private/foreign", "http://unmatched.invalid/blocked"):
        try:
            urllib.request.urlopen(url, timeout=10)
        except urllib.error.HTTPError as exc:
            assert exc.code == 403
        else:
            raise AssertionError("Native proxy did not reject an unmatched host/path")
    print('{"native_fake_client_verified":true}')
    PHASE = "self-revocation"
    from gradlab.training_vault import revoke_own_session
    revoke_own_session()


def worker_child():
    import training_vault as launcher
    import urllib.error
    import urllib.request
    from gradlab.training_vault import revoke_own_session, validate_session

    validate_session()
    launcher.check_storage()
    try:
        urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:19404/api/runs/list", data=b"{}"), timeout=10)
    except urllib.error.HTTPError as exc:
        assert exc.code == 403
    else:
        raise AssertionError("Worker must not reach dstack")
    revoke_own_session()
    import time
    deadline = time.monotonic() + 75
    while True:
        try:
            urllib.request.urlopen("http://127.0.0.1:19401/control-private/denied-after-revoke", timeout=10)
        except urllib.error.HTTPError as exc:
            if exc.code in {403, 407}:
                break
            assert exc.code == 404  # Cached native session until its next poll.
        if time.monotonic() >= deadline:
            raise AssertionError("Revoked capability was not rejected within native refresh bound")
        time.sleep(2)
    print('{"durable_worker_capability_verified":true}')


def main():
    global PHASE
    if sys.argv[1:] == ["--child"]:
        child()
        return
    if sys.argv[1:] == ["--worker-child"]:
        worker_child()
        return
    from training_vault_admin import Admin, CLI, INSTANCE, PROJECT
    import training_vault as launcher
    from gradlab import r2_signing_gateway as gateway

    gateway.HTTPSConnection = FakeUpstream
    servers, threads, services = [], [], []
    bundle = worker_bundle = None
    worker_session = None
    admin = Admin()
    source_pins = {name: launcher.digest(ROOT/name) for name in launcher.FILES}
    try:
        for _, bucket, port in ROLES:
            server = gateway.GatewayServer(("127.0.0.1", port), bucket=bucket, endpoint="https://" + HOST)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            servers.append(server)
            threads.append(thread)
        for port in (19404, 19405, 19406):
            server = ThreadingHTTPServer(("127.0.0.1", port), FakeServices)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            servers.append(server)
            threads.append(thread)
        PHASE = "create-fake-bundle"
        bundle = admin.create_bundle("gradlab-r2-native-fake-" + uuid.uuid4().hex[:12])
        worker_bundle = admin.create_bundle("gradlab-worker-native-fake-" + uuid.uuid4().hex[:12])
        for role, bucket, port in ROLES:
            PHASE = "create-fake-service"
            access, secret = FAKES[bucket]
            service = admin.create_service([bundle["id"], worker_bundle["id"]], {
                "type": "http", "name": "fake-r2-" + role.lower(),
                "hostPattern": f"127.0.0.1:{port}", "allowedMethods": ["GET", "HEAD", "PUT", "DELETE"],
                "allowedPathPrefixes": ["/" + bucket],
                "credential": {"type": "basic", "username": access, "password": secret},
            })
            services.append(service["id"])
        for name, paths, credential, host in (
            ("fake-training-dstack", ["/api/runs"], {"type": "bearer", "value": FAKE_DSTACK}, "127.0.0.1:19404"),
            ("fake-training-mlflow", ["/api/2.0/mlflow"], {"type": "passthrough"}, "127.0.0.1:19405"),
            ("fake-training-revoke", ["/api/v1/agent-vault/session/revoke"], {"type": "passthrough"}, "app.infisical.com"),
            ("fake-training-git", ["/repo.git"], {"type": "passthrough"}, "127.0.0.1:19406"),
        ):
            PHASE = "create-" + name
            service = admin.create_service([bundle["id"]] if name.endswith("dstack") else [bundle["id"], worker_bundle["id"]], {
                "type": "http", "name": name, "hostPattern": host, "allowedMethods": ["GET", "POST"] if name.endswith("git") else ["POST"],
                "allowedPathPrefixes": paths, "credential": credential,
            })
            services.append(service["id"])
        ca = Path.home() / ".config/infisical/agent-vault-ca.pem"
        der_hash = hashlib.sha256(__import__("ssl").PEM_cert_to_DER_cert(ca.read_text())).hexdigest()
        fingerprint = "SHA256:" + ":".join(der_hash[i:i+2].upper() for i in range(0, 64, 2))
        from gradlab.training_vault import CA_B64, CA_PIN, MARKERS, MODE, PROXY_ADDRESS
        environment = {
            **MARKERS, MODE: "agent-vault", PROXY_ADDRESS: "127.0.0.1:8283", CA_PIN: der_hash,
            CA_B64: base64.b64encode(ca.read_bytes()).decode(), "SSL_CERT_FILE": str(ca),
            "NO_PROXY": "", "no_proxy": "", "PYTHONPATH": str(ROOT/"src") + ":" + str(ROOT/"scripts"),
        }
        for role, _, port in ROLES:
            environment[f"GRADLAB_{role}_R2_ENDPOINT_URL"] = f"http://127.0.0.1:{port}"
        command = [str(CLI), "agent-vault", "run", "--domain=" + INSTANCE, "--project-id=" + PROJECT,
                   "--access-bundle=" + bundle["name"], "--proxy=127.0.0.1:8283", "--ca-fingerprint=" + fingerprint,
                   "--ttl=10m", "--no-ca-trust", "--no-skills", "--silent", "--telemetry=false", "--",
                   "/usr/bin/env", *[key + "=" + value for key, value in environment.items()],
                   str(ROOT/".venv/bin/python"), str(Path(__file__).resolve()), "--child"]
        PHASE = "native-child"
        process = subprocess.run(command, cwd=ROOT, env={"HOME": str(Path.home()), "PATH": str(Path.home()/".local/bin") + ":/usr/bin:/bin"},
                                 stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=180)
        output = process.stdout + process.stderr
        if process.returncode:
            for line in output.splitlines():
                if line.startswith('{"native_fake_failure":'):
                    print(line)
            print(json.dumps({"native_child_exit": process.returncode,
                              "fake_source_secret_reflected": any(secret in output for _, secret in FAKES.values()),
                              "session_capability_in_output": "agv_" in output}))
        if (process.returncode or '"native_fake_client_verified":true' not in output
            or "agv_" in output or any(secret in output for _, secret in FAKES.values())):
            raise RuntimeError("Native fake proof failed; child/provider output withheld")
        PHASE = "durable-worker-session"
        from training_vault_admin import native
        worker_session = json.loads(native(["agent-vault", "sessions", "create", "--project-id=" + PROJECT,
                                            "--access-bundle=" + worker_bundle["name"], "--ttl=10m", "-o", "json"]))
        # The creator command has exited. Only its finite worker capability is
        # inherited by this new process, exactly as for a queued supervisor.
        from urllib.parse import quote
        capability = "http://x-agent-vault:" + quote(worker_session["token"], safe="") + "@127.0.0.1:8283"
        worker_env = {"HOME": str(Path.home()), "PATH": "/usr/bin:/bin", **environment,
                      "HTTP_PROXY": capability, "HTTPS_PROXY": capability}
        process = subprocess.run([str(ROOT/".venv/bin/python"), str(Path(__file__).resolve()), "--worker-child"],
                                 env=worker_env, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=90)
        if process.returncode or '"durable_worker_capability_verified":true' not in process.stdout:
            for line in (process.stdout + process.stderr).splitlines():
                if line.startswith('{"native_fake_failure":'):
                    print(line)
            raise RuntimeError("Durable worker session proof failed; details withheld")
        worker_session = None  # The successful child proved its own revocation.
        assert not FakeUpstream.objects
        assert ("/api/runs/list", True) in FakeServices.requests
        assert any(path == "/api/2.0/mlflow/experiments/search" for path, _ in FakeServices.requests)
        proof = {"ok": True, "native_cli_version": "0.45.6", "native_basic_injection": True,
                 "actual_boto_client": True, "conditional_crud_and_hashes": True,
                 "bucket_path_and_host_denial": True, "reflected_upstream_key_redacted": True,
                 "source_credentials_retrieved": False, "fake_upstream_requests": FakeUpstream.requests,
                 "dstack_native_bearer_actual_cli": True, "mlflow_actual_client": True,
                 "git_actual_client": True,
                 "supervisor_session_self_revoke": True,
                 "worker_survives_creator_exit": True, "worker_cannot_reach_dstack": True,
                 "revoked_capability_denied": True,
                 "source_pins": source_pins}
        assert source_pins == {name: launcher.digest(ROOT/name) for name in launcher.FILES}
    finally:
        PHASE = "cleanup-worker-session"
        if worker_session is not None:
            native(["agent-vault", "sessions", "revoke", worker_session["id"]])
        PHASE = "cleanup-fake-services"
        failures = []
        for identifier in services:
            for attempt in range(3):
                try:
                    admin.api("DELETE", "services/" + identifier)
                    break
                except Exception:
                    if attempt == 2:
                        failures.append(identifier)
                    else:
                        import time
                        time.sleep(1)
        PHASE = "cleanup-fake-bundles"
        if bundle is not None:
            admin.api("DELETE", "access-bundles/" + bundle["id"])
        if worker_bundle is not None:
            admin.api("DELETE", "access-bundles/" + worker_bundle["id"])
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(5)
        assert not failures
    remaining = admin.api("GET", f"services?projectId={PROJECT}&limit=100")["services"]
    assert not set(services) & {item["id"] for item in remaining}
    assert not {bundle["id"], worker_bundle["id"]} & {item["id"] for item in admin.bundles()}
    proof["temporary_services_removed"] = True
    (OUT/"native-fake-proof.json").write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps({key: value for key, value in proof.items() if key != "source_pins"}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        import re
        frame = traceback.extract_tb(exc.__traceback__)[-1]
        print(json.dumps({"native_fake_failure": {"phase": PHASE, "type": type(exc).__name__,
                                                 "file": Path(frame.filename).name, "line": frame.lineno,
                                                 "category": getattr(exc, "category", None),
                                                 "http_status": (re.search(r"HTTP (\d{3})", str(exc)).group(1)
                                                                 if re.search(r"HTTP (\d{3})", str(exc)) else None)}}))
        raise SystemExit(1) from None
