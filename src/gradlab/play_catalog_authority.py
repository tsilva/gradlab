from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from gradlab.catalog_errors import CatalogIntegrityError, CatalogUnavailable
from gradlab.goal_catalog import GOAL_CATALOG_ROOT
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
CATALOG_EXECUTOR_ENV = "GRADLAB_CATALOG_EXECUTOR"
MAX_REQUEST_BYTES = 16 * 1024
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_BULK_READ_KEYS = 256
MAX_BULK_REQUEST_KEYS = 64
MAX_BULK_READ_WORKERS = 16

_ALLOWED_CONTROL_KEYS = (
    re.compile(rf"{re.escape(GOAL_CATALOG_ROOT)}/goals/[0-9a-f]{{64}}/current\.json"),
    re.compile(
        rf"{re.escape(GOAL_CATALOG_ROOT)}/goals/[0-9a-f]{{64}}/generations/"
        r"[0-9a-f]{64}\.json"
    ),
    re.compile(
        rf"{re.escape(GOAL_CATALOG_ROOT)}/goals/[0-9a-f]{{64}}/pages/"
        r"[0-9a-f]{64}\.json"
    ),
    re.compile(r"runs/gradlab-[0-9a-f]{32}/manifest\.json"),
    re.compile(
        r"runs/gradlab-[0-9a-f]{32}/monitoring/[0-9a-f]{64}/"
        r"[0-9a-f]{64}/state\.json"
    ),
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
            _stop_helper(self._process)


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
    executor = os.environ.get(CATALOG_EXECUTOR_ENV)
    if executor is not None:
        return _start_external_catalog(executor)
    if "INFISICAL_AGENT_VAULT_URL" in os.environ or "RUNBOX_POLICY" in os.environ:
        raise CatalogUnavailable("Configure an external catalog executor for this agent session")
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


def _start_external_catalog(executor: str) -> CatalogAuthorityClient:
    try:
        command = json.loads(executor)
    except ValueError:
        command = None
    if (
        not isinstance(command, list) or not command
        or any(not isinstance(arg, str) or not arg or "\0" in arg for arg in command)
        or not Path(command[0]).is_absolute()
    ):
        raise CatalogUnavailable("External catalog executor requires an absolute command array")
    with tempfile.TemporaryDirectory(prefix="glc-", dir="/tmp") as directory:
        endpoint = Path(directory).resolve() / "catalog"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(endpoint))
            listener.listen(1)
            listener.settimeout(0.25)
            try:
                process = subprocess.Popen(
                    [*command, str(endpoint)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env={
                        "HOME": str(Path.home()),
                        "PATH": "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
                    },
                    start_new_session=True,
                )
            except OSError:
                raise CatalogUnavailable("External catalog executor could not start") from None
            deadline = time.monotonic() + 25
            try:
                while True:
                    try:
                        connection, _ = listener.accept()
                        break
                    except TimeoutError:
                        if process.poll() is not None or time.monotonic() >= deadline:
                            raise CatalogUnavailable("External catalog executor did not connect") from None
            except BaseException:
                _stop_helper(process)
                raise
    connection.settimeout(55)
    client = CatalogAuthorityClient(connection, process=process)
    try:
        client.ready()
    except BaseException:
        client.close()
        raise
    return client


def _stop_helper(process: subprocess.Popen[bytes]) -> None:
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)


def scrub_protected_environment() -> None:
    for name in PROTECTED_ENV_NAMES:
        os.environ.pop(name, None)


def _scoped_control_bucket():
    from gradlab.operator_credentials import load_operator_environment
    from gradlab.r2_store import BucketConfig, R2Bucket
    from gradlab.training_vault import validate_session

    validate_session()
    load_operator_environment(requested_names={
        "GRADLAB_CONTROL_R2_URI", "GRADLAB_CONTROL_R2_REGION",
    })
    return R2Bucket(BucketConfig.from_env("GRADLAB_CONTROL_R2"))


def _serve(sock: socket.socket, repo_root: Path, *, scoped: bool = False) -> int:
    from gradlab.operator_environment import load_repository_operator_environment
    from gradlab.r2_store import BucketConfig, R2Bucket

    bucket = None
    startup_error: tuple[str, str] | None = None
    try:
        if scoped:
            bucket = _scoped_control_bucket()
        else:
            load_repository_operator_environment(
                repo_root,
                requested_names=CONTROL_R2_ENV_NAMES,
            )
            bucket = R2Bucket(BucketConfig.from_env("GRADLAB_CONTROL_R2"))
    except Exception as exc:
        startup_error = (
            "catalog_configuration",
            "Scoped catalog configuration unavailable" if scoped else str(exc),
        )

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
                    "message": (
                        "Scoped control catalog read failed"
                        if scoped else f"control catalog read failed: {exc}"
                    ),
                    "retryable": True,
                },
            )
        else:
            _send_json(sock, {"ok": True, "value": value})


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--serve-fd", type=int)
    operation.add_argument("--socket", type=Path)
    operation.add_argument("--probe-run")
    parser.add_argument("--repo-root", type=Path)
    return parser


def _serve_scoped_catalog(endpoint: Path) -> int:
    parent = endpoint.parent
    if (
        not endpoint.is_absolute() or endpoint.resolve() != endpoint
        or not parent.is_dir() or parent.stat().st_uid != os.getuid()
        or parent.stat().st_mode & 0o077 or not stat.S_ISSOCK(endpoint.stat().st_mode)
        or endpoint.stat().st_uid != os.getuid()
    ):
        raise ValueError("External catalog requires an owner-only local socket")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(45 * 60)
        connection.connect(str(endpoint))
        return _serve(connection, Path(__file__).resolve().parents[2], scoped=True)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.serve_fd is None:
        if args.repo_root is not None:
            parser.error("Scoped catalog commands use the selected checkout")
        if args.socket is not None:
            return _serve_scoped_catalog(args.socket)
        key = _allowed_control_key(f"runs/{args.probe_run}/manifest.json")
        from gradlab.r2_store import ObjectNotFound
        from gradlab.training_vault import SafeS3Error

        try:
            bucket = _scoped_control_bucket()
        except Exception:
            print(json.dumps({"ok": False, "stage": "session"}))
            return 1
        try:
            bucket.head(key)
        except (ObjectNotFound, SafeS3Error) as exc:
            status = 404 if isinstance(exc, ObjectNotFound) else (
                exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            )
            if type(status) is int and 400 <= status <= 599:
                print(f"runbox: Service request failed (HTTP {status}); details withheld", file=sys.stderr)
            else:
                print(json.dumps({"ok": False, "stage": "storage"}))
            return 1
        return 0
    if args.repo_root is None or any(name in os.environ for name in (
        "GRADLAB_TRAINING_TRANSPORT", "INFISICAL_AGENT_VAULT_URL", "RUNBOX_POLICY",
    )):
        parser.error("The standalone catalog helper requires a repository outside agent sessions")
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
