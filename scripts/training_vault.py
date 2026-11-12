"""Run reviewed training/operator commands with native Agent Vault sessions."""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import ssl
import stat
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
PROFILE = Path.home() / ".config/gradlab/agent-vault/training.json"
INSTANCE = "https://app.infisical.com"
PROJECT = "5078c850-3f93-46d8-8afe-82107df8b931"
FILES = {
    ".infisical.yaml", "pyproject.toml", "uv.lock", "scripts/training_vault.py",
    *{"src/gradlab/" + name + ".py" for name in (
        "training_vault", "r2_store", "r2_signing_gateway", "operator_credentials",
        "operator_environment", "dstack_backend", "experiment_cli", "run_supervisor",
    )},
}


class LaunchError(RuntimeError):
    pass


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def profile():
    for path in (PROFILE.parent, PROFILE):
        info = path.lstat()
        if path.resolve() != path or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise LaunchError("Training Vault metadata must be canonical and owner-only")
    if not stat.S_ISREG(PROFILE.lstat().st_mode):
        raise LaunchError("Training Vault profile must be a regular file")
    p = json.loads(PROFILE.read_text())
    fields = {
        "version", "repo_path", "instance", "project_id", "operator_bundle", "worker_bundle",
        "proxy_local", "proxy_remote", "ssh_host", "ca_file", "ca_sha256", "ca_file_sha256",
        "ca_fingerprint", "cli", "cli_sha256", "python", "python_sha256", "site_packages",
        "operator_file", "operator_sha256", "source_pins", "coordinator_id",
    }
    if (set(p) != fields or type(p["version"]) is not int or p["version"] != 1 or p["repo_path"] != str(ROOT)
        or p["instance"] != INSTANCE or p["project_id"] != PROJECT
        or p["operator_bundle"] != "local-gradlab-training-operator"
        or p["worker_bundle"] != "local-gradlab-training-worker"
        or set(p["source_pins"]) != FILES):
        raise LaunchError("Unreviewed training Vault binding")
    for name, checksum in p["source_pins"].items():
        path = ROOT / name
        if path.is_symlink() or digest(path) != checksum:
            raise LaunchError("Training Vault source changed; trusted review required")
    for key, checksum in (("cli", "cli_sha256"), ("python", "python_sha256"),
                          ("ca_file", "ca_file_sha256"), ("operator_file", "operator_sha256")):
        path = Path(p[key])
        if not path.is_absolute() or path.resolve() != path or digest(path) != p[checksum]:
            raise LaunchError("Training Vault runtime, public CA or operator binding changed")
    if Path(p["cli"]).parent.name != "0.45.6" or p["site_packages"] != str(ROOT / ".venv/lib/python3.14/site-packages"):
        raise LaunchError("Unreviewed training Vault runtime")
    pem = Path(p["ca_file"]).read_text()
    if "PRIVATE KEY" in pem or hashlib.sha256(ssl.PEM_cert_to_DER_cert(pem)).hexdigest() != p["ca_sha256"]:
        raise LaunchError("Invalid training Vault public CA")
    if (not re.fullmatch(r"127\.0\.0\.1:[1-9][0-9]{3,4}", p["proxy_local"])
        or not re.fullmatch(r"172\.17\.0\.1:[1-9][0-9]{3,4}", p["proxy_remote"])
        or not re.fullmatch(r"[A-Za-z0-9_.-]+", p["ssh_host"])):
        raise LaunchError("Unreviewed training Vault connection")
    return p


def native(p, args):
    result = subprocess.run(
        [p["cli"], *args, "--domain=" + INSTANCE, "--silent", "--telemetry=false"],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60,
        env={"HOME": str(Path.home()), "PATH": "/usr/bin:/bin"}, cwd="/",
    )
    if result.returncode:
        raise LaunchError("Native Vault operation failed; check login and binding. Details withheld.")
    return result.stdout


def forward(p):
    host, port = p["proxy_local"].split(":")
    try:
        with socket.create_connection((host, int(port)), timeout=1):
            return None
    except OSError:
        specification = p["proxy_local"] + ":" + p["proxy_remote"]
        result = subprocess.run(["ssh", "-O", "forward", "-L", specification, p["ssh_host"]],
                                capture_output=True, timeout=15)
        if result.returncode:
            raise LaunchError("Managed SSH forwarding unavailable; use FleetMan recovery")
        return specification


def check_storage():
    import uuid
    from gradlab.r2_store import ConditionalWriteConflict, ObjectNotFound, R2Bucket, RunStorageConfig

    from gradlab.operator_environment import load_repository_operator_environment
    load_repository_operator_environment(ROOT)
    storage = RunStorageConfig.from_env()
    key = "vault-verification/" + uuid.uuid4().hex
    payload = b"GradLab scoped Agent Vault R2 verification\n" * 32
    checked = []
    for role in ("control", "evaluation", "models"):
        bucket = R2Bucket(getattr(storage, role))
        created = False
        copy_key = key + "/copy"
        try:
            etag = bucket.put_bytes(key, payload, metadata={"sha256": hashlib.sha256(payload).hexdigest()})
            created = True
            assert bucket.get_bytes(key) == payload
            assert bucket.head(key)["metadata"]["sha256"] == hashlib.sha256(payload).hexdigest()
            try:
                bucket.put_bytes(key, b"conflicting synthetic bytes")
            except ConditionalWriteConflict:
                pass
            else:
                raise LaunchError("R2 conditional create did not reject conflicting bytes")
            changed = payload + b"conditional replacement\n"
            etag = bucket.put_bytes(key, changed, create_only=False, if_match=etag)
            assert bucket.get_bytes(key) == changed
            bucket.copy_within(key, copy_key)
            assert bucket.get_bytes(copy_key) == changed
            bucket.delete(copy_key)
            assert key in list(bucket.iter_keys("vault-verification/"))
            bucket.delete(key, if_match=etag)
            created = False
            try:
                bucket.head(key)
            except ObjectNotFound:
                pass
            else:
                raise LaunchError("R2 verification cleanup is incomplete")
            checked.append(role)
        finally:
            bucket.delete(copy_key)
            if created:
                bucket.delete(key)
    print(json.dumps({"ok": True, "r2_roles": checked, "conditional_writes": True,
                      "hashes_verified": True, "probe_objects_removed": True}))


def child(args):
    site = ROOT / ".venv/lib/python3.14/site-packages"
    sys.path[:0] = [str(ROOT / "src"), str(site)]
    from gradlab.training_vault import MODE, validate_session
    if os.environ.get(MODE) != "agent-vault":
        raise LaunchError("Native training Vault mode is required")
    validate_session()
    if args == ["check"]:
        check_storage()
    else:
        import runpy
        sys.argv = ["gradlab", *args]
        runpy.run_module("gradlab.main", run_name="__main__")


def main(args):
    if args[:1] == ["--child"]:
        child(args[1:])
        return
    if args != ["check"] and (len(args) < 2 or args[0] != "experiment" or args[1] not in {
        "operator-preflight", "launch", "status", "follow", "logs", "cancel", "retry", "resume-submit",
    }):
        raise LaunchError("Use check or a reviewed experiment lifecycle operation")
    p = profile()
    owned_forward = None
    worker = None
    bound = False
    try:
        owned_forward = forward(p)
        from urllib.parse import urlsplit
        ca_pem = Path(p["ca_file"]).read_bytes()
        import base64
        env = {
            "NO_PROXY": "", "no_proxy": "", "PYTHONHTTPSVERIFY": "1",
            "GRADLAB_TRAINING_TRANSPORT": "agent-vault", "GRADLAB_VAULT_PROXY_ADDRESS": p["proxy_local"],
            "GRADLAB_VAULT_CA_SHA256": p["ca_sha256"],
            "GRADLAB_VAULT_CA_B64": base64.b64encode(ca_pem).decode(),
            "SSL_CERT_FILE": p["ca_file"], "REQUESTS_CA_BUNDLE": p["ca_file"], "CURL_CA_BUNDLE": p["ca_file"],
            "GRADLAB_VAULT_COORDINATOR_ID": p["coordinator_id"],
        }
        sys.path.insert(0, str(ROOT / "src"))
        from gradlab.training_vault import MARKERS, WORKER_CAPABILITY, WORKER_CONTEXT
        env.update(MARKERS)
        for role, port in (("CONTROL", 19401), ("EVAL", 19402), ("MODELS", 19403)):
            env[f"GRADLAB_{role}_R2_ENDPOINT_URL"] = f"http://127.0.0.1:{port}"
        if args[:2] in (["experiment", "launch"], ["experiment", "retry"]):
            worker = json.loads(native(p, ["agent-vault", "sessions", "create", "--project-id=" + PROJECT,
                                         "--access-bundle=" + p["worker_bundle"], "--ttl=7h",
                                         "--label=gradlab-durable-training-attempt", "-o", "json"]))
            capability = "http://x-agent-vault:" + quote(worker["token"], safe="") + "@" + p["proxy_remote"]
            context = {"session_id": worker["id"], "expires_at": worker["expiresAt"],
                       "proxy_address": p["proxy_remote"], "secret_ref": "GRADLAB_VAULT_SESSION_" + worker["id"].replace("-", "").upper()}
            if not (urlsplit(capability).password or "").startswith("agv_"):
                raise LaunchError("Unexpected native session capability")
            env.update({WORKER_CONTEXT: json.dumps(context), WORKER_CAPABILITY: capability})
            if "--json" not in args:
                args = [*args, "--json"]
        command = [p["cli"], "agent-vault", "run", "--domain=" + INSTANCE,
                   "--project-id=" + PROJECT, "--access-bundle=" + p["operator_bundle"],
                   "--proxy=" + p["proxy_local"], "--ca-fingerprint=" + p["ca_fingerprint"],
                   "--ttl=" + ("8h" if args[:2] == ["experiment", "follow"] else "45m"),
                   "--no-ca-trust", "--no-skills", "--silent", "--telemetry=false", "--",
                   "/usr/bin/env", *[name + "=" + value for name, value in env.items()
                                       if name not in {WORKER_CAPABILITY}],
                   p["python"], "-I", "-S", "-B", str(ROOT / "scripts/training_vault.py"), "--child", *args]
        # A worker capability is a scoped session, but still never put it in argv.
        clean = {"HOME": str(Path.home()), "PATH": str(ROOT / ".venv/bin") + ":" + str(Path.home()/".local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"}
        if worker is not None:
            clean[WORKER_CAPABILITY] = env[WORKER_CAPABILITY]
        process = subprocess.Popen(command, cwd=ROOT, env=clean, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        for line in process.stdout:
            line = line.rstrip("\n")
            if line.startswith("{"):
                try:
                    row = json.loads(line)
                except ValueError:
                    row = {}
                if row.get("event") == "vault_session_bound":
                    bound = True
                    continue
            # Provider errors may include a proxy URL. Never expose capabilities.
            safe = re.sub(r"https?://[^\s/]+@", "http://<vault-session>@", line)
            safe = re.sub(r"agv_[A-Za-z0-9_.%-]+", "<vault-session>", safe)
            print(safe, flush=True)
        process.wait()
        if process.returncode:
            print("Training Vault command failed; provider details withheld", file=sys.stderr)
            raise SystemExit(process.returncode)
    finally:
        if worker is not None and not bound:
            native(p, ["agent-vault", "sessions", "revoke", worker["id"]])
        if owned_forward is not None:
            subprocess.run(["ssh", "-O", "cancel", "-L", owned_forward, p["ssh_host"]],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except (LaunchError, OSError, ValueError, KeyError):
        print("Training Vault unavailable; check the reviewed profile, login and route. No fallback.", file=sys.stderr)
        sys.exit(1)
