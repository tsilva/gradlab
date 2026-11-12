"""Protected beast-3 broker. Bootstrap arrives over stdin and stays in memory.

The trusted administrator installs this file beside the verified native CLI,
gateway module and public upstream CA bundle. No credential cache is written.
After a reboot, start it again through training_vault_admin.py bootstrap.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import resource
import signal
import stat
import subprocess
import sys
import threading
from pathlib import Path


PREFIX = Path("/opt/gradlab-training-vault")
PID = Path("/run/gradlab-training-vault.pid")
INSTANCE = "https://app.infisical.com"
ROLES = (("control-private", 19401), ("eval-private", 19402), ("models-public", 19403))
ENDPOINT = "https://83e1ce4a70ea388693e8525a772ccefa.r2.cloudflarestorage.com"


def protected(path, directory=False):
    info = path.lstat()
    kind = stat.S_ISDIR if directory else stat.S_ISREG
    if path.resolve() != path or not kind(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o077:
        raise ValueError("Unprotected broker runtime")


def profile():
    if os.geteuid() != 0:
        raise ValueError("Broker requires its trusted root installation")
    protected(PREFIX, directory=True)
    protected(PREFIX / "profile.json")
    p = json.loads((PREFIX / "profile.json").read_text())
    if (set(p) != {"version", "files", "proxy_id", "token_sha256"} or p["version"] != 1
        or set(p["files"]) != {"infisical", "gateway.py", "broker.py", "upstream-ca.pem"}):
        raise ValueError("Unreviewed broker metadata")
    for name, digest in p["files"].items():
        path = PREFIX / name
        protected(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Broker runtime changed")
    return p


def serve(token):
    spec = importlib.util.spec_from_file_location("trusted_r2_gateway", PREFIX / "gateway.py")
    gateway = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gateway)
    servers = []
    stopped = threading.Event()
    process = None

    def stop(*_):
        stopped.set()
        if process is not None and process.poll() is None:
            process.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for bucket, port in ROLES:
            server = gateway.GatewayServer(("127.0.0.1", port), bucket=bucket, endpoint=ENDPOINT)
            servers.append(server)
            threading.Thread(target=server.serve_forever, daemon=True).start()
        environment = {
            "HOME": str(PREFIX), "PATH": "/usr/bin:/bin", "SSL_CERT_FILE": str(PREFIX / "upstream-ca.pem"),
            "INFISICAL_AGENT_VAULT_PROXY_TOKEN": token,
        }
        command = [str(PREFIX / "infisical"), "agent-vault", "proxy", "--domain=" + INSTANCE,
                   "--listen-host=172.17.0.1", "--port=28383", "--data-dir=/dev/null",
                   "--log-level=error", "--silent", "--telemetry=false"]
        while not stopped.is_set():
            process = subprocess.Popen(command, env=environment, cwd=PREFIX,
                                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)
            process.wait()
            stopped.wait(5)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for server in servers:
            server.shutdown()
            server.server_close()
        PID.unlink(missing_ok=True)


def main():
    p = profile()
    if sys.argv[1:] == ["status"]:
        pid = int(PID.read_text())
        os.kill(pid, 0)
        print(json.dumps({"broker_running": True, "pid": pid, "proxy_id": p["proxy_id"]}))
        return
    if sys.argv[1:] != ["start"]:
        raise ValueError("Use start or status")
    packet = json.loads(sys.stdin.buffer.read(32768))
    token = packet.pop("proxy_token")
    if (packet or not isinstance(token, str)
        or hashlib.sha256(token.encode()).hexdigest() != p["token_sha256"]):
        raise ValueError("Unreviewed bootstrap capability")
    if PID.exists():
        # Never kill or replace an existing process, including an uncertain PID.
        raise ValueError("Broker PID already exists; inspect before recovery")
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.umask(0o077)
    pid = os.fork()
    if pid:
        print(json.dumps({"broker_bootstrap_accepted": True, "pid": pid}))
        return
    os.setsid()
    os.chdir(PREFIX)
    null = os.open(os.devnull, os.O_RDWR)
    for descriptor in (0, 1, 2):
        os.dup2(null, descriptor)
    if null > 2:
        os.close(null)
    try:
        with PID.open("x") as handle:
            handle.write(str(os.getpid()))
        serve(token)
    finally:
        os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("Training Vault broker unavailable; details withheld", file=sys.stderr)
        sys.exit(1)
