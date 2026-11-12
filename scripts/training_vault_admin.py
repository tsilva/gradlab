"""Trusted training Vault setup; never import this from ordinary tool startup.

Administrative/source credentials stay in this process's memory and HTTPS
request bodies. Output contains selected metadata only. No token files or raw
credential-bearing subprocess arguments/environments are permitted.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
import resource
import ssl
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INSTANCE = "https://app.infisical.com"
PROJECT = "5078c850-3f93-46d8-8afe-82107df8b931"
SOURCE_PROJECT = "8e69bfc0-24e8-4234-9ece-78e425ac4a8f"
CLI = Path.home() / ".config/infisical/cli/0.45.6/infisical"
CLI_SHA256 = "4f98044ecc959c7f71ce608c23092163c68a67c0aa8720dc46c552702207af6b"
LINUX_SHA256 = "df0eb7cfd3ccaa3f7ec30b31e0ba0931ef8e876bc63220ff74810653e0027fcf"
LINUX_ARCHIVE_SHA256 = "b427391e76113e351fa0509d6493cb400577dec26b28afb462c2c2f9c5a37d4d"
LINUX_ARCHIVE_URL = "https://github.com/Infisical/cli/releases/download/v0.45.6/cli_0.45.6_linux_amd64.tar.gz"
EVIDENCE = ROOT / ".internal/evidence/training-vault-migration-20261112"
PRIVATE = Path.home() / ".config/gradlab/agent-vault"
OPERATOR = Path.home() / ".config/gradlab/operator.toml"
PROXY_NAME = "gradlab-training-beast3"
BOOTSTRAP_PATH = "/agents/training-proxy"
BOOTSTRAP_SECRET = "AGENT_VAULT_PROXY_TOKEN"
INSTALLATION = "/opt/gradlab-training-vault"
ROLES = (("CONTROL", "control-private", 19401), ("EVAL", "eval-private", 19402), ("MODELS", "models-public", 19403))
ENDPOINT = "https://83e1ce4a70ea388693e8525a772ccefa.r2.cloudflarestorage.com"


class SetupError(RuntimeError):
    pass


def native(arguments):
    if hashlib.sha256(CLI.read_bytes()).hexdigest() != CLI_SHA256:
        raise SetupError("Trusted native CLI changed")
    try:
        result = subprocess.run(
            [str(CLI), *arguments, "--domain=" + INSTANCE, "--silent", "--telemetry=false"],
            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60,
            env={"HOME": str(Path.home()), "PATH": "/usr/bin:/bin"}, cwd="/",
        )
        if result.returncode:
            raise SetupError("Infisical access failed; renew the existing login. Provider output withheld.")
        return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        raise SetupError("Infisical CLI unavailable; details withheld") from None


class Admin:
    def __init__(self):
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        self.token = native(["user", "get", "token", "--plain"])
        if not self.token:
            raise SetupError("Infisical login renewal required")

    def request(self, method, path, payload=None):
        request = urllib.request.Request(
            INSTANCE + path,
            method=method, data=None if payload is None else json.dumps(payload).encode(),
            headers={"Authorization": "Bearer " + self.token, "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read()
                return json.loads(data) if data else {}
        except urllib.error.HTTPError as exc:
            raise SetupError(f"Agent Vault administration rejected request (HTTP {exc.code}); details withheld") from None
        except (OSError, ValueError) as exc:
            error = SetupError("Agent Vault administration failed; details withheld")
            error.category = type(exc).__name__
            raise error from None

    def api(self, method, path, payload=None):
        return self.request(method, "/api/v1/agent-vault/" + path, payload)

    def bundles(self):
        return self.api("GET", f"access-bundles?projectId={PROJECT}&limit=100")["accessBundles"]

    def create_bundle(self, name):
        if any(item["name"] == name for item in self.bundles()):
            raise SetupError("Refusing to overwrite an existing access bundle")
        return self.api("POST", "access-bundles", {"projectId": PROJECT, "name": name})["accessBundle"]

    def create_service(self, bundle_ids, service):
        return self.api("POST", "services", {
            "projectId": PROJECT, "accessBundleIds": list(bundle_ids), **service,
        })["service"]


def source(name, *, environment="prod", path="/"):
    return native(["secrets", "get", name, "--plain", "--projectId=" + SOURCE_PROJECT,
                   "--env=" + environment, "--path=" + path, "--expand=false",
                   "--include-imports=false", "--secret-overriding=false"])


def write_private(path, data):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.parent.is_symlink() or path.parent.stat().st_mode & 0o077:
        raise SetupError("Expected protected private metadata directory")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


def ssh(arguments, *, packet=None, timeout=60):
    result = subprocess.run(["ssh", "beast-3", *arguments],
                            input=packet, capture_output=True, timeout=timeout)
    if result.returncode:
        raise SetupError("Trusted beast-3 operation failed; details withheld")
    return result.stdout


def validate_source():
    """Validate references as data before the seven exact source reads."""
    if OPERATOR.is_symlink() or OPERATOR.stat().st_mode & 0o022:
        raise SetupError("Unprotected operator source binding")
    p = tomllib.loads(OPERATOR.read_text())
    if p["schema_version"] != 3:
        raise SetupError("Unsupported operator source schema")
    names = [f"GRADLAB_{role}_R2_{field}" for role, _, _ in ROLES
             for field in ("ACCESS_KEY_ID", "SECRET_ACCESS_KEY")]
    references = {name: p["infisical"][name] for name in names}
    b3 = p["dstack"]["coordinators"]["b3"]
    references["DSTACK_B3_TOKEN"] = b3["token"]
    for name, reference in references.items():
        if reference != {"instance": INSTANCE, "project_id": SOURCE_PROJECT, "environment": "prod",
                         "path": "/", "secret_name": name}:
            raise SetupError("Unreviewed training service source scope")
    if b3["project"] != "main" or b3["server_url"] != "http://127.0.0.1:3000":
        raise SetupError("Unreviewed coordinator binding")
    for role, bucket, _ in ROLES:
        if (p["environment"][f"GRADLAB_{role}_R2_URI"] != "s3://" + bucket
            or p["environment"][f"GRADLAB_{role}_R2_ENDPOINT_URL"] != ENDPOINT
            or p["environment"][f"GRADLAB_{role}_R2_REGION"] != "auto"):
            raise SetupError("Unreviewed R2 account or bucket binding")
    if p["environment"]["MLFLOW_TRACKING_URI"] != "https://192.168.0.151:5443":
        raise SetupError("Unreviewed MLflow binding")
    return p


def require_fake_proof():
    import training_vault as launcher

    proof = json.loads((EVIDENCE / "native-fake-proof.json").read_text())
    if (proof.get("ok") is not True or proof.get("source_credentials_retrieved") is not False
        or proof.get("temporary_services_removed") is not True
        or proof.get("worker_survives_creator_exit") is not True
        or proof.get("native_cli_version") != "0.45.6"
        or proof.get("source_pins") != {name: launcher.digest(ROOT / name) for name in launcher.FILES}):
        raise SetupError("Current runtime requires a successful native fake-credential proof")


def provision():
    require_fake_proof()
    operator = validate_source()
    if (PRIVATE / "catalog.json").exists():
        raise SetupError("Training catalog already exists; use its reviewed bootstrap operation")
    admin = Admin()
    if any(x["name"] == PROXY_NAME for x in admin.api("GET", f"proxies?projectId={PROJECT}&limit=100")["proxies"]):
        raise SetupError("Refusing to replace an existing training proxy")
    # Check the bootstrap destination before creating a one-time proxy token.
    folders = admin.request("GET", "/api/v2/folders?projectId=" + SOURCE_PROJECT + "&environment=dev&path=/agents")["folders"]
    if any(x["name"] == "training-proxy" for x in folders):
        raise SetupError("Bootstrap folder exists without a reviewed local receipt")
    admin.request("POST", "/api/v2/folders", {"projectId": SOURCE_PROJECT, "environment": "dev",
                                             "path": "/agents", "name": "training-proxy"})
    catalog = {"version": 1, "source_project": SOURCE_PROJECT, "source_environment": "prod",
               "source_path": "/", "services": [], "bundles": [], "proxy_id": None}
    receipt = PRIVATE / "catalog.json"

    def save():
        if receipt.exists():
            # Only this task's metadata receipt is replaced, never secret values.
            temporary = PRIVATE / "catalog.pending.json"
            write_private(temporary, (json.dumps(catalog, indent=2) + "\n").encode())
            temporary.replace(receipt)
        else:
            write_private(receipt, (json.dumps(catalog, indent=2) + "\n").encode())

    save()
    for suffix in ("operator", "worker"):
        bundle = admin.create_bundle("local-gradlab-training-" + suffix)
        catalog["bundles"].append({"id": bundle["id"], "name": bundle["name"]})
        save()
    ids = [x["id"] for x in catalog["bundles"]]

    def service(name, host, methods, paths, credential, *, worker=True):
        s = admin.create_service(ids if worker else ids[:1], {
            "type": "http", "name": name, "hostPattern": host, "allowedMethods": methods,
            "allowedPathPrefixes": paths, "credential": credential,
        })
        expected = set(ids if worker else ids[:1])
        if {row["id"] for row in s["accessBundles"]} != expected:
            raise SetupError("Created service has unexpected access-bundle membership")
        catalog["services"].append({"id": s["id"], "name": s["name"], "host": host,
                                    "methods": methods, "paths": paths, "bundle_ids": sorted(expected)})
        save()

    for role, bucket, port in ROLES:
        service("gradlab-training-r2-" + role.lower(), f"127.0.0.1:{port}", ["GET", "HEAD", "PUT", "DELETE"],
                ["/" + bucket], {"type": "basic", "username": source(f"GRADLAB_{role}_R2_ACCESS_KEY_ID"),
                                 "password": source(f"GRADLAB_{role}_R2_SECRET_ACCESS_KEY")})
    service("gradlab-training-dstack-b3", "127.0.0.1:3000", ["GET", "POST"], ["/api"],
            {"type": "bearer", "value": source("DSTACK_B3_TOKEN")}, worker=False)
    service("gradlab-training-mlflow", "192.168.0.151:5443", ["GET", "POST"], ["/api/2.0/mlflow"],
            {"type": "passthrough"})
    service("gradlab-training-public-models", "pub-fc35c0b186ce4aad8eea5e93d38c99db.r2.dev", ["GET", "HEAD"], ["/"],
            {"type": "passthrough"})
    service("gradlab-training-session-revoke", "app.infisical.com", ["POST"], ["/api/v1/agent-vault/session/revoke"],
            {"type": "passthrough"})
    # Runtime receipts retain the separately approved local gh login. No GitHub
    # source key is copied into Vault, and workers have no GitHub service.
    service("gradlab-training-github-api", "api.github.com", ["GET", "POST"], ["/repos/tsilva/gradlab", "/graphql"],
            {"type": "passthrough"}, worker=False)
    service("gradlab-training-runtime-artifact", "*.blob.core.windows.net", ["GET"], ["/"],
            {"type": "passthrough"}, worker=False)
    service("gradlab-training-github-git", "github.com", ["GET", "POST"], ["/tsilva/gradlab.git"],
            {"type": "passthrough"}, worker=False)
    response = admin.api("POST", "proxies", {"projectId": PROJECT, "name": PROXY_NAME,
                                              "trafficPolicy": "service-hosts", "allowedHosts": None, "pollInterval": 10})
    proxy = response["proxy"]
    catalog["proxy_id"] = proxy["id"]
    save()
    token = response.pop("proxyToken")
    # Persist encrypted recovery immediately; never write the bootstrap to disk.
    admin.request("POST", "/api/v4/secrets/" + BOOTSTRAP_SECRET, {
        "projectId": SOURCE_PROJECT, "environment": "dev", "secretPath": BOOTSTRAP_PATH,
        "secretValue": token, "type": "shared",
    })
    catalog["token_sha256"] = hashlib.sha256(token.encode()).hexdigest()
    ca = admin.api("GET", "proxies/" + proxy["id"] + "/ca")["ca"]
    pem = ca["certificate"].encode()
    ssl.PEM_cert_to_DER_cert(pem.decode())
    write_private(PRIVATE / "training-ca.pem", pem)
    catalog["ca_fingerprint"] = ca["fingerprint"]
    save()
    deploy(catalog, operator)
    bootstrap_token(catalog, token)
    refresh_profile()
    print(json.dumps({"ok": True, "catalog_services": len(catalog["services"]),
                      "separate_bundles": True, "proxy_id": proxy["id"], "source_keys_in_tool_environment": False}))


def deploy(catalog, operator):
    binary = EVIDENCE / "downloads/infisical-linux-amd64"
    if hashlib.sha256(binary.read_bytes()).hexdigest() != LINUX_SHA256:
        raise SetupError("Linux native CLI checksum mismatch")
    system_ca = ssh(["cat", "/etc/ssl/certs/ca-certificates.crt"])
    mlflow_ca = base64.b64decode(operator["environment"]["GRADLAB_MLFLOW_TLS_CA_B64"], validate=True)
    ssl.PEM_cert_to_DER_cert(mlflow_ca.decode())
    files = {"infisical": binary.read_bytes(), "gateway.py": (ROOT / "src/gradlab/r2_signing_gateway.py").read_bytes(),
             "broker.py": (ROOT / "scripts/training_vault_broker.py").read_bytes(),
             "upstream-ca.pem": system_ca + b"\n" + mlflow_ca + b"\n"}
    broker_profile = {"version": 1, "files": {k: hashlib.sha256(v).hexdigest() for k, v in files.items()},
                      "proxy_id": catalog["proxy_id"], "token_sha256": catalog["token_sha256"]}
    files["profile.json"] = (json.dumps(broker_profile, indent=2) + "\n").encode()
    installer = '''import base64,gzip,hashlib,io,json,os,pathlib,sys,tarfile,urllib.request
p=pathlib.Path("/opt/gradlab-training-vault")
if os.geteuid()!=0 or p.exists(): raise SystemExit(1)
packet=json.load(sys.stdin)
if set(packet)!={"gateway.py","broker.py","upstream-ca.pem","profile.json","cli_url","archive_sha256","binary_sha256"}: raise SystemExit(1)
url=packet.pop("cli_url"); archive_sha=packet.pop("archive_sha256"); binary_sha=packet.pop("binary_sha256")
if url!="https://github.com/Infisical/cli/releases/download/v0.45.6/cli_0.45.6_linux_amd64.tar.gz": raise SystemExit(1)
with urllib.request.urlopen(url,timeout=120) as r: archive=r.read(100*1024*1024)
if hashlib.sha256(archive).hexdigest()!=archive_sha: raise SystemExit(1)
with tarfile.open(fileobj=io.BytesIO(archive),mode="r:gz") as t:
 member=t.getmember("infisical")
 if not member.isfile() or member.size>300*1024*1024: raise SystemExit(1)
 binary=t.extractfile(member).read()
if hashlib.sha256(binary).hexdigest()!=binary_sha: raise SystemExit(1)
p.mkdir(mode=0o700)
fd=os.open(p/"infisical",os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o700)
with os.fdopen(fd,"wb") as h: h.write(binary)
for name,data in packet.items():
 f=p/name
 fd=os.open(f,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,"wb") as h: h.write(gzip.decompress(base64.b64decode(data,validate=True)))
profile=json.loads((p/"profile.json").read_text())
if any(hashlib.sha256((p/n).read_bytes()).hexdigest()!=v for n,v in profile["files"].items()): raise SystemExit(1)
print("broker_runtime_installed")
'''
    import shlex
    payload = {name: base64.b64encode(gzip.compress(data, compresslevel=1)).decode()
               for name, data in files.items() if name != "infisical"}
    payload.update({"cli_url": LINUX_ARCHIVE_URL, "archive_sha256": LINUX_ARCHIVE_SHA256,
                    "binary_sha256": LINUX_SHA256})
    packet = json.dumps(payload).encode()
    ssh(["sudo", "-n", "python3", "-c", shlex.quote(installer)], packet=packet, timeout=180)
    catalog["broker_source_pins"] = broker_profile["files"]
    (PRIVATE / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")


def bootstrap_token(catalog, token):
    if hashlib.sha256(token.encode()).hexdigest() != catalog["token_sha256"]:
        raise SetupError("Proxy bootstrap changed; trusted rotation review required")
    ssh(["sudo", "-n", "python3", INSTALLATION + "/broker.py", "start"],
        packet=json.dumps({"proxy_token": token}).encode())
    # Public metadata proves the running proxy serves the exact pinned CA.
    import training_vault as launcher
    connection = {"proxy_local": "127.0.0.1:28383", "proxy_remote": "172.17.0.1:28383", "ssh_host": "beast-3"}
    owned = launcher.forward(connection)
    try:
        for _ in range(30):
            try:
                with urllib.request.urlopen("http://127.0.0.1:28383/_agent-vault/ca", timeout=2) as response:
                    running = json.load(response)
                if running["proxyId"] != catalog["proxy_id"] or running["fingerprint"] != catalog["ca_fingerprint"]:
                    raise SetupError("Running broker identity differs from provisioned proxy")
                return
            except (OSError, ValueError):
                time.sleep(1)
        raise SetupError("Broker bootstrap did not become ready")
    finally:
        if owned:
            subprocess.run(["ssh", "-O", "cancel", "-L", owned, "beast-3"], capture_output=True, timeout=15)


def refresh_profile():
    require_fake_proof()
    validate_source()
    import training_vault as launcher
    catalog = json.loads((PRIVATE / "catalog.json").read_text())
    pem = (PRIVATE / "training-ca.pem").read_text()
    ca_sha = hashlib.sha256(ssl.PEM_cert_to_DER_cert(pem)).hexdigest()
    python = (ROOT / ".venv/bin/python").resolve()
    p = {"version": 1, "repo_path": str(ROOT), "instance": INSTANCE, "project_id": PROJECT,
         "operator_bundle": "local-gradlab-training-operator", "worker_bundle": "local-gradlab-training-worker",
         "proxy_local": "127.0.0.1:28383", "proxy_remote": "172.17.0.1:28383", "ssh_host": "beast-3",
         "ca_file": str(PRIVATE / "training-ca.pem"), "ca_sha256": ca_sha,
         "ca_file_sha256": launcher.digest(PRIVATE / "training-ca.pem"), "ca_fingerprint": catalog["ca_fingerprint"],
         "cli": str(CLI), "cli_sha256": CLI_SHA256, "python": str(python), "python_sha256": launcher.digest(python),
         "site_packages": str(ROOT / ".venv/lib/python3.14/site-packages"), "operator_file": str(OPERATOR),
         "operator_sha256": launcher.digest(OPERATOR), "source_pins": {name: launcher.digest(ROOT / name) for name in launcher.FILES},
         "coordinator_id": "b3"}
    path = PRIVATE / "training.json"
    if path.exists():
        path.unlink()
    write_private(path, (json.dumps(p, indent=2) + "\n").encode())
    print('{"training_profile_pins_reviewed":true}')


def inventory():
    admin = Admin()
    bundles = admin.bundles()
    proxies = admin.api("GET", f"proxies?projectId={PROJECT}&limit=100")["proxies"]
    print(json.dumps({
        "ok": True, "project_id": PROJECT,
        "training_bundles": [{"id": x["id"], "name": x["name"]} for x in bundles if "gradlab" in x["name"]],
        "proxies": [{"id": x["id"], "name": x["name"], "traffic_policy": x.get("trafficPolicy")} for x in proxies],
        "source_credentials_retrieved": False,
    }))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["inventory", "provision", "install", "bootstrap", "refresh-profile"])
    args = parser.parse_args()
    if args.operation == "inventory":
        inventory()
    elif args.operation == "provision":
        provision()
    elif args.operation == "refresh-profile":
        refresh_profile()
    else:
        require_fake_proof()
        operator = validate_source()
        catalog = json.loads((PRIVATE / "catalog.json").read_text())
        if args.operation == "install":
            deploy(catalog, operator)
        bootstrap_token(catalog, source(BOOTSTRAP_SECRET, environment="dev", path=BOOTSTRAP_PATH))
        refresh_profile()
        print('{"broker_bootstrap_verified":true}')


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc) if isinstance(exc, SetupError) else "Training Vault setup failed; details withheld", file=sys.stderr)
        sys.exit(1)
