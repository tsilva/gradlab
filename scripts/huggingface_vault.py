"""Opt-in native Vault launcher for standalone Hugging Face CLI operations."""

import hashlib
import ipaddress
import json
import os
import re
import signal
import stat
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
PROFILE = Path.home() / ".config/infisical/huggingface-clients/gradlab.json"
VAULT_PROJECT = "5078c850-3f93-46d8-8afe-82107df8b931"
SOURCE_PROJECT = "8e69bfc0-24e8-4234-9ece-78e425ac4a8f"
MARKER = "hf_gradlab_proxy_placeholder"
NATIVE = ["version: 1", "general:", "domain: https://app.infisical.com",
          "secrets-management:", "project-id: " + SOURCE_PROJECT,
          "environment: dev", "secret-path: /agents/huggingface"]
FILES = {".infisical.yaml", "pyproject.toml", "uv.lock",
         "scripts/huggingface_vault.py", "scripts/huggingface_vault.sb"} | {
             "src/gradlab/" + name + ".py" for name in (
                 "__init__", "publication_credentials", "huggingface_vault", "file_utils",
                 "json_utils", "local_paths", "infisical_credentials", "secret_reference")}
SDK_FILES = {"huggingface_hub/" + name for name in (
    "hf_api.py", "utils/_http.py", "utils/_auth.py", "cli/hf.py", "cli/upload.py")}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate metadata field")
        result[key] = value
    return result


def profile():
    for path in (PROFILE.parent, PROFILE):
        info = path.lstat()
        if path.resolve() != path or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("Expected protected metadata")
    if not stat.S_ISREG(PROFILE.lstat().st_mode):
        raise ValueError("Expected regular metadata")
    p = json.loads(PROFILE.read_text(), object_pairs_hook=unique_object)
    expected = {"version": 1, "repository": "gradlab", "repo_path": str(ROOT),
                "instance": "https://app.infisical.com", "source_project": SOURCE_PROJECT,
                "source_environment": "dev", "source_path": "/agents/huggingface",
                "agent_vault_project_id": VAULT_PROJECT, "bundle": "local-gradlab-huggingface"}
    fields = {"code_sha256", "infisical", "cli_sha256", "python", "python_sha256",
              "site_packages", "proxy", "ca_file", "ca_sha256", "ca_fingerprint", "sdk_sha256",
              "expected_identity"}
    if (set(p) != set(expected) | fields or type(p["version"]) is not int
        or any(p[k] != v for k, v in expected.items()) or set(p["code_sha256"]) != FILES):
        raise ValueError("Unreviewed service binding")
    if not isinstance(p["expected_identity"], str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", p["expected_identity"]):
        raise ValueError("Unreviewed provider identity")
    for name, checksum in p["code_sha256"].items():
        file = ROOT / name
        if file.is_symlink() or digest(file) != checksum:
            raise ValueError("Source changed; trusted review required")
    lines = [line.strip() for line in (ROOT / ".infisical.yaml").read_text().splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if lines != NATIVE or any((ROOT / name).exists() or (ROOT / name).is_symlink() for name in (
        ".infisical.json", ".access.toml", ".access.local.toml")):
        raise ValueError("Native scope changed or conflicts with legacy metadata")
    for field, checksum in (("infisical", "cli_sha256"), ("python", "python_sha256"),
                            ("ca_file", "ca_sha256")):
        path = Path(p[field])
        if not path.is_absolute() or path.resolve() != path or digest(path) != p[checksum]:
            raise ValueError("Runtime or public CA changed")
    if Path(p["infisical"]).parent.name != "0.45.6":
        raise ValueError("Use the reviewed native CLI version")
    site = Path(p["site_packages"])
    if site != ROOT / ".venv/lib/python3.14/site-packages" or not site.is_dir():
        raise ValueError("Unreviewed SDK runtime")
    if set(p["sdk_sha256"]) != SDK_FILES:
        raise ValueError("Incomplete SDK review pins")
    for name, checksum in p["sdk_sha256"].items():
        path = site / name
        if path.is_symlink() or not path.resolve().is_relative_to(site.resolve()) or digest(path) != checksum:
            raise ValueError("SDK source changed")
    proxy = urlsplit("http://" + p["proxy"])
    if (not proxy.hostname or not ipaddress.ip_address(proxy.hostname).is_loopback
        or not proxy.port or proxy.username or proxy.password or proxy.path or proxy.query or proxy.fragment
        or not re.fullmatch(r"SHA256:(?:[0-9A-F]{2}:){31}[0-9A-F]{2}", p["ca_fingerprint"])):
        raise ValueError("Unreviewed local proxy")
    return p


def cli_arguments(args):
    if args == ["auth", "whoami"]:
        return args
    if not args or args[0] not in {"upload", "repo-files"}:
        raise ValueError("Only reviewed Hub file operations and auth whoami are supported")
    if args[0] == "repo-files" and args[1:2] != ["ls"]:
        raise ValueError("Only repo-files ls is supported")
    forbidden = {"--token", "--endpoint", "--every", "--local-dir", "--cache-dir"}
    if any(arg.split("=", 1)[0] in forbidden for arg in args):
        raise ValueError("No token, endpoint, scheduler or output-root overrides")
    for index, arg in enumerate(args):
        if arg == "--repo-type" and (index + 1 == len(args) or args[index + 1] not in {"model", "dataset"}):
            raise ValueError("Only models and datasets are supported")
        if arg.startswith("--repo-type=") and arg.split("=", 1)[1] not in {"model", "dataset"}:
            raise ValueError("Only models and datasets are supported")
    if args[0] == "upload":
        if len(args) < 3 or args[1].startswith("-") or args[2].startswith("-"):
            raise ValueError("Upload requires an explicit repository and local payload")
        path = Path(args[2]).absolute().resolve(strict=True)
        relative = path.relative_to(ROOT)
        if not relative.parts or relative.parts[0] in {".git", ".local", ".venv"}:
            raise ValueError("Unreviewed payload root")
        if relative.parts[0] == ".internal" and not path.is_relative_to(
            ROOT / ".internal/huggingface-vault/payloads"
        ):
            raise ValueError("Private working notes are not upload payloads")
        args = [*args[:2], str(path), *args[3:]]
    return args


def command(p, workspace, mode, arguments=(), *, endpoint="https://huggingface.co", client=None):
    port = urlsplit("http://" + p["proxy"]).port
    env = ["NO_PROXY=", "no_proxy=", "HF_TOKEN=" + MARKER,
           "GRADLAB_HUGGINGFACE_TRANSPORT=agent-vault", "GRADLAB_VAULT_PROXY_ADDRESS=" + p["proxy"],
           "GRADLAB_VAULT_CA_FINGERPRINT=" + p["ca_fingerprint"],
           "GRADLAB_HF_EXPECTED_IDENTITY=" + p["expected_identity"],
           "SSL_CERT_FILE=" + p["ca_file"], "HF_HOME=" + str(workspace / ".hf"),
           "TMPDIR=" + str(workspace), "HF_HUB_DISABLE_XET=1", "HF_HUB_DISABLE_TELEMETRY=1",
           "HF_HUB_DISABLE_IMPLICIT_TOKEN=1", "HF_DEBUG=0", "HF_ENDPOINT=" + endpoint,
           "GRADLAB_HF_SITE_PACKAGES=" + p["site_packages"]]
    sandbox = ["/usr/bin/sandbox-exec"]
    for key, value in {"USER_HOME": Path.home(), "REPO": ROOT, "PYTHON": p["python"],
                       "RUNTIME": Path(p["python"]).parents[1], "VENV": ROOT / ".venv",
                       "WORKSPACE": workspace, "CA": p["ca_file"], "PROXY_SOCKET": f"localhost:{port}",
                       "INPUT": arguments[2] if mode == "hf" and arguments[:1] == ["upload"] else ROOT / "scripts/huggingface_vault.py"}.items():
        sandbox += ["-D", key + "=" + str(value)]
    sandbox += ["-f", str(ROOT / "scripts/huggingface_vault.sb")]
    return [p["infisical"], "agent-vault", "run", "--domain=" + p["instance"],
            "--project-id=" + p["agent_vault_project_id"], "--access-bundle=" + p["bundle"],
            "--proxy=" + p["proxy"], "--ca-fingerprint=" + p["ca_fingerprint"], "--ttl=30m",
            "--no-ca-trust", "--no-skills", "--silent", "--telemetry=false", "--",
            "/usr/bin/env", *env, *sandbox, p["python"], "-I", "-S", "-B",
            str(client or ROOT / "scripts/huggingface_vault.py"), "--child", mode, *arguments]


def child(args):
    import contextlib
    import io
    import ssl

    site = ROOT / ".venv/lib/python3.14/site-packages"
    if os.environ.get("GRADLAB_HF_SITE_PACKAGES") != str(site):
        raise ValueError("Unreviewed site path")
    sys.path[:0] = [str(ROOT / "src"), str(site)]
    from gradlab.publication_credentials import resolve_huggingface_credential

    credential = resolve_huggingface_credential()
    if credential.token != MARKER or credential.source != "agent-vault":
        raise ValueError("Expected the actual publisher's Vault credential resolver")
    import httpx
    from huggingface_hub import HfApi, set_client_factory

    class Transport(httpx.BaseTransport):
        def __init__(self):
            self.inner = httpx.HTTPTransport(proxy=os.environ["HTTPS_PROXY"],
                                            verify=ssl.create_default_context(cafile=os.environ["SSL_CERT_FILE"]), retries=0)

        def handle_request(self, request):
            if re.search(r"/info/lfs/|/xet/|/xet-(?:read|write)-token(?:/|$)", request.url.path):
                raise ValueError("Large-file transport needs separate compatibility review")
            response = self.inner.handle_request(request)
            if response.status_code >= 400:
                response.close()
                return httpx.Response(response.status_code,
                                      json={"error": "Upstream request failed; details withheld"})
            return response

        def close(self):
            self.inner.close()

    set_client_factory(lambda: httpx.Client(transport=Transport(), timeout=30, follow_redirects=False))
    who = HfApi(endpoint=os.environ["HF_ENDPOINT"], token=credential.token).whoami()
    if who.get("name") != os.environ.get("GRADLAB_HF_EXPECTED_IDENTITY"):
        raise ValueError("Unexpected Hub identity")
    if args == ["check"]:
        print('{"ok":true,"read_only":true,"credential_mode":"public-placeholder","publisher_resolver":true}')
    elif args[:1] == ["hf"]:
        rest = cli_arguments(args[1:])
        # Native CLI console output and provider errors are never forwarded.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from huggingface_hub.cli.hf import app

            app(args=rest, standalone_mode=False)
        print('{"ok":true,"operation":"hub-cli","credential_mode":"public-placeholder"}')
    else:
        raise ValueError("Unreviewed child operation")


def main(args):
    if args[:1] == ["--child"]:
        child(args[1:])
        return
    if args == ["check"]:
        mode, rest = "check", []
    elif args[:1] == ["hf"]:
        mode, rest = "hf", cli_arguments(args[1:])
    else:
        raise ValueError("Use check or hf followed by a reviewed Hub file operation")
    p = profile()
    import tempfile

    output = ROOT / ".internal/huggingface-vault"
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    if output.resolve() != output or output.stat().st_uid != os.getuid():
        raise ValueError("Unreviewed artifact workspace")
    with tempfile.TemporaryDirectory(prefix="run-", dir=output) as name:
        workspace = Path(name).resolve()
        process = subprocess.Popen(command(p, workspace, mode, rest), cwd=workspace,
                                   env={"HOME": str(Path.home()), "PATH": "/usr/bin:/bin"},
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, start_new_session=True)
        try:
            stdout, _ = process.communicate(timeout=300)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.communicate(timeout=10)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate(timeout=5)
            raise
        records = [json.loads(line) for line in stdout.splitlines() if line.startswith('{"ok":')]
        if process.returncode or len(records) != 1 or records[0].get("ok") is not True:
            raise ValueError("Vault operation failed; no credential fallback")
        print(json.dumps(records[0]))


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except BaseException:
        print('{"ok":false,"reason":"Hugging Face Vault operation unavailable; details withheld"}')
        sys.exit(1)
