"""Validate an explicitly selected, credential-free Hugging Face Vault session."""

import hashlib
import ipaddress
import os
import re
import sys
import ssl
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit

MARKER = "hf_gradlab_proxy_placeholder"


def validate_session(values: Mapping[str, str]) -> None:
    """Reject raw keys, cache fallback, bypasses and unreviewed TLS/session settings."""
    if values.get("GRADLAB_HUGGINGFACE_TRANSPORT") != "agent-vault":
        raise ValueError("Expected the reviewed Hugging Face Vault transport")
    if values.get("HF_TOKEN") != MARKER:
        raise ValueError("Expected only the Hugging Face public placeholder")
    if any(values.get(name) for name in (
        "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH", "HF_STORED_TOKENS_PATH",
        "INFISICAL_TOKEN", "AWS_SECRET_ACCESS_KEY", "R2_SECRET_ACCESS_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS", "WANDB_API_KEY",
    )):
        raise ValueError("Source credentials and token-cache overrides are not allowed")
    proxy = values.get("HTTPS_PROXY", "")
    parsed = urlsplit(proxy)
    expected = values.get("GRADLAB_VAULT_PROXY_ADDRESS", "")
    if (parsed.scheme != "http" or not parsed.hostname or not parsed.port
        or not ipaddress.ip_address(parsed.hostname).is_loopback
        or parsed.netloc.rsplit("@", 1)[-1] != expected
        or not parsed.username or not (parsed.password or "").startswith("agv_")
        or parsed.path not in ("", "/") or parsed.query or parsed.fragment
        or values.get("HTTP_PROXY") != proxy):
        raise ValueError("Expected the reviewed matching loopback proxy session")
    if any(values.get(name) for name in ("NO_PROXY", "no_proxy")) or any(
        values.get(name, proxy) != proxy for name in ("http_proxy", "https_proxy")
    ):
        raise ValueError("Proxy bypasses and conflicting proxy settings are rejected")
    if (values.get("HF_HUB_DISABLE_XET") != "1"
        or values.get("HF_HUB_DISABLE_TELEMETRY") != "1"
        or values.get("PYTHONHTTPSVERIFY") == "0"
        or values.get("HF_DEBUG") not in (None, "", "0")):
        raise ValueError("Expected reviewed transfer, telemetry and TLS settings")
    ca_path = Path(values.get("SSL_CERT_FILE", ""))
    pem = ca_path.read_text()
    if ca_path.is_symlink() or "PRIVATE KEY" in pem:
        raise ValueError("Expected only the reviewed public CA")
    fingerprint = hashlib.sha256(ssl.PEM_cert_to_DER_cert(pem)).hexdigest().upper()
    expected_fingerprint = values.get("GRADLAB_VAULT_CA_FINGERPRINT", "")
    if fingerprint != expected_fingerprint.removeprefix("SHA256:").replace(":", ""):
        raise ValueError("The public CA differs from the reviewed pin")


ROOT = Path(__file__).resolve().parents[2]


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
        path = Path(args[2])
        path = (path if path.is_absolute() else ROOT / path).absolute().resolve(strict=True)
        relative = path.relative_to(ROOT)
        if not relative.parts or relative.parts[0] in {".git", ".local", ".venv"}:
            raise ValueError("Unreviewed payload root")
        if relative.parts[0] == ".internal" and not path.is_relative_to(
            ROOT / ".internal/huggingface-vault/payloads"
        ):
            raise ValueError("Private working notes are not upload payloads")
        args = [*args[:2], str(path), *args[3:]]
    return args




def child(args):
    import contextlib
    import io
    import ssl

    from gradlab.publication_credentials import resolve_huggingface_credential

    credential = resolve_huggingface_credential()
    if credential.token != MARKER or credential.source != "agent-vault":
        raise ValueError("Expected the actual publisher's Vault credential resolver")
    import httpx2 as httpx
    from huggingface_hub import HfApi, set_client_factory

    class Transport(httpx.BaseTransport):
        def __init__(self):
            self.inner = httpx.HTTPTransport(proxy=os.environ["HTTPS_PROXY"],
                                            verify=ssl.create_default_context(cafile=os.environ["SSL_CERT_FILE"]), retries=0)

        def handle_request(self, request):
            if request.url.path == "/api/agent-harnesses":
                raise ValueError("Agent discovery is outside the reviewed route")
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

            if rest[:2] == ["repo-files", "ls"]:
                import argparse
                parser = argparse.ArgumentParser(add_help=False)
                parser.add_argument("repo_id")
                parser.add_argument("--repo-type", choices=("model", "dataset"), default="model")
                parser.add_argument("--revision", default="main")
                selected = parser.parse_args(rest[2:])
                HfApi(endpoint=os.environ["HF_ENDPOINT"], token=credential.token).list_repo_files(
                    selected.repo_id, repo_type=selected.repo_type, revision=selected.revision)
            else:
                # The public CLI rejects caller token overrides. The selected
                # SDK still needs its inert marker explicitly for upload auth.
                supplied = [*rest, "--token", credential.token] if rest[:1] == ["upload"] else rest
                app(args=supplied, standalone_mode=False)
        print('{"ok":true,"operation":"hub-cli","credential_mode":"public-placeholder"}')
    else:
        raise ValueError("Unreviewed child operation")




def main(arguments=None):
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    if arguments == ["check"]:
        child(arguments)
    else:
        child(["hf", *cli_arguments(arguments)])


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        print('{"ok":false,"reason":"Hugging Face operation unavailable; details withheld"}')
        raise SystemExit(1) from None
