"""Validate an explicitly selected, credential-free Hugging Face Vault session."""

import hashlib
import ipaddress
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
