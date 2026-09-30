"""Install an operator supplied private CA for a queued MLflow service."""

from __future__ import annotations

import base64
import binascii
import os
import ssl
import tempfile
from pathlib import Path


def _system_ca_bundle() -> Path:
    return Path(ssl.get_default_verify_paths().cafile or "/etc/ssl/certs/ca-certificates.crt")


def validate_private_mlflow_ca_b64(encoded: str) -> bytes:
    try:
        certificate = base64.b64decode(encoded, validate=True)
        ssl.PEM_cert_to_DER_cert(certificate.decode("ascii"))
    except (ValueError, UnicodeError, binascii.Error, ssl.SSLError) as exc:
        raise ValueError("GRADLAB_MLFLOW_TLS_CA_B64 must contain one PEM certificate") from exc
    if certificate.count(b"-----BEGIN CERTIFICATE-----") != 1 or certificate.count(
        b"-----END CERTIFICATE-----"
    ) != 1:
        raise ValueError("GRADLAB_MLFLOW_TLS_CA_B64 must contain one PEM certificate")
    return certificate


def configure_private_mlflow_ca() -> Path | None:
    encoded = str(os.environ.get("GRADLAB_MLFLOW_TLS_CA_B64") or "").strip()
    if not encoded:
        return None
    certificate = validate_private_mlflow_ca_b64(encoded)
    system_bundle = _system_ca_bundle()
    if not system_bundle.is_file():
        raise RuntimeError("private MLflow CA requires the system CA bundle")
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix="gradlab-mlflow-ca-", suffix=".pem", delete=False
    ) as handle:
        handle.write(system_bundle.read_bytes())
        handle.write(b"\n")
        handle.write(certificate)
        handle.write(b"\n")
        bundle = Path(handle.name)
    os.environ["REQUESTS_CA_BUNDLE"] = str(bundle)
    os.environ["SSL_CERT_FILE"] = str(bundle)
    os.environ.pop("GRADLAB_MLFLOW_TLS_CA_B64", None)
    return bundle
