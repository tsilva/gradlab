"""Fail-closed native Agent Vault transport for queued training infrastructure."""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import os
import re
import ssl
import tempfile
import urllib.request
from collections.abc import Mapping, MutableMapping
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlsplit


MODE = "GRADLAB_TRAINING_TRANSPORT"
CA_B64 = "GRADLAB_VAULT_CA_B64"
CA_PIN = "GRADLAB_VAULT_CA_SHA256"
PROXY_ADDRESS = "GRADLAB_VAULT_PROXY_ADDRESS"
TOKEN = "GRADLAB_AGENT_VAULT_SESSION_TOKEN"
INSTANCE = "https://app.infisical.com"
ROLES = ("CONTROL", "EVAL", "MODELS")
MARKERS = {"DSTACK_TOKEN": "gradlab_dstack_vault_placeholder"} | {
    f"GRADLAB_{role}_R2_{field}": f"gradlab_{role.lower()}_r2_vault_placeholder"
    for role in ROLES for field in ("ACCESS_KEY_ID", "SECRET_ACCESS_KEY")
}
WORKER_CONTEXT = "GRADLAB_VAULT_WORKER_CONTEXT"
WORKER_CAPABILITY = "GRADLAB_VAULT_WORKER_CAPABILITY_TOKEN"


class TrainingVaultError(RuntimeError):
    pass


class TrainingVaultCommandError(RuntimeError):
    """A fixed diagnostic category without native provider text or capability."""

    def __init__(self, stdout: str, stderr: str):
        text = (stdout[:8192] + "\n" + stderr[:8192]).lower()
        self.category = "nativecall_error"
        for category, markers in (
            ("proxy_rejected", ("407 proxy", "proxy authentication required")),
            ("authentication_failed", ("401 unauthorized", "401 client error", "403 forbidden", "403 client error")),
            ("tls_failed", ("certificate verify failed", "certificate_verify_failed", "sslcertverificationerror", "tls handshake")),
            ("connection_failed", ("connection refused", "connectionrefusederror", "failed to establish a new connection", "connecttimeout", "readtimeout", "timed out", "name or service not known")),
        ):
            if any(marker in text for marker in markers):
                self.category = category
                break
        super().__init__("Training inventory request failed: " + self.category)


def selected(values: Mapping[str, str] | None = None) -> bool:
    values = os.environ if values is None else values
    if MODE not in values:
        return False
    if values[MODE] != "agent-vault":
        raise TrainingVaultError("Invalid training Vault mode; no credential fallback")
    return True


def validate_session(values: Mapping[str, str] | None = None) -> str:
    values = os.environ if values is None else values
    if not selected(values):
        raise TrainingVaultError("An explicit native training Vault session is required")
    for name, marker in MARKERS.items():
        if values.get(name) != marker:
            raise TrainingVaultError("Training Vault requires public placeholders; no credential fallback")
    if any(values.get(name) for name in (
        "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE",
        "WANDB_API_KEY", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "HF_TOKEN",
        "INFISICAL_TOKEN", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD",
    )):
        raise TrainingVaultError("Source credentials cannot enter the training Vault process")
    proxy = values.get("HTTPS_PROXY", "")
    parsed = urlsplit(proxy)
    expected = values.get(PROXY_ADDRESS, "")
    try:
        address = ipaddress.ip_address(parsed.hostname or "")
        allowed = address.is_loopback or address.is_private
        valid = (
            parsed.scheme == "http" and allowed and parsed.port
            and parsed.username == "x-agent-vault" and (parsed.password or "").startswith("agv_")
            and parsed.netloc.rsplit("@", 1)[-1] == expected
            and parsed.path in {"", "/"} and not parsed.query and not parsed.fragment
            and values.get("HTTP_PROXY") == proxy
            and all(not values.get(name) for name in ("NO_PROXY", "no_proxy"))
            and all(values.get(name, proxy) == proxy for name in ("http_proxy", "https_proxy"))
        )
    except ValueError:
        valid = False
    if not valid:
        raise TrainingVaultError("Expected the pinned native Vault proxy session; bypasses are rejected")
    try:
        ca = Path(values["SSL_CERT_FILE"])
        pem = ca.read_text()
        certificates = re.findall(r"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----", pem, re.S)
        valid_ca = (
            not ca.is_symlink() and "PRIVATE KEY" not in pem
            and values[CA_PIN] in {
                hashlib.sha256(ssl.PEM_cert_to_DER_cert(certificate)).hexdigest()
                for certificate in certificates
            }
        )
    except (KeyError, OSError, ValueError):
        valid_ca = False
    if not valid_ca or values.get("PYTHONHTTPSVERIFY") == "0":
        raise TrainingVaultError("Training Vault requires its verified public proxy CA")
    return proxy


def activate_public_ca(values: MutableMapping[str, str] | None = None) -> Path | None:
    """Materialize only public trust data in the task-owned temporary directory."""
    values = os.environ if values is None else values
    if not selected(values) or not values.get(CA_B64):
        return None
    values["NO_PROXY"] = values["no_proxy"] = ""
    if values.get("HTTP_PROXY") and values.get("HTTPS_PROXY") == values["HTTP_PROXY"]:
        values["http_proxy"] = values["https_proxy"] = values["HTTP_PROXY"]
    try:
        pem = base64.b64decode(values[CA_B64], validate=True).decode("ascii")
        if "PRIVATE KEY" in pem or hashlib.sha256(ssl.PEM_cert_to_DER_cert(pem)).hexdigest() != values[CA_PIN]:
            raise ValueError
    except (ValueError, KeyError, UnicodeError):
        raise TrainingVaultError("Invalid training Vault public CA") from None
    directory = Path(tempfile.mkdtemp(prefix="gradlab-vault-ca-"))
    path = directory / "public-ca.pem"
    path.write_text(pem)
    values.update({"SSL_CERT_FILE": str(path), "REQUESTS_CA_BUNDLE": str(path), "CURL_CA_BUNDLE": str(path)})
    return path


def install_placeholders(values: MutableMapping[str, str]) -> None:
    for name, marker in MARKERS.items():
        if values.get(name) not in {None, "", marker}:
            raise TrainingVaultError("Refusing raw training credentials in Vault mode")
        values[name] = marker


def gateway_endpoint(role: str, values: Mapping[str, str] | None = None) -> str:
    values = os.environ if values is None else values
    validate_session(values)
    if role not in ROLES:
        raise TrainingVaultError("Unreviewed R2 role")
    bucket = {"CONTROL": "control-private", "EVAL": "eval-private", "MODELS": "models-public"}[role]
    if values.get(f"GRADLAB_{role}_R2_URI") != "s3://" + bucket:
        raise TrainingVaultError("Training Vault requires the reviewed R2 bucket; no storage fallback")
    endpoint = values.get(f"GRADLAB_{role}_R2_ENDPOINT_URL", "")
    parsed = urlsplit(endpoint)
    if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1"
        or parsed.port != {"CONTROL": 19401, "EVAL": 19402, "MODELS": 19403}[role]
        or parsed.path not in {"", "/"} or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise TrainingVaultError("Expected the reviewed broker-side R2 gateway")
    return endpoint


def revoke_own_session(values: Mapping[str, str] | None = None) -> None:
    values = os.environ if values is None else values
    proxy = validate_session(values)
    token = unquote(urlsplit(proxy).password or "")
    request = urllib.request.Request(
        INSTANCE + "/api/v1/agent-vault/session/revoke", data=b"{}", method="POST",
        headers={"Authorization": "Bearer " + str(token), "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            result = json.load(response)
        if not isinstance(result.get("session"), dict) or not result["session"].get("revokedAt"):
            raise ValueError
    except Exception:
        raise TrainingVaultError("Training Vault session revocation failed; bounded expiry remains enforced") from None


def worker_context(values: Mapping[str, str] | None = None) -> dict | None:
    values = os.environ if values is None else values
    if not selected(values):
        return None
    validate_session(values)
    try:
        context = json.loads(values[WORKER_CONTEXT])
        if set(context) != {"session_id", "expires_at", "proxy_address", "secret_ref"}:
            raise ValueError
        if not re.fullmatch(r"[a-f0-9-]{36}", context["session_id"]):
            raise ValueError
        if context["secret_ref"] != "GRADLAB_VAULT_SESSION_" + context["session_id"].replace("-", "").upper():
            raise ValueError
        expiry = datetime.fromisoformat(context["expires_at"].replace("Z", "+00:00"))
        if expiry.tzinfo is None or not 0 < (expiry - datetime.now(UTC)).total_seconds() <= 24 * 3600:
            raise ValueError
        capability = values[WORKER_CAPABILITY]
        parsed = urlsplit(capability)
        if (parsed.netloc.rsplit("@", 1)[-1] != context["proxy_address"]
            or not re.fullmatch(r"172\.17\.0\.1:[1-9][0-9]{3,4}", context["proxy_address"])
            or parsed.scheme != "http" or parsed.username != "x-agent-vault"
            or parsed.path or parsed.query or parsed.fragment
            or not (parsed.password or "").startswith("agv_")):
            raise ValueError
        return context
    except (KeyError, TypeError, ValueError):
        raise TrainingVaultError("A launch requires one finite worker-owned Vault session") from None


def offline_learner_environment(values: Mapping[str, str]) -> dict[str, str]:
    environment = dict(values)
    if selected(values):
        for name in tuple(environment):
            if (name.startswith(("GRADLAB_VAULT_", "INFISICAL_AGENT_VAULT_"))
                or name in {MODE, TOKEN, "DSTACK_TOKEN"}):
                environment.pop(name, None)
        # The learner publishes only to its local journal. Retain a refusing
        # proxy configuration rather than passing its supervisor's capability.
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            environment[name] = "http://127.0.0.1:9"
        environment["NO_PROXY"] = environment["no_proxy"] = ""
    return environment


class SafeS3Error(TrainingVaultError):
    def __init__(self, error):
        response = getattr(error, "response", {})
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        self.response = {"ResponseMetadata": {"HTTPStatusCode": status}}
        super().__init__("Brokered R2 request failed; provider details withheld")


class SafeS3Client:
    """Preserve status semantics without leaking proxy capabilities in errors."""

    def __init__(self, client):
        self.client = client

    def __getattr__(self, name):
        if name == "get_paginator":
            return self.get_paginator
        if name == "generate_presigned_url":
            return self.generate_presigned_url
        method = getattr(self.client, name)

        def call(*args, **kwargs):
            try:
                result = method(*args, **kwargs)
                if isinstance(result, dict) and isinstance(result.get("Metadata"), dict):
                    # HTTP proxies may canonicalize x-amz-meta-* field casing.
                    # S3 user-metadata keys have lowercase wire semantics.
                    result["Metadata"] = {key.lower(): value for key, value in result["Metadata"].items()}
                return result
            except Exception as exc:
                raise SafeS3Error(exc) from None
        return call

    def get_paginator(self, operation):
        client = self.client

        class Paginator:
            def paginate(self, **kwargs):
                try:
                    yield from client.get_paginator(operation).paginate(**kwargs)
                except Exception as exc:
                    raise SafeS3Error(exc) from None
        return Paginator()

    def generate_presigned_url(self, ClientMethod, *, Params, ExpiresIn, **kwargs):
        from urllib.parse import quote, urlencode

        method = {"get_object": "GET", "put_object": "PUT"}.get(ClientMethod)
        if method is None or kwargs or set(Params) - {"Bucket", "Key", "ContentType"}:
            raise TrainingVaultError("Unreviewed broker-side signing operation")
        endpoint = self.client.meta.endpoint_url.rstrip("/")
        path = "/" + quote(Params["Bucket"], safe="") + "/" + quote(Params["Key"], safe="/._-~")
        query = {"__gradlab_presign": method, "expires": str(ExpiresIn)}
        if Params.get("ContentType"):
            query["content_type"] = Params["ContentType"]
        try:
            with urllib.request.urlopen(endpoint + path + "?" + urlencode(query), timeout=20) as response:
                payload = json.load(response)
            result = urlsplit(payload["url"])
            if result.scheme != "https" or not result.hostname.endswith(".r2.cloudflarestorage.com"):
                raise ValueError
            return payload["url"]
        except Exception:
            raise TrainingVaultError("Broker-side R2 signing failed; details withheld") from None
