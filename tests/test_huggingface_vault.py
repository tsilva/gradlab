"""Vault mode must fail closed before reading a token cache or running a helper."""

import ssl
from unittest.mock import patch

import pytest

from gradlab import huggingface_vault as guard
from gradlab import publication_credentials as credentials

launcher = guard

def session(tmp_path):
    ca = tmp_path / "public-ca.pem"
    ca.write_text("synthetic-public-certificate")
    fingerprint = __import__("hashlib").sha256(b"synthetic-der").hexdigest().upper()
    return {
        "GRADLAB_HUGGINGFACE_TRANSPORT": "agent-vault", "HF_TOKEN": guard.MARKER,
        "HTTP_PROXY": "http://session:agv_fixture@127.0.0.1:12345",
        "HTTPS_PROXY": "http://session:agv_fixture@127.0.0.1:12345",
        "GRADLAB_VAULT_PROXY_ADDRESS": "127.0.0.1:12345",
        "GRADLAB_VAULT_CA_FINGERPRINT": "SHA256:" + fingerprint,
        "SSL_CERT_FILE": str(ca), "HF_HUB_DISABLE_XET": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
    }


def test_valid_vault_mode_uses_only_placeholder_without_cache(tmp_path):
    values = session(tmp_path)
    with patch.object(ssl, "PEM_cert_to_DER_cert", return_value=b"synthetic-der"), \
            patch.object(credentials, "_read_private_text", side_effect=AssertionError("cache read")):
        result = credentials.resolve_huggingface_credential(values)
    assert result.token == guard.MARKER and result.source == "agent-vault"


@pytest.mark.parametrize("change", [
    {"GRADLAB_HUGGINGFACE_TRANSPORT": ""}, {"HF_TOKEN": "fake-raw-key"}, {"HF_TOKEN": ""},
    {"HF_TOKEN_PATH": "/fake/token"}, {"HF_STORED_TOKENS_PATH": "/fake/stored"},
    {"HTTPS_PROXY": "http://session:agv_fixture@example.invalid:12345"},
    {"http_proxy": "http://different.invalid"}, {"NO_PROXY": "*"},
    {"GRADLAB_VAULT_PROXY_ADDRESS": "127.0.0.1:9999"}, {"GRADLAB_VAULT_CA_FINGERPRINT": "changed"},
    {"HF_HUB_DISABLE_XET": "0"}, {"HF_DEBUG": "1"}, {"PYTHONHTTPSVERIFY": "0"},
    {"R2_SECRET_ACCESS_KEY": "fake-private"},
])
def test_invalid_vault_session_never_falls_back(tmp_path, change):
    values = {**session(tmp_path), **change}
    with patch.object(ssl, "PEM_cert_to_DER_cert", return_value=b"synthetic-der"), \
            patch.object(credentials, "_read_private_text", side_effect=AssertionError("cache read")), \
            pytest.raises(credentials.CredentialSecurityError, match="no credential fallback"):
        credentials.resolve_huggingface_credential(values)


@pytest.mark.parametrize("args", [
    ["auth", "login"], ["auth", "token"], ["jobs", "run"], ["update"], ["download", "owner/model"],
    ["upload", "owner/model", "--token=fake-key"], ["repo-files", "delete", "owner/model"],
    ["upload", "owner/model", "file", "--repo-type=space"],
])
def test_unreviewed_commands_and_raw_token_arguments_rejected(args):
    with pytest.raises(ValueError):
        launcher.cli_arguments(args)


def test_upload_payload_root_is_explicit_and_canonical(tmp_path):
    with patch.object(launcher, "ROOT", tmp_path):
        allowed = tmp_path / "weights.bin"
        allowed.write_bytes(b"synthetic model bytes")
        assert launcher.cli_arguments(["upload", "fixture/test", str(allowed)])[2] == str(allowed)
        with pytest.raises(ValueError):
            launcher.cli_arguments(["upload", "fixture/test", str(tmp_path)])
        private = tmp_path / ".local/credential"
        private.parent.mkdir()
        private.write_text("synthetic-only")
        with pytest.raises(ValueError):
            launcher.cli_arguments(["upload", "fixture/test", str(private)])
