import hashlib
import ssl
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from gradlab import training_vault as vault
from gradlab.operator_credentials import load_operator_environment
from gradlab.r2_store import BucketConfig


@pytest.fixture
def session(tmp_path, monkeypatch):
    ca = tmp_path / "public-ca.pem"
    ca.write_text("-----BEGIN CERTIFICATE-----\nfixture\n-----END CERTIFICATE-----\n")
    monkeypatch.setattr(ssl, "PEM_cert_to_DER_cert", lambda value: b"synthetic-public-der")
    return {
        vault.MODE: "agent-vault", **vault.MARKERS,
        "HTTP_PROXY": "http://x-agent-vault:agv_syntheticCapability@127.0.0.1:17323",
        "HTTPS_PROXY": "http://x-agent-vault:agv_syntheticCapability@127.0.0.1:17323",
        vault.PROXY_ADDRESS: "127.0.0.1:17323", "SSL_CERT_FILE": str(ca),
        vault.CA_PIN: hashlib.sha256(b"synthetic-public-der").hexdigest(),
        "GRADLAB_CONTROL_R2_URI": "s3://control-private",
        "GRADLAB_CONTROL_R2_ENDPOINT_URL": "http://127.0.0.1:19401",
        "GRADLAB_VAULT_COORDINATOR_ID": "test-coordinator",
    }


def test_r2_configuration_selects_unsigned_broker_transport_with_only_markers(session):
    assert vault.validate_session(session) == session["HTTP_PROXY"]
    config = BucketConfig.from_env("GRADLAB_CONTROL_R2", environment=session)
    assert config.transport == "agent-vault"
    assert config.access_key_id == vault.MARKERS["GRADLAB_CONTROL_R2_ACCESS_KEY_ID"]
    assert config.secret_access_key == vault.MARKERS["GRADLAB_CONTROL_R2_SECRET_ACCESS_KEY"]


@pytest.mark.parametrize("change", [
    {vault.MODE: "direct"}, {"GRADLAB_CONTROL_R2_SECRET_ACCESS_KEY": "synthetic-raw-key"},
    {"DSTACK_TOKEN": "synthetic-raw-token"}, {"AWS_SECRET_ACCESS_KEY": "synthetic-raw-key"},
    {"INFISICAL_TOKEN": "synthetic-admin-token"}, {"WANDB_API_KEY": "synthetic-source-key"},
    {"HF_TOKEN": "synthetic-source-key"}, {"NO_PROXY": "*"}, {"PYTHONHTTPSVERIFY": "0"},
    {"HTTPS_PROXY": "http://x-agent-vault:agv_syntheticCapability@8.8.8.8:17323"},
    {"http_proxy": "http://different.invalid"}, {vault.CA_PIN: "changed"},
    {"GRADLAB_CONTROL_R2_ENDPOINT_URL": "https://unreviewed.invalid"},
])
def test_invalid_training_session_never_selects_direct_credentials(session, change):
    with pytest.raises(vault.TrainingVaultError):
        BucketConfig.from_env("GRADLAB_CONTROL_R2", environment={**session, **change})


def test_selected_vault_never_reads_unused_source_keys_or_keychain(session, tmp_path):
    path = tmp_path / "operator.toml"
    path.write_text('''schema_version = 3
[infisical.GRADLAB_CONTROL_R2_SECRET_ACCESS_KEY]
instance = "https://app.infisical.com"
project_id = "00000000-0000-4000-8000-000000000001"
secret_name = "GRADLAB_CONTROL_R2_SECRET_ACCESS_KEY"
environment = "prod"
path = "/"
[infisical.MODAL_TOKEN_SECRET]
instance = "https://app.infisical.com"
project_id = "00000000-0000-4000-8000-000000000002"
secret_name = "MODAL_TOKEN_SECRET"
environment = "prod"
path = "/modal"
''')
    with patch("gradlab.operator_credentials.read_secret", side_effect=AssertionError("source retrieval")), \
         patch("gradlab.operator_credentials._keychain_lookup", side_effect=AssertionError("keychain fallback")):
        report = load_operator_environment(environment=session, config_path=path)
    assert report.loaded_sources["GRADLAB_CONTROL_R2_SECRET_ACCESS_KEY"] == "agent-vault"
    assert "MODAL_TOKEN_SECRET" not in session


def test_worker_context_requires_finite_unexpired_session(session):
    import json

    context = {"session_id": "00000000-0000-4000-8000-000000000001",
               "expires_at": (datetime.now(UTC) + timedelta(hours=7)).isoformat(),
               "proxy_address": "172.17.0.1:17323", "secret_ref": "GRADLAB_VAULT_SESSION_00000000000040008000000000000001"}
    session[vault.WORKER_CONTEXT] = json.dumps(context)
    session[vault.WORKER_CAPABILITY] = "http://x-agent-vault:agv_workerCapability@172.17.0.1:17323"
    assert vault.worker_context(session) == context
    for expiry in ("never", (datetime.now(UTC) - timedelta(seconds=1)).isoformat(),
                   (datetime.now(UTC) + timedelta(days=3)).isoformat()):
        session[vault.WORKER_CONTEXT] = json.dumps({**context, "expires_at": expiry})
        with pytest.raises(vault.TrainingVaultError):
            vault.worker_context(session)


def test_offline_learner_cannot_inherit_supervisor_vault_capabilities(session):
    session[vault.WORKER_CAPABILITY] = "synthetic-private-worker-capability"
    session["INFISICAL_AGENT_VAULT_SESSION_TOKEN"] = "agv_nativeSession"
    session["INFISICAL_AGENT_VAULT_URL"] = "http://x-agent-vault:agv_nativeSession@127.0.0.1:17323"
    result = vault.offline_learner_environment(session)
    assert "agv_" not in str(result)
    assert "synthetic-private-worker-capability" not in str(result)
    assert vault.MODE not in result
    assert result["HTTP_PROXY"] == "http://127.0.0.1:9"
    assert result["NO_PROXY"] == ""


@pytest.mark.parametrize("failure", [False, True])
def test_supervisor_revokes_its_session_and_removes_public_ca_on_exit(tmp_path, monkeypatch, failure):
    from gradlab import run_supervisor

    directory = tmp_path / "vault-ca"
    directory.mkdir()
    ca = directory / "public-ca.pem"
    ca.write_text("synthetic public CA")
    mlflow = tmp_path / "combined-public-ca.pem"
    mlflow.write_text("synthetic combined public CA")
    monkeypatch.setenv(vault.MODE, "agent-vault")
    with patch.object(vault, "activate_public_ca", return_value=ca), \
         patch.object(vault, "validate_session"), \
         patch.object(vault, "revoke_own_session") as revoke, \
         patch("gradlab.mlflow_tls.configure_private_mlflow_ca", return_value=mlflow), \
         patch.object(run_supervisor, "RunSupervisor") as supervisor:
        supervisor.return_value.run.side_effect = RuntimeError("synthetic supervisor fault") if failure else None
        supervisor.return_value.run.return_value = 0
        if failure:
            with pytest.raises(RuntimeError, match="synthetic supervisor fault"):
                run_supervisor.main(["--manifest-uri", "s3://fixture/manifest.json"])
        else:
            assert run_supervisor.main(["--manifest-uri", "s3://fixture/manifest.json"]) == 0
        revoke.assert_called_once()
    assert not directory.exists()
    assert not mlflow.exists()


def test_provider_errors_preserve_only_status_and_never_proxy_capability():
    class Client:
        def put_object(self, **kwargs):
            error = RuntimeError("http://x-agent-vault:agv_doNotExpose@127.0.0.1:17323")
            error.response = {"ResponseMetadata": {"HTTPStatusCode": 412},
                              "Error": {"Message": "synthetic-source-key"}}
            raise error

    with pytest.raises(vault.SafeS3Error) as error:
        vault.SafeS3Client(Client()).put_object(Bucket="test", Key="test")
    assert error.value.response["ResponseMetadata"]["HTTPStatusCode"] == 412
    assert "agv_doNotExpose" not in str(error.value)
    assert "synthetic-source-key" not in str(error.value)
    assert "Error" not in error.value.response


def test_proxy_header_canonicalization_preserves_s3_metadata_keys():
    class Client:
        def head_object(self, **kwargs):
            return {"Metadata": {"Sha256": "synthetic-digest", "Content-Role": "checkpoint"}}

    assert vault.SafeS3Client(Client()).head_object()["Metadata"] == {
        "sha256": "synthetic-digest", "content-role": "checkpoint",
    }
