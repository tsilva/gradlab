from __future__ import annotations

import os
from pathlib import Path

import pytest

from gradlab.operator_credentials import (
    DstackCoordinatorProfile,
    KeychainReference,
    OperatorConfigurationError,
    SshTunnelProfile,
    load_operator_environment,
    reject_protected_dotenv,
    resolve_dstack_token,
)
from gradlab.operator_environment import load_repository_operator_environment


def _write(path: Path, text: str, *, mode: int = 0o600) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)
    return path


def test_operator_config_loads_metadata_keychain_and_active_modal_profile(
    tmp_path: Path,
) -> None:
    modal_path = _write(
        tmp_path / "modal.toml",
        """
[workspace]
token_id = "modal-id"
token_secret = "modal-secret"
active = true
""".strip()
        + "\n",
    )
    config_path = _write(
        tmp_path / "operator.toml",
        f"""
schema_version = 3

[environment]
WANDB_ENTITY = "example"

[dstack]
default_coordinator = "primary"
default_fleet = "local-gpu"

[dstack.coordinators.primary]
project = "main"
server_url = "http://127.0.0.1:3000"
token = {{ service = "gradlab-dstack-admin", account = "operator" }}
ssh_tunnel = {{ destinations = ["operator@gpu.local", "operator@gpu.fallback"], remote_host = "127.0.0.1", remote_port = 3000 }}

[dstack.fleets.local-gpu]
coordinator = "primary"
cpu = 12
memory = "40GB"
gpu = "1"
disk = "50GB"

[keychain.WANDB_API_KEY]
service = "gradlab-wandb"
account = "api-key"

[modal]
path = "{modal_path}"
""".strip()
        + "\n",
    )
    stored = {
        KeychainReference("gradlab-dstack-admin", "operator"): "dstack-token",
        KeychainReference("gradlab-wandb", "api-key"): "wandb-key",
    }
    environment: dict[str, str] = {}

    report = load_operator_environment(
        environment=environment,
        config_path=config_path,
        keychain_lookup=stored.get,
    )

    assert environment == {
        "WANDB_ENTITY": "example",
        "WANDB_API_KEY": "wandb-key",
        "MODAL_TOKEN_ID": "modal-id",
        "MODAL_TOKEN_SECRET": "modal-secret",
    }
    assert report.dstack is not None
    assert report.dstack.fleet().coordinator_id == "primary"
    assert report.dstack.coordinator().token == KeychainReference(
        "gradlab-dstack-admin", "operator"
    )
    assert report.dstack.coordinator().ssh_tunnel == SshTunnelProfile(
        destinations=("operator@gpu.local", "operator@gpu.fallback"),
        remote_host="127.0.0.1",
        remote_port=3000,
    )
    assert report.loaded_sources["MODAL_TOKEN_SECRET"] == "modal-profile"
    assert not report.unavailable_sources


def test_operator_config_loads_private_per_fleet_dstack_resources(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "operator.toml",
        """
schema_version = 3

[dstack]
default_coordinator = "primary"
default_fleet = "small-gpu"

[dstack.coordinators.primary]
project = "main"
server_url = "http://127.0.0.1:3000"
token = { service = "gradlab-dstack-admin", account = "operator" }

[dstack.fleets.small-gpu]
coordinator = "primary"
cpu = 12
memory = "28GB"
gpu = "1"
disk = "50GB"
""".strip()
        + "\n",
    )

    report = load_operator_environment(
        environment={},
        config_path=config_path,
        keychain_lookup=lambda _reference: None,
    )

    assert report.dstack is not None
    assert report.dstack.fleet("small-gpu").resources.as_manifest() == {
        "cpu": 12,
        "memory": "28GB",
        "gpu": "1",
        "disk": "50GB",
    }


def test_operator_config_rejects_invalid_dstack_resource_profile(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "operator.toml",
        """
schema_version = 3

[dstack]
default_coordinator = "primary"
default_fleet = "small-gpu"

[dstack.coordinators.primary]
project = "main"
server_url = "http://127.0.0.1:3000"
token = { service = "gradlab-dstack-admin", account = "operator" }

[dstack.fleets.small-gpu]
coordinator = "primary"
cpu = 12
memory = "28 gigabytes"
gpu = "1"
disk = "50GB"
""".strip()
        + "\n",
    )

    with pytest.raises(OperatorConfigurationError, match="memory must be"):
        load_operator_environment(
            environment={},
            config_path=config_path,
            keychain_lookup=lambda _reference: None,
        )


def test_dual_coordinator_tokens_are_resolved_only_after_fleet_selection(
    tmp_path: Path,
) -> None:
    config_path = _write(
        tmp_path / "operator.toml",
        """
schema_version = 3

[dstack]
default_coordinator = "b3"
default_fleet = "b3"

[dstack.coordinators.b3]
project = "main"
server_url = "http://127.0.0.1:3000"
token = { service = "dstack-b3", account = "operator" }

[dstack.coordinators.b2]
project = "main"
server_url = "http://127.0.0.1:3002"
token = { service = "dstack-b2", account = "operator" }

[dstack.fleets.b3]
coordinator = "b3"
cpu = 12
memory = "40GB"
gpu = "1"
disk = "50GB"

[dstack.fleets.b2]
coordinator = "b2"
cpu = 12
memory = "28GB"
gpu = "1"
disk = "50GB"
""".strip()
        + "\n",
    )
    calls: list[KeychainReference] = []
    report = load_operator_environment(
        environment={},
        config_path=config_path,
        keychain_lookup=lambda reference: calls.append(reference) or "unexpected",
    )

    assert calls == []
    assert report.dstack is not None
    fleet = report.dstack.fleet("b2")
    assert fleet.coordinator_id == "b2"
    token, source = resolve_dstack_token(
        report.dstack.coordinator(fleet.coordinator_id),
        environment={},
        keychain_lookup=lambda reference: calls.append(reference) or "b2-token",
    )
    assert token == "b2-token"
    assert source == "macos-keychain"
    assert calls == [KeychainReference("dstack-b2", "operator")]


def test_process_dstack_token_overrides_only_the_selected_profile() -> None:
    profile = DstackCoordinatorProfile(
        coordinator_id="b2",
        project="main",
        server_url="http://127.0.0.1:3002",
        token=KeychainReference("dstack-b2", "operator"),
    )
    calls: list[KeychainReference] = []

    token, source = resolve_dstack_token(
        profile,
        environment={"DSTACK_TOKEN": "process-token"},
        keychain_lookup=lambda reference: calls.append(reference) or "stored-token",
    )

    assert token == "process-token"
    assert source == "process-environment"
    assert calls == []


@pytest.mark.parametrize(
    ("ssh_tunnel", "message"),
    [
        ('{ destinations = [], remote_host = "127.0.0.1", remote_port = 3000 }', "non-empty"),
        (
            '{ destinations = ["-oBad=yes"], remote_host = "127.0.0.1", remote_port = 3000 }',
            "without options",
        ),
        (
            '{ destinations = ["operator@gpu"], remote_host = "127.0.0.1", remote_port = 70000 }',
            "between 1 and 65535",
        ),
    ],
)
def test_operator_config_rejects_invalid_ssh_tunnel_metadata(
    tmp_path: Path,
    ssh_tunnel: str,
    message: str,
) -> None:
    config_path = _write(
        tmp_path / "operator.toml",
        f"""
schema_version = 3

[dstack]
default_coordinator = "primary"
default_fleet = "local-gpu"

[dstack.coordinators.primary]
project = "main"
server_url = "http://127.0.0.1:3000"
token = {{ service = "gradlab-dstack-admin", account = "operator" }}
ssh_tunnel = {ssh_tunnel}

[dstack.fleets.local-gpu]
coordinator = "primary"
cpu = 12
memory = "40GB"
gpu = "1"
disk = "50GB"
""".strip()
        + "\n",
    )

    with pytest.raises(OperatorConfigurationError, match=message):
        load_operator_environment(
            environment={},
            config_path=config_path,
            keychain_lookup=lambda _reference: None,
        )


def test_schema_v3_rejects_ambient_dstack_routing_metadata(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "operator.toml",
        """
schema_version = 3

[environment]
DSTACK_SERVER_URL = "http://127.0.0.1:3000"
""".strip()
        + "\n",
    )

    with pytest.raises(OperatorConfigurationError, match="routing belongs under"):
        load_operator_environment(
            environment={},
            config_path=config_path,
            keychain_lookup=lambda _reference: None,
        )


def test_process_environment_wins_without_reading_keychain(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "operator.toml",
        """
schema_version = 3

[keychain.WANDB_API_KEY]
service = "gradlab-wandb"
account = "api-key"
""".strip()
        + "\n",
    )
    environment = {"WANDB_API_KEY": "process-value"}
    calls: list[KeychainReference] = []

    load_operator_environment(
        environment=environment,
        config_path=config_path,
        keychain_lookup=lambda reference: calls.append(reference) or "stored-value",
    )

    assert environment["WANDB_API_KEY"] == "process-value"
    assert calls == []


def test_plaintext_protected_operator_value_is_rejected(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "operator.toml",
        """
schema_version = 3

[environment]
WANDB_API_KEY = "plaintext-is-not-allowed"
""".strip()
        + "\n",
    )

    with pytest.raises(
        OperatorConfigurationError,
        match=r"WANDB_API_KEY must use \[keychain\]",
    ):
        load_operator_environment(
            environment={},
            config_path=config_path,
            keychain_lookup=lambda _reference: None,
        )


def test_modal_credential_file_must_be_private(tmp_path: Path) -> None:
    modal_path = _write(
        tmp_path / "modal.toml",
        """
[workspace]
token_id = "modal-id"
token_secret = "modal-secret"
active = true
""".strip()
        + "\n",
        mode=0o644,
    )
    config_path = _write(
        tmp_path / "operator.toml",
        f"""
schema_version = 3

[modal]
path = "{modal_path}"
""".strip()
        + "\n",
    )

    with pytest.raises(OperatorConfigurationError, match="must use mode 0600"):
        load_operator_environment(
            environment={},
            config_path=config_path,
            keychain_lookup=lambda _reference: None,
        )


def test_protected_dotenv_values_are_rejected(tmp_path: Path) -> None:
    dotenv_path = _write(
        tmp_path / ".env",
        "WANDB_ENTITY=example\nWANDB_API_KEY=must-not-live-here\n",
    )

    with pytest.raises(OperatorConfigurationError, match="WANDB_API_KEY"):
        reject_protected_dotenv(dotenv_path)


def test_repository_operator_environment_loads_safe_dotenv_before_operator_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write(tmp_path / ".env", "WANDB_ENTITY=repository-entity\n")
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.setenv(
        "GRADLAB_OPERATOR_CONFIG",
        str(tmp_path / "missing-operator.toml"),
    )

    report = load_repository_operator_environment(tmp_path)

    assert report.config_present is False
    assert report.config_path == (tmp_path / "missing-operator.toml").resolve()
    assert os.environ["WANDB_ENTITY"] == "repository-entity"


def test_missing_keychain_item_is_reported_without_a_value(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "operator.toml",
        """
schema_version = 3

[keychain.WANDB_API_KEY]
service = "gradlab-wandb"
account = "api-key"
""".strip()
        + "\n",
    )
    environment: dict[str, str] = {}

    report = load_operator_environment(
        environment=environment,
        config_path=config_path,
        keychain_lookup=lambda _reference: None,
    )

    assert "WANDB_API_KEY" not in environment
    assert report.unavailable_sources == {"WANDB_API_KEY": "macos-keychain"}


def test_scoped_load_does_not_resolve_unrelated_keychain_or_modal(
    tmp_path: Path,
) -> None:
    config_path = _write(
        tmp_path / "operator.toml",
        """
schema_version = 3

[environment]
WANDB_ENTITY = "research"
GRADLAB_CONTROL_R2_URI = "s3://control"

[keychain.WANDB_API_KEY]
service = "gradlab-wandb"
account = "api-key"

[keychain.GRADLAB_CONTROL_R2_ACCESS_KEY_ID]
service = "gradlab-r2-control"
account = "access-key-id"

[modal]
path = "/does/not/exist"
""".strip()
        + "\n",
    )
    calls: list[KeychainReference] = []
    environment: dict[str, str] = {}

    report = load_operator_environment(
        environment=environment,
        config_path=config_path,
        keychain_lookup=lambda reference: calls.append(reference) or "wandb-key",
        requested_names={"WANDB_API_KEY", "WANDB_ENTITY"},
    )

    assert environment == {
        "WANDB_API_KEY": "wandb-key",
        "WANDB_ENTITY": "research",
    }
    assert calls == [KeychainReference("gradlab-wandb", "api-key")]
    assert report.loaded_sources == {
        "WANDB_API_KEY": "macos-keychain",
        "WANDB_ENTITY": "operator-config",
    }


def test_scoped_load_does_not_validate_unrelated_entries(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "operator.toml",
        """
schema_version = 3

[environment]
WANDB_ENTITY = "research"
WANDB_API_KEY = "unrelated protected plaintext"

[keychain.GRADLAB_CONTROL_R2_ACCESS_KEY_ID]
service = "gradlab-control"
account = "access-key"

[keychain.WANDB_API_KEY]
unknown = "invalid but unrelated"

[modal]
path = "/missing/unrelated-modal.toml"
""".strip()
        + "\n",
    )
    environment: dict[str, str] = {}

    report = load_operator_environment(
        environment=environment,
        config_path=config_path,
        keychain_lookup=lambda reference: (
            "control-access"
            if reference == KeychainReference("gradlab-control", "access-key")
            else None
        ),
        requested_names={
            "GRADLAB_CONTROL_R2_ACCESS_KEY_ID",
        },
    )

    assert environment == {
        "GRADLAB_CONTROL_R2_ACCESS_KEY_ID": "control-access",
    }
    assert report.loaded_sources == {
        "GRADLAB_CONTROL_R2_ACCESS_KEY_ID": "macos-keychain",
    }
