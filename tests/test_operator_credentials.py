from __future__ import annotations

from pathlib import Path

import pytest

from rlab.operator_credentials import (
    KeychainReference,
    OperatorConfigurationError,
    load_operator_environment,
    reject_protected_dotenv,
)


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
schema_version = 1

[environment]
DSTACK_SERVER_URL = "http://127.0.0.1:3000"
WANDB_ENTITY = "example"

[keychain.DSTACK_TOKEN]
service = "rlab-dstack-admin"
account = "operator"

[keychain.WANDB_API_KEY]
service = "rlab-wandb"
account = "api-key"

[modal]
path = "{modal_path}"
""".strip()
        + "\n",
    )
    stored = {
        KeychainReference("rlab-dstack-admin", "operator"): "dstack-token",
        KeychainReference("rlab-wandb", "api-key"): "wandb-key",
    }
    environment: dict[str, str] = {}

    report = load_operator_environment(
        environment=environment,
        config_path=config_path,
        keychain_lookup=stored.get,
    )

    assert environment == {
        "DSTACK_SERVER_URL": "http://127.0.0.1:3000",
        "WANDB_ENTITY": "example",
        "DSTACK_TOKEN": "dstack-token",
        "WANDB_API_KEY": "wandb-key",
        "MODAL_TOKEN_ID": "modal-id",
        "MODAL_TOKEN_SECRET": "modal-secret",
    }
    assert report.loaded_sources["DSTACK_SERVER_URL"] == "operator-config"
    assert report.loaded_sources["DSTACK_TOKEN"] == "macos-keychain"
    assert report.loaded_sources["MODAL_TOKEN_SECRET"] == "modal-profile"
    assert not report.unavailable_sources


def test_process_environment_wins_without_reading_keychain(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "operator.toml",
        """
schema_version = 1

[keychain.WANDB_API_KEY]
service = "rlab-wandb"
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
schema_version = 1

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
schema_version = 1

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


def test_missing_keychain_item_is_reported_without_a_value(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "operator.toml",
        """
schema_version = 1

[keychain.WANDB_API_KEY]
service = "rlab-wandb"
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
