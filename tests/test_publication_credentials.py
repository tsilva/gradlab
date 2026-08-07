from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from gradlab.publication_credentials import (
    CredentialSecurityError,
    resolve_huggingface_credential,
    youtube_credential_paths,
)


def test_huggingface_environment_token_takes_precedence_without_files(tmp_path: Path) -> None:
    credential = resolve_huggingface_credential(
        {"HF_TOKEN": "secret-environment-token", "HF_HOME": str(tmp_path / "missing")}
    )

    assert credential.source == "environment"
    assert credential.token == "secret-environment-token"


def test_huggingface_file_store_is_secured_before_read(tmp_path: Path) -> None:
    home = tmp_path / "huggingface"
    home.mkdir(mode=0o755)
    token = home / "token"
    stored = home / "stored_tokens"
    token.write_text("hf_private_value\n", encoding="utf-8")
    stored.write_text("another private value\n", encoding="utf-8")
    os.chmod(token, 0o644)
    os.chmod(stored, 0o644)

    credential = resolve_huggingface_credential({"HF_HOME": str(home)})

    assert credential.token == "hf_private_value"
    assert stat.S_IMODE(home.stat().st_mode) == 0o700
    assert stat.S_IMODE(token.stat().st_mode) == 0o600
    assert stat.S_IMODE(stored.stat().st_mode) == 0o600


def test_huggingface_symlinked_token_fails_closed(tmp_path: Path) -> None:
    home = tmp_path / "huggingface"
    home.mkdir(mode=0o700)
    outside = tmp_path / "outside"
    outside.write_text("secret", encoding="utf-8")
    (home / "token").symlink_to(outside)

    with pytest.raises(CredentialSecurityError, match="regular non-symlink"):
        resolve_huggingface_credential({"HF_HOME": str(home)})


def test_youtube_secret_paths_normalize_owned_modes(tmp_path: Path) -> None:
    config = tmp_path / ".config" / "gradlab"
    config.mkdir(parents=True, mode=0o755)
    client = config / "youtube_client_secret.json"
    client.write_text('{"installed": {"client_id": "id", "client_secret": "secret"}}')
    os.chmod(client, 0o644)

    paths = youtube_credential_paths(environment={"HOME": str(tmp_path)})

    assert paths.root == config.resolve()
    assert stat.S_IMODE(paths.root.stat().st_mode) == 0o700
    assert stat.S_IMODE(paths.client.stat().st_mode) == 0o600
    assert stat.S_IMODE(paths.lock.stat().st_mode) == 0o600


def test_youtube_symlinked_client_secret_fails_closed(tmp_path: Path) -> None:
    config = tmp_path / ".config" / "gradlab"
    config.mkdir(parents=True, mode=0o700)
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    (config / "youtube_client_secret.json").symlink_to(outside)

    with pytest.raises(CredentialSecurityError, match="regular non-symlink"):
        youtube_credential_paths(environment={"HOME": str(tmp_path)})
