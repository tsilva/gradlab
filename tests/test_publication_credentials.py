from __future__ import annotations

import os
import stat
import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from gradlab.publication_credentials import (
    CredentialSecurityError,
    resolve_huggingface_credential,
    youtube_credential_paths,
    load_private_json,
    save_private_json,
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


def test_youtube_vault_reads_and_refresh_writes_with_local_files_absent(tmp_path, monkeypatch):
    from gradlab import infisical_credentials as storage
    from gradlab.player_publication import _youtube_access
    from gradlab.youtube_publication import YOUTUBE_SCOPES, YouTubePublicationError

    root = tmp_path / ".config/gradlab"
    root.mkdir(parents=True)
    (root / "youtube_infisical.json").write_text(
        json.dumps(
            {
                "project_id": "11111111-1111-1111-1111-111111111111",
                "environment": "prod",
                "path": "/youtube",
                "instance": "https://vault.example.test",
                "secrets": {"client": "CLIENT_JSON", "token": "TOKEN_JSON"},
            }
        )
    )
    values = {
        "CLIENT_JSON": {"installed": {"client_id": "client", "client_secret": "secret"}},
        "TOKEN_JSON": {
            "access_token": "expired",
            "refresh_token": "old",
            "scopes": list(YOUTUBE_SCOPES),
        },
    }

    def run(args, **kwargs):
        if args[2] == "set":
            name = args[3].split("=", 1)[0]
            values[name] = json.loads(kwargs["input"])
            return SimpleNamespace(returncode=0, stdout="")
        return SimpleNamespace(returncode=0, stdout=json.dumps(values[args[3]]))

    class Client:
        def __init__(self, token):
            self.token = token

        def channel_identity(self):
            if self.token == "expired":
                raise YouTubePublicationError("expired", status=401)
            return {"channel_id": "expected"}

    monkeypatch.setattr(storage.subprocess, "run", run)
    paths = youtube_credential_paths(environment={"HOME": str(tmp_path)})
    monkeypatch.setattr("gradlab.player_publication.youtube_credential_paths", lambda: paths)
    monkeypatch.setattr(
        "gradlab.youtube_publication.post_form",
        lambda *args: {"access_token": "fresh", "refresh_token": "rotated"},
    )
    client, principal = _youtube_access(client_factory=Client)
    assert client.token == "fresh"
    assert principal["channel_id"] == "expected"
    assert values["TOKEN_JSON"]["refresh_token"] == "rotated"
    assert not paths.client.exists() and not paths.token.exists()
    assert load_private_json(paths.token, root=paths.root)["access_token"] == "fresh"
    save_private_json(paths.token, values["TOKEN_JSON"], root=paths.root)


def test_youtube_vault_failure_never_reads_plaintext(tmp_path, monkeypatch):
    from gradlab import infisical_credentials as storage

    root = tmp_path
    path = root / "youtube_token.json"
    path.write_text('{"refresh_token":"private-local"}')
    (root / "youtube_infisical.json").write_text(
        json.dumps(
            {
                "project_id": "11111111-1111-1111-1111-111111111111",
                "environment": "prod",
                "path": "/youtube",
                "instance": "https://vault.example.test",
                "secrets": {"token": "TOKEN_JSON"},
            }
        )
    )
    monkeypatch.setattr(
        storage.subprocess,
        "run",
        lambda *args, **kw: SimpleNamespace(
            returncode=1, stdout="private-error", stderr="private-error"
        ),
    )
    with pytest.raises(CredentialSecurityError) as error:
        load_private_json(path, root=root)
    assert "private-" not in str(error.value)


@pytest.mark.parametrize("missing", ["instance", "project_id", "environment", "path"])
def test_incomplete_vault_scope_does_not_read_or_overwrite_local_tokens(tmp_path, monkeypatch, missing):
    from gradlab import infisical_credentials as storage

    token = tmp_path / "youtube_token.json"
    original = '{"refresh_token":"private-local"}'
    token.write_text(original)
    config = {
        "instance": "https://vault.example.test",
        "project_id": "11111111-1111-1111-1111-111111111111",
        "environment": "prod", "path": "/youtube",
        "secrets": {"token": "TOKEN_JSON"},
    }
    del config[missing]
    (tmp_path / "youtube_infisical.json").write_text(json.dumps(config))
    monkeypatch.setattr(storage.subprocess, "run", lambda *a, **kw: pytest.fail("provider called"))
    with pytest.raises(CredentialSecurityError, match="configuration"):
        load_private_json(token, root=tmp_path)
    with pytest.raises(CredentialSecurityError, match="configuration"):
        save_private_json(token, {"refresh_token": "replacement"}, root=tmp_path)
    assert token.read_text() == original
