from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

import gradlab.infisical_credentials as storage


def settings():
    return {
        "project_id": "11111111-1111-1111-1111-111111111111",
        "environment": "prod",
        "path": "/youtube",
        "instance": "https://vault.example.test",
        "secrets": {"token": "TEST_TOKEN_JSON"},
    }


def test_vault_write_uses_stdin_and_creates_no_plaintext(tmp_path, monkeypatch):
    payload = {"refresh_token": "private-refresh-material"}
    calls = []

    def run(args, **kwargs):
        assert payload["refresh_token"] not in repr(args)
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(storage.subprocess, "run", run)
    store = storage.InfisicalJsonStore(settings())
    store.write("token", payload)
    assert json.loads(calls[0][1]["input"]) == payload
    assert list(tmp_path.iterdir()) == []
    args = calls[0][0]
    assert args[args.index("--domain") + 1] == "https://vault.example.test"
    assert args[args.index("--path") + 1] == "/youtube"


@pytest.mark.parametrize("field", ["project_id", "environment", "path", "instance"])
def test_scope_fields_are_required_before_any_provider_access(field, monkeypatch):
    monkeypatch.setattr(storage.subprocess, "run", lambda *a, **kw: pytest.fail("provider called"))
    config = settings()
    del config[field]
    with pytest.raises(storage.SecretStoreError, match="configuration"):
        storage.InfisicalJsonStore(config)


@pytest.mark.parametrize("field,value", [
    ("instance", "http://vault.example.test"),
    ("instance", "https://user:private@vault.example.test"),
    ("path", "/youtube/../other"),
    ("environment", "--other-option"),
    ("domain", "https://unexpected.example.test"),
    ("secrets", {}),
    ("login_profile", ""),
    ("executable", ""),
])
def test_invalid_store_settings_rejected(field, value):
    config = settings()
    config[field] = value
    with pytest.raises(storage.SecretStoreError, match="configuration"):
        storage.InfisicalJsonStore(config)


def test_read_pins_scope_and_does_not_expand_or_import(monkeypatch):
    calls = []
    monkeypatch.setattr(storage.subprocess, "run", lambda args, **kw: (
        calls.append((args, kw)) or SimpleNamespace(returncode=0, stdout='{"ok":true}')
    ))
    store = storage.InfisicalJsonStore(settings())
    assert store.read("token") == {"ok": True}
    args, kwargs = calls[0]
    for flag, expected in [("--domain", "https://vault.example.test"),
                           ("--path", "/youtube"), ("--env", "prod"),
                           ("--projectId", settings()["project_id"])]:
        assert args[args.index(flag) + 1] == expected
    assert "--expand=false" in args and "--include-imports=false" in args
    assert "--silent" in args and "--telemetry=false" in args
    assert kwargs["stdin"] == subprocess.DEVNULL


@pytest.mark.parametrize("failure", ["cli", "json", "timeout"])
def test_failures_never_expose_provider_values(failure, monkeypatch):
    private = "private-provider-material"

    def run(*args, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args[0], 60, output=private)
        return SimpleNamespace(
            returncode=1 if failure == "cli" else 0, stdout=private, stderr=private
        )

    monkeypatch.setattr(storage.subprocess, "run", run)
    with pytest.raises(storage.SecretStoreError) as exc:
        storage.InfisicalJsonStore(settings()).read("token")
    assert private not in str(exc.value)


def test_present_invalid_config_does_not_select_file_backend(tmp_path):
    path = tmp_path / "infisical.json"
    assert storage.load_store(path) is None
    path.write_text("invalid-private-material")
    with pytest.raises(storage.SecretStoreError) as exc:
        storage.load_store(path)
    assert "invalid-private-material" not in str(exc.value)


def test_symlinked_configuration_rejected(tmp_path):
    path = tmp_path / "infisical.json"
    path.symlink_to(tmp_path / "missing")
    with pytest.raises(storage.SecretStoreError):
        storage.load_store(path)


def test_distinct_profile_bindings_and_unconfigured_role_fail_closed(monkeypatch):
    config = settings()
    config["secrets"] = {"alice": "ALICE_TOKEN_JSON", "bob": "BOB_TOKEN_JSON"}
    observed = []

    def run(args, **kwargs):
        observed.append(args[3])
        return SimpleNamespace(returncode=0, stdout='{"identity":"expected"}')

    monkeypatch.setattr(storage.subprocess, "run", run)
    store = storage.InfisicalJsonStore(config)
    store.read("alice")
    store.read("bob")
    assert observed == ["ALICE_TOKEN_JSON", "BOB_TOKEN_JSON"]
    with pytest.raises(storage.SecretStoreError):
        store.read("charlie")
