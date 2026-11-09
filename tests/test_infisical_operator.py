import subprocess
from types import SimpleNamespace

import pytest

from gradlab import operator_credentials as operator
from gradlab import secret_reference as transport


PROJECT = "11111111-1111-4111-8111-111111111111"
INSTANCE = "https://vault.example.test"


def config(tmp_path):
    path = tmp_path / "operator.toml"
    path.write_text(
        f'schema_version = 3\n[infisical.WANDB_API_KEY]\nproject_id = "{PROJECT}"\n'
        'secret_name = "WANDB_API_KEY"\nenvironment = "prod"\npath = "/wandb"\n'
        f'instance = "{INSTANCE}"\n'
    )
    path.chmod(0o600)
    return path


def test_infisical_binding_loads_without_touching_legacy_provider(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(operator, "read_secret", lambda ref: calls.append(ref) or "test-value")
    env = {}
    report = operator.load_operator_environment(
        environment=env, config_path=config(tmp_path),
        keychain_lookup=lambda _: pytest.fail("legacy provider used"),
    )
    assert env == {"WANDB_API_KEY": "test-value"}
    assert calls == [transport.InfisicalReference(PROJECT, "WANDB_API_KEY", "prod", "/wandb", INSTANCE)]
    assert report.loaded_sources == {"WANDB_API_KEY": "infisical"}


def test_existing_environment_and_unrelated_scope_skip_fetch(tmp_path, monkeypatch):
    monkeypatch.setattr(operator, "read_secret", lambda _: pytest.fail("unexpected fetch"))
    operator.load_operator_environment(environment={"WANDB_API_KEY": "injected"}, config_path=config(tmp_path))
    operator.load_operator_environment(environment={}, config_path=config(tmp_path), requested_names={"MODAL_TOKEN_ID"})


def test_failure_never_falls_back(tmp_path, monkeypatch):
    def fail(_):
        raise ValueError("Infisical access failed")
    monkeypatch.setattr(operator, "read_secret", fail)
    env = {}
    with pytest.raises(operator.OperatorConfigurationError, match="Infisical access failed"):
        operator.load_operator_environment(environment=env, config_path=config(tmp_path))
    assert env == {}


def test_transport_scopes_cli_and_suppresses_failure_output(monkeypatch):
    calls = []
    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=1, stdout="private-output", stderr="private-output")
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(ValueError) as exc:
        transport.read_secret(transport.InfisicalReference(PROJECT, "WANDB_API_KEY", "prod", "/wandb", INSTANCE))
    assert "private-output" not in str(exc.value)
    args, kwargs = calls[0]
    assert args[args.index("--projectId") + 1] == PROJECT
    assert args[args.index("--path") + 1] == "/wandb"
    assert args[args.index("--domain") + 1] == INSTANCE
    assert "--include-imports=false" in args
    assert kwargs["stdin"] == subprocess.DEVNULL


def test_invalid_reference_rejects_extra_keys_and_wrong_scope():
    for ref in [
        {"project_id": PROJECT, "secret_name": "WANDB_API_KEY", "domain": "http://bad"},
        {"project_id": "other", "secret_name": "WANDB_API_KEY"},
        {"project_id": PROJECT, "secret_name": "WANDB_API_KEY", "path": "/../../other"},
    ]:
        with pytest.raises(ValueError, match="Invalid Infisical reference"):
            transport.InfisicalReference.parse(ref)


def reference_values():
    return dict(project_id=PROJECT, secret_name="WANDB_API_KEY", environment="prod",
                path="/wandb", instance=INSTANCE)


@pytest.mark.parametrize("field", ["project_id", "secret_name", "environment", "path", "instance"])
def test_reference_requires_explicit_fields(field):
    values = reference_values()
    del values[field]
    with pytest.raises(ValueError, match="Invalid Infisical reference"):
        transport.InfisicalReference.parse(values)


@pytest.mark.parametrize("field,value", [
    ("instance", "https://user:private@vault.example.test"),
    ("instance", "https://vault.example.test?token=private"),
    ("instance", "https://vault.example.test#fragment"),
    ("instance", "http://vault.example.test"),
    ("instance", "https://"),
    ("instance", "https://vault.example.test:invalid"),
    ("instance", "https://vault.example.test\n"),
    ("path", "/folder//other"),
    ("path", "/../other"),
    ("environment", ""),
    ("secret_name", "--token"),
])
def test_direct_constructor_and_parser_share_validation(field, value):
    values = reference_values()
    values[field] = value
    for build in (lambda: transport.InfisicalReference(**values),
                  lambda: transport.InfisicalReference.parse(values)):
        with pytest.raises(ValueError, match="Invalid Infisical reference") as error:
            build()
        assert "private" not in str(error.value)


def test_explicit_root_is_supported_without_becoming_a_default():
    values = reference_values()
    values["path"] = "/"
    assert transport.InfisicalReference.parse(values).path == "/"
