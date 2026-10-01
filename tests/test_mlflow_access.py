from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from gradlab.experiment_cli import (
    _required_operator_environment,
    build_parser,
    cmd_operator_preflight,
)
from gradlab.mlflow_access import mlflow_auth_mode, validate_mlflow_access
from gradlab.operator_credentials import OperatorConfigurationError


@pytest.fixture(autouse=True)
def clean_auth_environment(monkeypatch):
    for name in ("MLFLOW_AUTH_MODE", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"):
        monkeypatch.delenv(name, raising=False)


def test_default_online_preflight_requires_mlflow_not_wandb():
    required = _required_operator_environment("none")
    assert "MLFLOW_TRACKING_URI" in required
    assert "MLFLOW_TRACKING_PASSWORD" in required
    assert "WANDB_API_KEY" not in required


def test_basic_auth_remains_required_and_network_mode_is_explicit(monkeypatch):
    with pytest.raises(OperatorConfigurationError, match="username and password"):
        validate_mlflow_access("https://192.168.1.2")
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "writer")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "test-password")
    validate_mlflow_access("https://192.168.1.2")
    monkeypatch.setenv("MLFLOW_AUTH_MODE", "network")
    with pytest.raises(OperatorConfigurationError, match="must not supply"):
        validate_mlflow_access("https://192.168.1.2")


@pytest.mark.parametrize("uri", ["https://192.168.1.2:5443", "https://100.64.58.181:5443"])
def test_explicit_network_mode_uses_private_https_without_fake_credentials(monkeypatch, uri):
    monkeypatch.setenv("MLFLOW_AUTH_MODE", "network")
    validate_mlflow_access(uri)
    required = _required_operator_environment(
        "none", {"backend": "mlflow", "delivery": "online", "auth_mode": "network"}
    )
    assert "MLFLOW_AUTH_MODE" in required
    assert "MLFLOW_TRACKING_USERNAME" not in required
    assert "MLFLOW_TRACKING_PASSWORD" not in required


@pytest.mark.parametrize(
    "uri",
    [
        "http://192.168.1.2",
        "https://8.8.8.8",
        "https://127.0.0.1",
        "https://0.0.0.0",
        "https://224.0.0.1",
        "https://user:password@192.168.1.2",
    ],
)
def test_network_mode_rejects_unprotected_and_container_loopback_routes(monkeypatch, uri):
    monkeypatch.setenv("MLFLOW_AUTH_MODE", "network")
    with pytest.raises(OperatorConfigurationError):
        validate_mlflow_access(uri)


def test_network_hostname_rejects_any_public_resolution(monkeypatch):
    monkeypatch.setenv("MLFLOW_AUTH_MODE", "network")
    with mock.patch(
        "gradlab.mlflow_access.socket.getaddrinfo",
        return_value=[(2, 1, 6, "", ("192.168.1.2", 443)), (2, 1, 6, "", ("8.8.8.8", 443))],
    ):
        with pytest.raises(OperatorConfigurationError, match="private address"):
            validate_mlflow_access("https://private.example.test")


def test_frozen_auth_mode_cannot_change_on_retry(monkeypatch):
    monkeypatch.setenv("MLFLOW_AUTH_MODE", "network")
    assert mlflow_auth_mode({"auth_mode": "basic"}) == "basic"
    with pytest.raises(OperatorConfigurationError, match="frozen Run"):
        validate_mlflow_access("https://192.168.1.2", {"auth_mode": "basic"})


def test_preflight_command_resolves_project_tracking_default(tmp_path, capsys):
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments/tracking.yaml").write_text("backend: mlflow\ndelivery: online\n")
    args = build_parser().parse_args(["operator-preflight", "--json"])
    with (
        mock.patch("gradlab.experiment_cli.repository_root", return_value=tmp_path),
        mock.patch(
            "gradlab.experiment_cli._operator_preflight",
            return_value=(None, None, None, {"status": "ready"}),
        ) as preflight,
    ):
        assert cmd_operator_preflight(args) == 0
    assert preflight.call_args.kwargs["tracking"]["backend"] == "mlflow"
    assert preflight.call_args.kwargs["checkpoint_eval_backend"] == "modal"


def test_preflight_uses_recipe_tracking_and_eval_mode_with_overrides(tmp_path):
    recipe = Path("experiments/goals/game/goal/recipes/ppo.yaml")
    args = build_parser().parse_args(
        [
            "operator-preflight",
            "--recipe-file",
            str(recipe),
            "--set",
            "tracking.backend=mlflow",
            "--json",
        ]
    )
    config = {
        "tracking": {"backend": "mlflow", "delivery": "online"},
        "checkpoint_eval_backend": "none",
    }
    with (
        mock.patch("gradlab.experiment_cli.repository_root", return_value=tmp_path),
        mock.patch("gradlab.experiment_cli._git", return_value="a" * 40),
        mock.patch(
            "gradlab.experiment_cli._goal_path_for_recipe", return_value=tmp_path / "goal.yaml"
        ),
        mock.patch(
            "gradlab.experiment_cli.compose_resolved_train_documents",
            return_value=SimpleNamespace(effective={"train_config": config}),
        ) as compose,
        mock.patch(
            "gradlab.experiment_cli._operator_preflight",
            return_value=(None, None, None, {"status": "ready"}),
        ) as preflight,
    ):
        cmd_operator_preflight(args)
    assert compose.call_args.kwargs["recipe_overrides"] == ("tracking.backend=mlflow",)
    assert preflight.call_args.kwargs["tracking"] == config["tracking"]
    assert preflight.call_args.kwargs["checkpoint_eval_backend"] == "none"
