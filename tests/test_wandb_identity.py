from __future__ import annotations

import sys
import tempfile
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gradlab.env import EnvConfig
from gradlab.metric_names import METRICS_SCHEMA_VERSION
from gradlab.wandb_publisher import WandbProjector, _start_wandb
from gradlab.wandb_utils import (
    canonical_wandb_environment,
    game_family_for_environment,
    resolve_wandb_project,
)


@pytest.mark.parametrize(
    ("provider", "game", "project", "family"),
    [
        ("gradlab", "Bandit-v0", "Bandit-v0", "Bandit"),
        (
            "supermariobrosnes-turbo",
            "SuperMarioBros-Nes-v0",
            "SuperMarioBros-Nes-v0",
            "NES-SuperMarioBros",
        ),
        (
            "stable-retro-turbo",
            "SuperMarioBros-Nes-v0",
            "SuperMarioBros-Nes-v0",
            "NES-SuperMarioBros",
        ),
        ("ale-py", "breakout", "Breakout-Atari2600-v0", "Atari2600-Breakout"),
        (
            "breakout-turbo-env",
            "Breakout-Atari2600-v0",
            "Breakout-Atari2600-v0",
            "Atari2600-Breakout",
        ),
        (
            "stable-retro-turbo",
            "Breakout-Atari2600-v0",
            "Breakout-Atari2600-v0",
            "Atari2600-Breakout",
        ),
        ("ale-py", "ms_pacman", "MsPacman-Atari2600-v0", "Atari2600-MsPacman"),
        (
            "stable-retro-turbo",
            "MsPacman-Atari2600-v0",
            "MsPacman-Atari2600-v0",
            "Atari2600-MsPacman",
        ),
        (
            "stable-retro-turbo",
            "SuperMarioBros3-Nes-v0",
            "SuperMarioBros3-Nes-v0",
            "NES-SuperMarioBros3",
        ),
    ],
)
def test_canonical_wandb_environment_mapping(provider, game, project, family) -> None:
    assert canonical_wandb_environment(provider, game) == (project, family)


def test_explicit_project_wins_and_registered_providers_reject_unknown_environments() -> None:
    assert (
        resolve_wandb_project("custom-project", "breakout", env_provider="ale-py")
        == "custom-project"
    )
    with pytest.raises(ValueError, match="not registered"):
        resolve_wandb_project(None, "CustomNativeVector-v0", env_provider="gymnasium")
    with pytest.raises(ValueError, match="not registered"):
        game_family_for_environment("gymnasium", "CustomNativeVector-v0")


def test_environment_identity_requires_a_current_registered_provider() -> None:
    with pytest.raises(ValueError, match="provider identity is required"):
        canonical_wandb_environment(None, "SuperMarioBros-Nes-v0")
    with pytest.raises(ValueError, match="unknown environment provider"):
        canonical_wandb_environment("unknown-provider", "breakout")


def test_init_wandb_records_resolved_identity_and_submission_group() -> None:
    captured = {}

    class FakeRun:
        def define_metric(self, *_args, **_kwargs) -> None:
            return None

    def fake_init(**kwargs):
        captured.update(kwargs)
        return FakeRun()

    train_config = {
        "wandb_tags": "goal_id:alepy__breakout,recipe_id:base",
        "wandb_entity": "entity",
        "wandb_project": None,
        "wandb_display_name": "breakout__base__s123__01234567",
        "wandb_group": "bx0123456789abcdef",
        "run_name": "bx0123456789abcdef-base-s123-20260714T120000Z",
        "run_description": "offline identity canary",
        "wandb_mode": "offline",
        "wandb_run_id": "gradlab-0123456789abcdef01234567",
    }
    config = EnvConfig(
        env_provider="ale-py",
        game="breakout",
        state=None,
    )

    with (
        tempfile.TemporaryDirectory() as tmp,
        patch("gradlab.wandb_publisher.load_wandb_env"),
        patch.dict(
            sys.modules,
            {
                "wandb": SimpleNamespace(
                    init=fake_init,
                    Settings=lambda **kwargs: kwargs,
                )
            },
        ),
    ):
        _start_wandb(train_config, run_dir=tmp, config=config)

    assert captured["project"] == "Breakout-Atari2600-v0"
    assert captured["group"] == "bx0123456789abcdef"
    assert captured["id"] == "gradlab-0123456789abcdef01234567"
    assert captured["name"] == train_config["wandb_display_name"]
    assert captured["config"]["wandb_project"] == "Breakout-Atari2600-v0"
    assert captured["config"]["game_family"] == "Atari2600-Breakout"
    assert captured["config"]["environment"]["env_id"] == "ale-py:breakout"
    assert "environment_hash" in captured["config"]
    assert "game_family:Atari2600-Breakout" in captured["tags"]
    assert captured["settings"]["x_server_side_expand_glob_metrics"] is False


def test_init_wandb_requires_display_name() -> None:
    captured = {}

    class FakeRun:
        def define_metric(self, *_args, **_kwargs) -> None:
            return None

    train_config = {
        "wandb_tags": "",
        "wandb_entity": "entity",
        "wandb_project": None,
        "wandb_group": "local-group",
        "run_name": "local-run-name",
        "run_description": "local identity canary",
        "wandb_mode": "offline",
        "wandb_run_id": "gradlab-0123456789abcdef01234567",
    }
    config = EnvConfig(env_provider="ale-py", game="breakout", state=None)

    with (
        tempfile.TemporaryDirectory() as tmp,
        patch("gradlab.wandb_publisher.load_wandb_env"),
        patch.dict(
            sys.modules,
            {
                "wandb": SimpleNamespace(
                    init=lambda **kwargs: captured.update(kwargs) or FakeRun(),
                    Settings=lambda **kwargs: kwargs,
                )
            },
        ),
    ):
        with pytest.raises(ValueError, match="wandb_display_name"):
            _start_wandb(train_config, run_dir=tmp, config=config)

    assert captured == {}


@pytest.mark.parametrize(
    ("display_name", "expected_name"),
    [
        ("Level1-1__ppo__s7__01234567", "Level1-1__ppo__s7__01234567"),
        (None, None),
    ],
)
def test_resume_wandb_requires_or_uses_display_name(
    display_name: str | None,
    expected_name: str | None,
) -> None:
    captured = {}

    class FakeRun:
        def define_metric(self, *_args, **_kwargs) -> None:
            return None

    train_config = {
        "wandb_run_id": "gradlab-0123456789abcdef0123456789abcdef",
        "wandb_entity": "entity",
        "wandb_project": "SuperMarioBros-Nes-v0",
        "wandb_display_name": display_name,
        "wandb_group": "cohort::SuperMarioBros-Nes-v0/Level1-1::ppo::base",
        "wandb_mode": "offline",
        "run_name": "gradlab-0123456789abcdef0123456789abcdef",
        "env_provider": "supermariobrosnes-turbo",
        "game": "SuperMarioBros-Nes-v0",
        "metrics_schema_version": METRICS_SCHEMA_VERSION,
    }
    fake_wandb = SimpleNamespace(
        init=lambda **kwargs: captured.update(kwargs) or FakeRun(),
        Settings=lambda **kwargs: kwargs,
    )

    with (
        patch("gradlab.wandb_publisher.load_wandb_env"),
        patch.dict(sys.modules, {"wandb": fake_wandb}),
    ):
        if expected_name is None:
            with pytest.raises(ValueError, match="wandb_display_name"):
                WandbProjector.resume(train_config)
        else:
            WandbProjector.resume(train_config)

    if expected_name is None:
        assert captured == {}
        return

    assert captured["name"] == expected_name
    assert captured["id"] == train_config["wandb_run_id"]
    assert captured["group"] == train_config["wandb_group"]
    assert captured["settings"]["x_update_finish_state"] is True
    assert captured["settings"]["x_server_side_expand_glob_metrics"] is False


def test_resume_wandb_requires_current_metrics_schema() -> None:
    train_config = {
        "wandb_run_id": "gradlab-0123456789abcdef0123456789abcdef",
        "wandb_entity": "entity",
        "wandb_project": "SuperMarioBros-Nes-v0",
        "wandb_display_name": "Level1-1__ppo__s7__01234567",
        "wandb_group": "cohort::SuperMarioBros-Nes-v0/Level1-1::ppo::base",
        "wandb_mode": "offline",
        "env_provider": "supermariobrosnes-turbo",
        "game": "SuperMarioBros-Nes-v0",
    }
    with (
        patch("gradlab.wandb_publisher.load_wandb_env"),
        patch.dict(
            sys.modules,
            {
                "wandb": SimpleNamespace(
                    init=lambda **_kwargs: object(),
                    Settings=lambda **kwargs: kwargs,
                )
            },
        ),
        pytest.raises(ValueError, match="unsupported metrics schema version"),
    ):
        WandbProjector.resume(train_config)


def test_wandb_finish_has_a_hard_timeout() -> None:
    release = threading.Event()

    class FakeRun:
        def finish(self) -> None:
            release.wait()

    projector = WandbProjector(FakeRun())
    with pytest.raises(TimeoutError, match="did not finish uploading"):
        projector.close(timeout_seconds=0.01)
    release.set()


def test_wandb_finish_can_publish_a_failed_terminal_state() -> None:
    exit_codes: list[int] = []

    class FakeRun:
        def finish(self, *, exit_code: int) -> None:
            exit_codes.append(exit_code)

    WandbProjector(FakeRun()).close(timeout_seconds=1, exit_code=1)

    assert exit_codes == [1]
