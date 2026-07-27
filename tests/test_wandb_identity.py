from __future__ import annotations

import sys
import tempfile
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from rlab.env import EnvConfig
from rlab.wandb_publisher import WandbProjector, _start_wandb
from rlab.wandb_utils import (
    canonical_wandb_environment,
    game_family_for_environment,
    resolve_wandb_project,
)


@pytest.mark.parametrize(
    ("provider", "game", "project", "family"),
    [
        ("rlab", "Bandit-v0", "Bandit-v0", "Bandit"),
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


def test_explicit_project_wins_and_unknown_environment_falls_back() -> None:
    assert (
        resolve_wandb_project("custom-project", "breakout", env_provider="ale-py")
        == "custom-project"
    )
    assert resolve_wandb_project(None, "CustomNativeVector-v0", env_provider="gymnasium") == (
        "CustomNativeVector-v0"
    )
    assert game_family_for_environment("gymnasium", "CustomNativeVector-v0") == (
        "custom-native-vector-v0"
    )


def test_historical_env_id_fallbacks_are_preserved() -> None:
    assert canonical_wandb_environment(None, "SuperMarioBros-Nes-v0") == (
        "SuperMarioBros-Nes-v0",
        "NES-SuperMarioBros",
    )
    assert canonical_wandb_environment("legacy-provider", "breakout") == (
        "breakout",
        "Atari2600-Breakout",
    )
    with pytest.raises(ValueError, match="no registered canonical game family"):
        game_family_for_environment(None, "SuperMarioBros-Nes-v0", strict=True)


def test_init_wandb_records_resolved_identity_and_submission_group() -> None:
    captured = {}

    class FakeRun:
        def define_metric(self, *_args, **_kwargs) -> None:
            return None

    def fake_init(**kwargs):
        captured.update(kwargs)
        return FakeRun()

    train_config = {
        "wandb": True,
        "wandb_tags": "goal_id:alepy__breakout,recipe_id:base",
        "wandb_entity": "entity",
        "wandb_project": None,
        "wandb_display_name": "breakout__base__s123__01234567",
        "wandb_group": "bx0123456789abcdef",
        "run_name": "bx0123456789abcdef-base-s123-20260714T120000Z",
        "run_description": "offline identity canary",
        "wandb_mode": "offline",
        "wandb_run_id": "rlab-0123456789abcdef01234567",
    }
    config = EnvConfig(
        env_provider="ale-py",
        game="breakout",
        state=None,
    )

    with (
        tempfile.TemporaryDirectory() as tmp,
        patch("rlab.wandb_publisher.load_wandb_env"),
        patch.dict(sys.modules, {"wandb": SimpleNamespace(init=fake_init)}),
    ):
        _start_wandb(train_config, run_dir=tmp, config=config)

    assert captured["project"] == "Breakout-Atari2600-v0"
    assert captured["group"] == "bx0123456789abcdef"
    assert captured["id"] == "rlab-0123456789abcdef01234567"
    assert captured["name"] == train_config["wandb_display_name"]
    assert captured["config"]["wandb_project"] == "Breakout-Atari2600-v0"
    assert captured["config"]["game_family"] == "Atari2600-Breakout"
    assert captured["config"]["environment"]["env_id"] == "ale-py:breakout"
    assert "environment_hash" in captured["config"]
    assert "game_family:Atari2600-Breakout" in captured["tags"]


def test_init_wandb_falls_back_to_run_name_for_legacy_config() -> None:
    captured = {}

    class FakeRun:
        def define_metric(self, *_args, **_kwargs) -> None:
            return None

    train_config = {
        "wandb": True,
        "wandb_tags": "",
        "wandb_entity": "entity",
        "wandb_project": None,
        "wandb_group": "legacy-group",
        "run_name": "legacy-run-name",
        "run_description": "legacy identity canary",
        "wandb_mode": "offline",
        "wandb_run_id": "rlab-0123456789abcdef01234567",
    }
    config = EnvConfig(env_provider="ale-py", game="breakout", state=None)

    with (
        tempfile.TemporaryDirectory() as tmp,
        patch("rlab.wandb_publisher.load_wandb_env"),
        patch.dict(
            sys.modules,
            {"wandb": SimpleNamespace(init=lambda **kwargs: captured.update(kwargs) or FakeRun())},
        ),
    ):
        _start_wandb(train_config, run_dir=tmp, config=config)

    assert captured["name"] == train_config["run_name"]
    assert captured["group"] == "legacy-group"


@pytest.mark.parametrize(
    ("display_name", "expected_name"),
    [
        ("Level1-1__ppo__s7__01234567", "Level1-1__ppo__s7__01234567"),
        (None, "rlab-0123456789abcdef0123456789abcdef"),
    ],
)
def test_resume_wandb_prefers_display_name_with_legacy_fallback(
    display_name: str | None,
    expected_name: str,
) -> None:
    captured = {}

    class FakeRun:
        def define_metric(self, *_args, **_kwargs) -> None:
            return None

    train_config = {
        "wandb_run_id": "rlab-0123456789abcdef0123456789abcdef",
        "wandb_entity": "entity",
        "wandb_project": "SuperMarioBros-Nes-v0",
        "wandb_display_name": display_name,
        "wandb_group": "cohort::SuperMarioBros-Nes-v0/Level1-1::ppo::base",
        "wandb_mode": "offline",
        "run_name": "rlab-0123456789abcdef0123456789abcdef",
        "env_provider": "supermariobrosnes-turbo",
        "game": "SuperMarioBros-Nes-v0",
    }
    fake_wandb = SimpleNamespace(
        init=lambda **kwargs: captured.update(kwargs) or FakeRun(),
        Settings=lambda **kwargs: kwargs,
    )

    with (
        patch("rlab.wandb_publisher.load_wandb_env"),
        patch.dict(sys.modules, {"wandb": fake_wandb}),
    ):
        WandbProjector.resume(train_config)

    assert captured["name"] == expected_name
    assert captured["id"] == train_config["wandb_run_id"]
    assert captured["group"] == train_config["wandb_group"]


def test_wandb_finish_has_a_hard_timeout() -> None:
    release = threading.Event()

    class FakeRun:
        def finish(self) -> None:
            release.wait()

    projector = WandbProjector(FakeRun())
    with pytest.raises(TimeoutError, match="did not finish uploading"):
        projector.close(timeout_seconds=0.01)
    release.set()
