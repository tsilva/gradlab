from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from rlab.artifacts import (
    checkpoint_step,
    load_model_metadata,
    load_playback_env_config,
    playback_env_config,
    write_model_metadata,
)
from rlab.env import EnvConfig
from rlab.eval import build_parser as build_eval_parser
from rlab.model_sources import (
    is_huggingface_model_ref,
    model_source_ref,
    parse_huggingface_model_ref,
    positional_model_source_arg,
)
from rlab.play import (
    _PlaybackSession,
    build_parser as build_play_parser,
    playback_should_end_episode,
    task_conditioning_change_message,
)
from rlab.play_termination import (
    configured_termination_ids,
    termination_condition_payload,
    with_enabled_termination_conditions,
)


def _metadata_args() -> argparse.Namespace:
    return argparse.Namespace(
        rom_asset_manifest=None,
        run_name="run",
        run_description="test",
        wandb_run_id="rlab-" + "a" * 32,
        wandb_project="Game-v0",
        campaign_id="",
        game_family="Game",
        goal_slug="game/goal",
        goal_path="experiments/goals/game/goal/_goal.yaml",
        goal_sha256="b" * 64,
        goal_contract_sha256="c" * 64,
        effective_goal_contract_sha256="c" * 64,
        reward_program_kind="task",
        reward_program_revision="1",
        reward_shape="default",
        reward_shape_sha256="d" * 64,
        reward_shape_is_default=True,
        recipe_slug="ppo",
        recipe_path="recipe.yaml",
        recipe_sha256="e" * 64,
        runtime_image_ref="docker:example/image@sha256:" + "f" * 64,
        seed=123,
        source_sha="1" * 40,
        training_backend_id="sb3.ppo",
        training_backend_config_hash="2" * 64,
        algorithm_id="ppo",
        model_class="stable_baselines3.ppo.ppo.PPO",
    )


def test_checkpoint_step_is_derived_from_learner_filename() -> None:
    assert checkpoint_step(Path("model_250000_steps.zip")) == 250_000
    assert checkpoint_step(Path("final_model.zip")) is None


def test_model_metadata_round_trips_playback_environment(tmp_path: Path) -> None:
    model = tmp_path / "model_250000_steps.zip"
    model.write_bytes(b"checkpoint")
    config = EnvConfig(env_provider="rlab", game="Bandit-v0", state=None)

    path = write_model_metadata(
        model,
        _metadata_args(),
        config,
        "checkpoint",
        checkpoint_step_value=250_000,
    )

    assert path is not None
    assert load_model_metadata(model)["checkpoint_step"] == 250_000
    assert load_playback_env_config(model).game == "Bandit-v0"


def test_continuous_play_removes_task_owned_termination() -> None:
    config = EnvConfig(
        env_provider="rlab",
        game="Bandit-v0",
        task={
            "termination": {
                "failure": ["loss"],
                "success": ["win"],
                "timeout": ["stalled"],
                "max_episode_steps": 100,
            },
            "events": {"stalled": {"signal": "x", "operation": "unchanged"}},
        },
    )

    continuous = playback_env_config(config, respect_task_termination=False)
    assert continuous.task["termination"]["failure"] == []
    assert continuous.task["termination"]["success"] == []
    assert continuous.task["termination"]["max_episode_steps"] == 0
    assert "stalled" not in continuous.task["events"]


def test_playback_termination_conditions_can_be_toggled_independently() -> None:
    config = EnvConfig(
        env_provider="rlab",
        game="Bandit-v0",
        task={
            "termination": {
                "failure": ["life_loss", "stalled"],
                "success": ["level_change"],
                "max_episode_steps": 4500,
            },
        },
    )

    assert configured_termination_ids(config) == (
        "event:life_loss",
        "event:stalled",
        "event:level_change",
        "limit:max_episode_steps",
    )

    updated = with_enabled_termination_conditions(
        config,
        ["event:life_loss", "event:stalled", "limit:max_episode_steps"],
    )

    assert updated.task["termination"]["failure"] == ["life_loss", "stalled"]
    assert updated.task["termination"]["success"] == []
    assert updated.task["termination"]["max_episode_steps"] == 4500
    payload = termination_condition_payload(config, updated)
    enabled = {condition["id"] for condition in payload if condition["enabled"]}
    assert enabled == {
        "event:life_loss",
        "event:stalled",
        "limit:max_episode_steps",
    }


def test_playback_termination_conditions_reject_unknown_ids() -> None:
    config = EnvConfig(
        env_provider="rlab",
        game="Bandit-v0",
        task={"termination": {"success": ["win"]}},
    )

    with pytest.raises(ValueError, match="unknown termination condition"):
        with_enabled_termination_conditions(config, ["event:missing"])


def test_playback_session_rebuilds_environment_for_termination_change() -> None:
    class FakeEnv:
        def __init__(self) -> None:
            self.action_space = object()
            self.reset_infos = [{}]
            self.closed = False
            self.seed_value = None

        def seed(self, seed: int) -> None:
            self.seed_value = seed

        def reset(self):
            return object()

        def close(self) -> None:
            self.closed = True

    base_config = EnvConfig(
        env_provider="rlab",
        game="Bandit-v0",
        task={
            "termination": {
                "failure": ["life_loss"],
                "success": ["level_change"],
            },
        },
    )
    initial_env = FakeEnv()
    created: list[tuple[EnvConfig, int, FakeEnv]] = []

    def env_factory(config: EnvConfig, seed: int) -> FakeEnv:
        env = FakeEnv()
        created.append((config, seed, env))
        return env

    session = _PlaybackSession(
        model=SimpleNamespace(),
        env=initial_env,
        config=base_config,
        initial_seed=1_000_000_000,
        attributor=None,
        attribution_mode="none",
        attribution_interval=1,
        attribution_opacity=0.45,
        env_factory=env_factory,
        termination_base_config=base_config,
        termination_source="evaluation",
    )

    session.set_termination_conditions(["event:life_loss"])

    assert initial_env.closed is True
    assert len(created) == 1
    updated_config, seed, replacement_env = created[0]
    assert seed == 1_000_000_000
    assert updated_config.task["termination"]["failure"] == ["life_loss"]
    assert updated_config.task["termination"]["success"] == []
    assert replacement_env.seed_value == 1_000_000_000
    assert session.env is replacement_env
    assert session.termination_source == "evaluation"


def test_public_source_parsers_exclude_wandb_artifacts() -> None:
    manifest = (
        "https://models.example/runs/rlab-"
        + "a" * 32
        + "/checkpoints/250000-"
        + "b" * 64
        + "/manifest.json"
    )
    assert positional_model_source_arg(manifest) == manifest
    assert positional_model_source_arg("hf://owner/repo") == "hf://owner/repo"
    with pytest.raises(argparse.ArgumentTypeError):
        positional_model_source_arg("entity/project/artifact:v1")

    eval_help = build_eval_parser().format_help()
    play_help = build_play_parser().format_help()
    assert "--artifact" not in eval_help
    assert "W&B artifact" not in play_help


def test_huggingface_refs_parse_and_resolve_from_cli_namespace() -> None:
    assert is_huggingface_model_ref("hf://owner/repo@deadbeef/model.zip")
    assert parse_huggingface_model_ref("hf://owner/repo@deadbeef/model.zip") == (
        "owner/repo",
        "model.zip",
        "deadbeef",
    )
    args = argparse.Namespace(
        model_ref="hf://owner/repo",
        artifact_ref=None,
        model=None,
    )
    assert model_source_ref(args) == "hf://owner/repo"


def test_playback_only_ends_on_environment_done() -> None:
    assert not playback_should_end_episode(False, False, True)
    assert playback_should_end_episode(True, False, False)
    assert playback_should_end_episode(False, True, False)


def test_task_conditioning_message_contains_explicit_one_hot() -> None:
    message = task_conditioning_change_message(
        episode=1,
        step=2,
        old_task="A",
        new_task="B",
        task_index=1,
        task_count=3,
    )
    assert "one_hot=[0, 1, 0]" in message
