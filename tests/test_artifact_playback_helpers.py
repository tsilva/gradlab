from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from gradlab.artifacts import (
    checkpoint_step,
    playback_env_config,
)
from gradlab.env import EnvConfig, resolve_env_config
from gradlab.env_config import env_config_from_mapping
from gradlab.env_metadata import env_config_from_config_dict, training_metadata
from gradlab.eval import build_parser as build_eval_parser
from gradlab.model_sources import (
    ResolvedModelSource,
    is_huggingface_model_ref,
    model_source_ref,
    parse_huggingface_model_ref,
    positional_model_source_arg,
    resolve_model_source,
)
from gradlab.play import (
    _PlaybackSession,
    build_parser as build_play_parser,
    playback_should_end_episode,
    task_conditioning_change_message,
)
from gradlab.play_termination import (
    configured_termination_ids,
    termination_condition_payload,
    with_enabled_termination_conditions,
)
from gradlab.policy_bundle import (
    build_model_document,
    build_recipe_document,
    load_policy_bundle_from_checkpoint,
    playback_contract,
    write_canonical_json,
)
from gradlab.recipe_documents import compose_train_document
from gradlab.training_backend import training_backend_config_hash


def test_checkpoint_step_is_derived_from_learner_filename() -> None:
    assert checkpoint_step(Path("model_250000_steps.zip")) == 250_000
    assert checkpoint_step(Path("final_model.zip")) is None


def test_model_metadata_round_trips_playback_environment(tmp_path: Path) -> None:
    model = tmp_path / "model_250000_steps.zip"
    model.write_bytes(b"checkpoint")
    materialized = compose_train_document(
        Path("experiments/goals/gradlab__bandit/_goal.yaml"),
        Path("experiments/goals/gradlab__bandit/recipes/ppo.yaml"),
    )
    recipe_document = build_recipe_document(
        materialized,
        repo_root=Path.cwd(),
        source_commit="a" * 40,
        run_description="Versioned playback metadata regression.",
        seed=123,
        runtime_image_ref="docker:example/image@sha256:" + "f" * 64,
    )
    train_config = recipe_document["recipe"]["train_config"]
    config = resolve_env_config(env_config_from_mapping(train_config))
    recipe_path = write_canonical_json(
        model.with_suffix(".recipe.json"),
        recipe_document,
    )
    write_canonical_json(
        model.with_suffix(".model.json"),
        build_model_document(
            model,
            recipe_path,
            {
                "kind": "checkpoint",
                "checkpoint_step": 250_000,
                "algorithm_id": "ppo",
                "model_class": "stable_baselines3.ppo.ppo.PPO",
                "training_backend_id": "sb3.ppo",
                "training_backend_config_hash": training_backend_config_hash(
                    train_config
                ),
                "training_metadata": {
                    **training_metadata(config),
                    "preprocessing": {"legacy": True},
                    "action": {"legacy": True},
                },
            },
        ),
    )

    bundle = load_policy_bundle_from_checkpoint(model)
    assert bundle is not None
    assert bundle.model["checkpoint"]["step"] == 250_000
    assert bundle.recipe["recipe"]["environment"]["preprocessing"] == recipe_document[
        "recipe"
    ]["environment"]["preprocessing"]
    saved_config = playback_contract(bundle.recipe, mode="training")["environment"]
    playback_config = env_config_from_config_dict(saved_config)
    assert playback_config is not None
    assert playback_config.game == "Bandit-v0"


def test_continuous_play_removes_task_owned_termination() -> None:
    config = EnvConfig(
        env_provider="gradlab",
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
        env_provider="gradlab",
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
        env_provider="gradlab",
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
        env_provider="gradlab",
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
        "https://models.example/runs/gradlab-"
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
    assert is_huggingface_model_ref("hf://owner/repo@deadbeef")
    assert parse_huggingface_model_ref("hf://owner/repo@deadbeef") == (
        "owner/repo",
        "deadbeef",
    )
    with pytest.raises(ValueError, match="owner/repo"):
        parse_huggingface_model_ref("hf://owner/repo/model.zip")
    args = argparse.Namespace(
        model_ref="hf://owner/repo",
        artifact_ref=None,
        model=None,
    )
    assert model_source_ref(args) == "hf://owner/repo"


def test_model_source_resolution_has_one_kind_aware_owner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_ref = (
        "https://models.example/runs/gradlab-"
        + "a" * 32
        + "/checkpoints/1-"
        + "b" * 64
        + "/manifest.json"
    )
    bundle = SimpleNamespace()
    expected = ResolvedModelSource(tmp_path / "resolved.zip", bundle=bundle)
    monkeypatch.setattr(
        "gradlab.model_sources.download_public_checkpoint_manifest_source",
        lambda ref, *, root: expected,
    )

    resolved = resolve_model_source(
        "manifest",
        manifest_ref,
        public_root=tmp_path / "public",
        hf_root=tmp_path / "hf",
    )

    assert resolved is expected
    assert resolved.artifact_ref == manifest_ref

    local = tmp_path / "local.zip"
    local.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        "gradlab.model_sources.load_policy_bundle_from_checkpoint",
        lambda path: bundle,
    )
    resolved = resolve_model_source(
        "local",
        str(local),
        public_root=tmp_path / "public",
        hf_root=tmp_path / "hf",
    )
    assert resolved.model_path == local
    assert resolved.artifact_ref is None


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
