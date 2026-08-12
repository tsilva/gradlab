from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from gradlab.action_contract import declared_action_contract
from gradlab.artifacts import install_model_bundle
from gradlab.env import resolve_env_config
from gradlab.env_config import env_config_from_mapping
from gradlab.env_metadata import training_metadata
from gradlab.policy_bundle import (
    MODEL_DOCUMENT_TYPE,
    PolicyDocumentError,
    UnsupportedPolicyDocumentVersion,
    build_model_document,
    build_recipe_document,
    canonical_json_bytes,
    canonical_json_sha256,
    critic_value_contract,
    evaluation_contract,
    playback_contract_sha256,
    playback_contract,
    load_policy_bundle,
    load_policy_bundle_from_checkpoint,
    load_recipe_document,
    validate_recipe_document,
    write_canonical_json,
)
from gradlab.eval_runner import normalized_evaluation_request
from gradlab.recipe_documents import (
    compose_resolved_train_documents,
    compose_train_document,
)
from gradlab.train_config import validate_and_normalize_train_config
from gradlab.training_backend import training_backend_config, training_backend_config_hash


GOAL = Path("experiments/goals/SuperMarioBros-Nes-v0/Level1-1/_goal.yaml")
RECIPE = Path("experiments/goals/SuperMarioBros-Nes-v0/Level1-1/recipes/ppo.yaml")
LEVEL1_3_GOAL = Path("experiments/goals/SuperMarioBros-Nes-v0/Level1-3/_goal.yaml")
LEVEL1_3_TRAIN_CLEAR_RECIPE = LEVEL1_3_GOAL.parent / "recipes/ppo-train-clear-100.yaml"
RUNTIME = "docker:ghcr.io/tsilva/gradlab/gradlab-train@sha256:" + "b" * 64
BREAKOUT_GOAL = Path("experiments/goals/Breakout-Atari2600-v0/_goal.yaml")
BREAKOUT_RECIPES = tuple(sorted((BREAKOUT_GOAL.parent / "recipes").glob("*.yaml")))
BANDIT_GOAL = Path("experiments/goals/gradlab__bandit/_goal.yaml")
BANDIT_RECIPE = BANDIT_GOAL.parent / "recipes/ppo.yaml"
VIZDOOM_GOAL = Path("experiments/goals/VizdoomBasic-v1/_goal.yaml")
VIZDOOM_RECIPE = VIZDOOM_GOAL.parent / "recipes/ppo.yaml"
MARIO_ASSET = {
    "schema_version": 2,
    "game": "SuperMarioBros-Nes-v0",
    "filename": "mario.nes",
    "size_bytes": 1024,
    "sha256": "c" * 64,
    "provider_rom_identity": "d" * 40,
    "provider_rom_identity_algorithm": "sha1-provider-body-v1",
    "object_uri": "s3://private-bucket/mario.nes",
}


def bind_mario_asset(resolved) -> None:
    resolved.effective["train_config"]["rom_asset_manifest"] = deepcopy(MARIO_ASSET)
    resolved.base["train_config"]["rom_asset_manifest"] = deepcopy(MARIO_ASSET)


def level1_1_recipe_document(*, seed: int = 7) -> dict:
    resolved = compose_resolved_train_documents(
        GOAL,
        RECIPE,
        source_sha="a" * 40,
    )
    bind_mario_asset(resolved)
    return build_recipe_document(
        resolved.effective,
        repo_root=Path.cwd(),
        source_commit="a" * 40,
        run_description=f"Level1-1 PPO seed {seed}",
        seed=seed,
        runtime_image_ref=RUNTIME,
        base_materialized_recipe=resolved.base,
        canonical_goal=resolved.canonical_goal,
    )


def test_recipe_v4_embeds_verified_goal_and_recipe_bases() -> None:
    resolved = compose_resolved_train_documents(
        BANDIT_GOAL,
        BANDIT_RECIPE,
        recipe_overrides=("train.backend.config.gamma=0.97",),
        source_sha="a" * 40,
    )
    document = build_recipe_document(
        resolved.effective,
        repo_root=Path.cwd(),
        source_commit="a" * 40,
        run_description="Bandit recipe v4 proof",
        seed=7,
        runtime_packages=("gradlab==0.1.0",),
        base_materialized_recipe=resolved.base,
        canonical_goal=resolved.canonical_goal,
    )

    assert document["format_version"] == 4
    assert document["recipe"]["train_config"]["metrics_schema_version"] == 19
    assert document["resolution"]["recipe"]["base"]["train_config"]["metrics_schema_version"] == 19
    assert document["resolution"]["goal"]["base"] == resolved.canonical_goal
    assert document["resolution"]["recipe"]["variant_id"].startswith("v-")
    assert (
        document["resolution"]["recipe"]["base"]["train_config"]["training_backend"]["config"][
            "gamma"
        ]
        != document["recipe"]["train_config"]["training_backend"]["config"]["gamma"]
    )
    assert validate_recipe_document(document) == document

    tampered = deepcopy(document)
    tampered["resolution"]["recipe"]["base"]["train_config"]["training_backend"]["config"][
        "gamma"
    ] = 0.5
    with pytest.raises(PolicyDocumentError, match="base_sha256"):
        validate_recipe_document(tampered)


def test_portable_recipe_reader_preserves_source_bound_retired_backend_options() -> None:
    document = level1_1_recipe_document()
    base_recipe = document["resolution"]["recipe"]["base"]

    for recipe in (document["recipe"], base_recipe):
        train_config = recipe["train_config"]
        train_config.pop("policy_model")
        backend_config = train_config["training_backend"]["config"]
        backend_config["policy_net_arch"] = ""
        backend_config["value_net_arch"] = ""

    document["resolution"]["recipe"]["base_sha256"] = canonical_json_sha256(base_recipe)
    document["resolution"]["recipe"]["effective_sha256"] = canonical_json_sha256(document["recipe"])

    assert validate_recipe_document(document) == document


def test_portable_recipe_reader_preserves_historical_failure_plateau() -> None:
    document = level1_1_recipe_document()
    base_recipe = document["resolution"]["recipe"]["base"]

    for recipe in (document["recipe"], base_recipe):
        recipe["train_config"]["early_stop"]["conditions"]["return_plateau"] = {
            "metric": "train/episode/return/shaped/origin/target/rolling/mean",
            "trigger": "no_improvement",
            "direction": "maximize",
            "min_delta": 0.01,
            "delta_mode": "relative",
            "start_after_steps": 1_000_000,
            "patience_steps": 1_000_000,
            "outcome": "failure",
            "action": "stop",
        }

    document["resolution"]["recipe"]["base_sha256"] = canonical_json_sha256(base_recipe)
    document["resolution"]["recipe"]["effective_sha256"] = canonical_json_sha256(document["recipe"])

    assert validate_recipe_document(document) == document


def test_portable_recipe_reader_preserves_source_bound_metrics_schema() -> None:
    document = level1_1_recipe_document()
    base_recipe = document["resolution"]["recipe"]["base"]

    for recipe in (document["recipe"], base_recipe):
        recipe["train_config"]["metrics_schema_version"] = 18
        conditions = recipe["train_config"]["early_stop"]["conditions"]
        conditions["clear_100"]["metric"] = (
            "train/outcome/success/across_starts/window_100/rate/min"
        )
        conditions["return_plateau"] = {
            "metric": "train/episode/return/shaped/from/target/rolling_up_to_100/mean",
            "trigger": "no_improvement",
            "direction": "maximize",
            "min_delta": 0.01,
            "delta_mode": "absolute",
            "start_after_steps": 25_000_000,
            "patience_steps": 25_000_000,
            "outcome": "neutral",
            "action": "observe",
        }

    document["resolution"]["recipe"]["base_sha256"] = canonical_json_sha256(base_recipe)
    document["resolution"]["recipe"]["effective_sha256"] = canonical_json_sha256(document["recipe"])

    assert validate_recipe_document(document) == document


def test_portable_recipe_reader_rejects_malformed_source_bound_metric_name() -> None:
    document = level1_1_recipe_document()
    base_recipe = document["resolution"]["recipe"]["base"]

    for recipe in (document["recipe"], base_recipe):
        recipe["train_config"]["metrics_schema_version"] = 18
        recipe["train_config"]["early_stop"]["conditions"]["clear_100"]["metric"] = (
            "train/episode return"
        )

    document["resolution"]["recipe"]["base_sha256"] = canonical_json_sha256(base_recipe)
    document["resolution"]["recipe"]["effective_sha256"] = canonical_json_sha256(document["recipe"])

    with pytest.raises(PolicyDocumentError, match="is not a registered metric"):
        validate_recipe_document(document)


def test_portable_recipe_reader_rejects_unregistered_current_metric_name() -> None:
    document = level1_1_recipe_document()
    base_recipe = document["resolution"]["recipe"]["base"]

    for recipe in (document["recipe"], base_recipe):
        recipe["train_config"]["early_stop"]["conditions"]["clear_100"]["metric"] = (
            "train/unregistered/value"
        )

    document["resolution"]["recipe"]["base_sha256"] = canonical_json_sha256(base_recipe)
    document["resolution"]["recipe"]["effective_sha256"] = canonical_json_sha256(document["recipe"])

    with pytest.raises(PolicyDocumentError, match="is not a registered metric"):
        validate_recipe_document(document)


@pytest.mark.parametrize("invalid", [0, -1, True, "15"])
def test_portable_recipe_reader_rejects_invalid_metrics_schema(invalid: object) -> None:
    document = level1_1_recipe_document()
    document["recipe"]["train_config"]["metrics_schema_version"] = invalid

    with pytest.raises(PolicyDocumentError, match="must be a positive integer"):
        validate_recipe_document(document)


def test_wandb_display_name_is_not_part_of_portable_recipe() -> None:
    resolved = compose_resolved_train_documents(GOAL, RECIPE, source_sha="a" * 40)
    bind_mario_asset(resolved)
    resolved.effective["train_config"]["wandb_display_name"] = "Level1-1__ppo__s7__01234567"

    document = build_recipe_document(
        resolved.effective,
        repo_root=Path.cwd(),
        source_commit="a" * 40,
        run_description="W&B presentation identity regression",
        seed=7,
        runtime_image_ref=RUNTIME,
        base_materialized_recipe=resolved.base,
        canonical_goal=resolved.canonical_goal,
    )

    assert "wandb_display_name" not in document["recipe"]["train_config"]


def test_noncurrent_derived_acceptance_flag_is_rejected() -> None:
    document = level1_1_recipe_document()
    noncurrent = deepcopy(document)
    noncurrent["recipe"]["train_config"]["stop_on_acceptance"] = True
    base_recipe = noncurrent["resolution"]["recipe"]["base"]
    base_recipe["train_config"]["stop_on_acceptance"] = True
    noncurrent["resolution"]["recipe"]["base_sha256"] = canonical_json_sha256(base_recipe)
    noncurrent["resolution"]["recipe"]["effective_sha256"] = canonical_json_sha256(
        noncurrent["recipe"]
    )

    with pytest.raises(PolicyDocumentError, match="stop_on_acceptance"):
        validate_recipe_document(noncurrent)


@pytest.mark.parametrize("recipe_path", BREAKOUT_RECIPES)
def test_breakout_bundle_is_playable_but_has_no_evaluation_contract(
    recipe_path: Path,
) -> None:
    resolved = compose_resolved_train_documents(
        BREAKOUT_GOAL,
        recipe_path,
        source_sha="a" * 40,
    )
    document = build_recipe_document(
        resolved.effective,
        repo_root=Path.cwd(),
        source_commit="a" * 40,
        run_description="training-only Breakout",
        seed=7,
        runtime_image_ref=RUNTIME,
        base_materialized_recipe=resolved.base,
        canonical_goal=resolved.canonical_goal,
    )

    assert document["recipe"]["train_config"]["checkpoint_eval_backend"] == "none"
    assert "eval" not in document["recipe"]
    assert playback_contract(document)["environment"]["game"] == "Breakout-Atari2600-v0"
    assert len(playback_contract_sha256(document)) == 64
    with pytest.raises(PolicyDocumentError, match="no evaluation contract"):
        evaluation_contract(document)


def test_level1_3_training_clear_bundle_omits_eval_and_preserves_early_stop() -> None:
    resolved = compose_resolved_train_documents(
        LEVEL1_3_GOAL,
        LEVEL1_3_TRAIN_CLEAR_RECIPE,
        source_sha="a" * 40,
    )
    asset = {
        "schema_version": 2,
        "game": "SuperMarioBros-Nes-v0",
        "filename": "mario.nes",
        "size_bytes": 1024,
        "sha256": "c" * 64,
        "provider_rom_identity": "d" * 40,
        "provider_rom_identity_algorithm": "sha1-provider-body-v1",
        "object_uri": "s3://private-bucket/mario.nes",
    }
    resolved.effective["train_config"]["rom_asset_manifest"] = asset
    resolved.base["train_config"]["rom_asset_manifest"] = deepcopy(asset)
    document = build_recipe_document(
        resolved.effective,
        repo_root=Path.cwd(),
        source_commit="a" * 40,
        run_description="Level1-3 training-only 100-of-100 clear-rate run",
        seed=123,
        runtime_image_ref=RUNTIME,
        base_materialized_recipe=resolved.base,
        canonical_goal=resolved.canonical_goal,
    )

    recipe = document["recipe"]
    assert "eval" not in recipe
    assert recipe["train_config"]["checkpoint_eval_backend"] == "none"
    assert set(recipe["train_config"]["early_stop"]["conditions"]) == {
        "clear_100",
    }
    assert recipe["train_config"]["early_stop"]["conditions"]["clear_100"] == {
        "metric": "train/outcome/success/starts/all/rolling/rate/min",
        "trigger": "threshold",
        "outcome": "success",
        "action": "stop",
        "start_after_steps": 0,
        "patience_steps": 0,
        "operator": ">=",
        "progress_baseline": 0.0,
        "threshold": 1.0,
    }


def test_atomic_bundle_install_commits_only_a_complete_replayable_bundle(
    tmp_path: Path,
) -> None:
    recipe_document = level1_1_recipe_document()
    recipe_path = write_canonical_json(tmp_path / "recipe.json", recipe_document)
    train_config = dict(recipe_document["recipe"]["train_config"])
    config = resolve_env_config(env_config_from_mapping(train_config))
    model_path = tmp_path / "checkpoints" / "model_100_steps.zip"
    runtime_config = {
        **train_config,
        "recipe_json_path": str(recipe_path),
        "run_name": "atomic-bundle",
        "run_description": "Atomic bundle regression.",
        "runtime_image_ref": RUNTIME,
        "source_sha": "a" * 40,
        "algorithm_id": "ppo",
        "model_class": "stable_baselines3.ppo.ppo.PPO",
        "training_backend_id": "sb3.ppo",
        "training_backend_config_hash": training_backend_config_hash(train_config),
    }

    install_model_bundle(
        model_path,
        save_checkpoint=lambda path: path.write_bytes(b"checkpoint"),
        train_config=runtime_config,
        config=config,
        kind="checkpoint",
        checkpoint_step_value=100,
    )

    bundle = load_policy_bundle_from_checkpoint(model_path)
    assert bundle is not None
    assert bundle.model["checkpoint"]["step"] == 100
    assert bundle.checkpoint_path.read_bytes() == b"checkpoint"
    assert set(bundle.model["provenance"]["training_metadata"]) == {"versions"}
    assert bundle.recipe["recipe"]["environment"] == recipe_document["recipe"]["environment"]
    action = declared_action_contract(playback_contract(bundle.recipe)["environment"])
    assert action["preset"] == "basic"

    # An exact producer replay is accepted, but the same destination can never
    # be rebound to different checkpoint bytes.
    install_model_bundle(
        model_path,
        save_checkpoint=lambda path: path.write_bytes(b"checkpoint"),
        train_config=runtime_config,
        config=config,
        kind="checkpoint",
        checkpoint_step_value=100,
    )
    with pytest.raises(FileExistsError, match="conflicts with an existing committed bundle"):
        install_model_bundle(
            model_path,
            save_checkpoint=lambda path: path.write_bytes(b"different"),
            train_config=runtime_config,
            config=config,
            kind="checkpoint",
            checkpoint_step_value=100,
        )
    assert not list(model_path.parent.glob(".*.zip"))


@pytest.mark.parametrize(
    ("goal_path", "recipe_path", "expected_mode", "expected_preset"),
    (
        (BANDIT_GOAL, BANDIT_RECIPE, None, None),
        (VIZDOOM_GOAL, VIZDOOM_RECIPE, "custom_discrete", "minimal"),
        (GOAL, RECIPE, "custom_discrete", "basic"),
    ),
)
def test_bundle_metadata_reconstructs_provider_action_contract(
    tmp_path: Path,
    goal_path: Path,
    recipe_path: Path,
    expected_mode: str | None,
    expected_preset: str | None,
) -> None:
    checkpoint_path = tmp_path / f"{goal_path.parent.name}.zip"
    checkpoint_path.write_bytes(b"checkpoint")
    resolved = compose_resolved_train_documents(
        goal_path,
        recipe_path,
        source_sha="a" * 40,
    )
    if goal_path == GOAL:
        bind_mario_asset(resolved)
    recipe_document = build_recipe_document(
        resolved.effective,
        repo_root=Path.cwd(),
        source_commit="a" * 40,
        run_description="Bundle metadata action-contract regression.",
        seed=7,
        runtime_image_ref=RUNTIME,
        base_materialized_recipe=resolved.base,
        canonical_goal=resolved.canonical_goal,
    )
    train_config = recipe_document["recipe"]["train_config"]
    recipe_sidecar = write_canonical_json(
        checkpoint_path.with_suffix(".recipe.json"),
        recipe_document,
    )
    write_canonical_json(
        checkpoint_path.with_suffix(".model.json"),
        build_model_document(
            checkpoint_path,
            recipe_sidecar,
            {
                "kind": "checkpoint",
                "checkpoint_step": 100,
                "algorithm_id": "ppo",
                "model_class": "stable_baselines3.ppo.ppo.PPO",
                "training_backend_id": train_config["training_backend"]["id"],
                "training_backend_config_hash": training_backend_config_hash(train_config),
                "training_metadata": {"versions": {}},
            },
        ),
    )

    bundle = load_policy_bundle_from_checkpoint(checkpoint_path)
    assert bundle is not None
    action = declared_action_contract(playback_contract(bundle.recipe)["environment"])
    if expected_mode is None:
        assert action is None
    else:
        assert action["mode"] == expected_mode
        assert action["preset"] == expected_preset


def write_bundle(root: Path) -> None:
    checkpoint = root / "model.zip"
    checkpoint.write_bytes(b"checkpoint bytes")
    recipe_document = level1_1_recipe_document()
    recipe_path = write_canonical_json(root / "recipe.json", recipe_document)
    metadata = {
        "kind": "checkpoint",
        "checkpoint_step": 500_000,
        "algorithm_id": "ppo",
        "model_class": "stable_baselines3.ppo.ppo.PPO",
        "training_backend_id": "sb3.ppo",
        "training_backend_config_hash": training_backend_config_hash(
            recipe_document["recipe"]["train_config"]
        ),
        "repo_git_commit": "a" * 40,
    }
    write_canonical_json(
        root / "model.json",
        build_model_document(checkpoint, recipe_path, metadata),
    )


def test_level1_1_recipe_fixture_preserves_aligned_train_and_eval_contracts() -> None:
    document = level1_1_recipe_document()
    train_task = document["recipe"]["train_config"]["task"]
    eval_contract = evaluation_contract(document)
    eval_task = eval_contract["environment"]["task"]

    assert train_task["termination"]["failure"] == ["life_loss"]
    assert train_task["termination"]["timeout"] == ["stalled"]
    assert eval_task["termination"]["failure"] == ["life_loss"]
    assert eval_task["termination"]["timeout"] == ["stalled"]
    assert playback_contract(document)["environment"]["task"] == eval_task
    assert document["recipe"]["train_config"]["obs_crop"] == [32, 0, 0, 0]
    assert eval_contract["action_sampling"] == "stochastic"
    assert eval_contract["seed_protocol"] == "vector-lane-v1"
    assert eval_contract["episodes"] == 100


def test_evaluated_bundle_defaults_to_training_contract_and_exposes_eval_explicitly() -> None:
    resolved = compose_resolved_train_documents(
        VIZDOOM_GOAL,
        VIZDOOM_RECIPE,
        recipe_overrides=(
            "train.checkpoint_eval_backend=modal",
            "train.environment.task.reward.reward_clip=true",
        ),
        source_sha="a" * 40,
    )
    document = build_recipe_document(
        resolved.effective,
        repo_root=Path.cwd(),
        source_commit="a" * 40,
        run_description="training-faithful playback regression",
        seed=7,
        runtime_image_ref=RUNTIME,
        base_materialized_recipe=resolved.base,
        canonical_goal=resolved.canonical_goal,
    )

    training = playback_contract(document)
    evaluation = playback_contract(document, mode="evaluation")
    assert training["mode"] == "training"
    assert training["environment"]["task"]["reward"]["reward_clip"] == [-1.0, 1.0]
    assert evaluation["mode"] == "evaluation"
    assert evaluation["matches_training"] is True
    assert critic_value_contract(document)["discount"] == 0.99


def test_gradlab_ppo_recipe_value_contract_round_trips_for_checkpoint_playback() -> None:
    resolved = compose_resolved_train_documents(
        VIZDOOM_GOAL,
        VIZDOOM_RECIPE,
        recipe_overrides=("train.backend.id=gradlab.ppo",),
        source_sha="a" * 40,
    )
    document = build_recipe_document(
        resolved.effective,
        repo_root=Path.cwd(),
        source_commit="a" * 40,
        run_description="GradLab PPO checkpoint playback regression",
        seed=7,
        runtime_image_ref=RUNTIME,
        base_materialized_recipe=resolved.base,
        canonical_goal=resolved.canonical_goal,
    )

    stored = document["recipe"]["value_contract"]
    assert stored["truncation_bootstrap"] == "terminal-value"
    assert critic_value_contract(document) == stored


def test_evaluated_goal_preserves_manual_eval_when_automatic_eval_is_disabled() -> None:
    resolved = compose_resolved_train_documents(
        VIZDOOM_GOAL,
        VIZDOOM_RECIPE,
        source_sha="a" * 40,
    )
    assert resolved.effective["train_config"]["checkpoint_eval_backend"] == "none"

    document = build_recipe_document(
        resolved.effective,
        repo_root=Path.cwd(),
        source_commit="a" * 40,
        run_description="post-training manual evaluation regression",
        seed=123,
        runtime_image_ref=RUNTIME,
        base_materialized_recipe=resolved.base,
        canonical_goal=resolved.canonical_goal,
    )

    recipe = document["recipe"]
    assert recipe["train_config"]["checkpoint_eval_backend"] == "none"
    assert "eval" not in recipe
    assert "playback" in recipe
    contract = evaluation_contract(document)
    assert contract["episodes"] == 100
    assert contract["acceptance"] == resolved.effective["goal"]["eval"]["acceptance"]


def test_recipe_materializes_the_backend_config_executed_by_the_learner() -> None:
    resolved = compose_resolved_train_documents(GOAL, RECIPE, source_sha="a" * 40)
    bind_mario_asset(resolved)
    document = build_recipe_document(
        resolved.effective,
        repo_root=Path.cwd(),
        source_commit="a" * 40,
        run_description="normalized backend contract",
        seed=7,
        runtime_image_ref=RUNTIME,
        base_materialized_recipe=resolved.base,
        canonical_goal=resolved.canonical_goal,
    )

    recipe_train_config = document["recipe"]["train_config"]
    executed_train_config = validate_and_normalize_train_config(resolved.effective["train_config"])

    assert training_backend_config(recipe_train_config) == training_backend_config(
        executed_train_config
    )
    assert training_backend_config_hash(recipe_train_config) == training_backend_config_hash(
        executed_train_config
    )
    assert training_backend_config(recipe_train_config)["device"] == "auto"


def test_resume_approval_does_not_rebind_the_scientific_backend_config() -> None:
    materialized = compose_train_document(
        LEVEL1_3_GOAL,
        LEVEL1_3_TRAIN_CLEAR_RECIPE,
    )
    train_config = validate_and_normalize_train_config(materialized["train_config"])
    resumed = deepcopy(train_config)
    resumed_backend = resumed["training_backend"]["config"]
    resumed_backend.update(
        {
            "resume": "https://models.example/checkpoint/manifest.json",
            "resume_approval_hash": "a" * 64,
            "resume_manifest": [
                {
                    "path": "model.zip",
                    "sha256": "b" * 64,
                    "size_bytes": 123,
                }
            ],
        }
    )

    assert training_backend_config_hash(resumed) == training_backend_config_hash(train_config)
    resumed_backend["batch_size"] = int(resumed_backend["batch_size"]) // 2
    assert training_backend_config_hash(resumed) != training_backend_config_hash(train_config)


def test_recipe_materializes_the_environment_identity_executed_by_the_learner() -> None:
    document = level1_1_recipe_document()
    recipe = document["recipe"]
    effective_training_metadata = training_metadata(
        resolve_env_config(env_config_from_mapping(recipe["train_config"])),
        rom_asset_manifest=MARIO_ASSET,
    )

    assert recipe["environment"] == effective_training_metadata["environment"]
    assert recipe["environment_hash"] == effective_training_metadata["environment_hash"]
    assert recipe["environment"]["states"] == []
    assert recipe["environment"]["state_probs"] == []


def test_recipe_keeps_eval_asset_identity_but_removes_private_locations() -> None:
    resolved = compose_resolved_train_documents(GOAL, RECIPE, source_sha="a" * 40)
    asset = {
        "schema_version": 2,
        "game": "SuperMarioBros-Nes-v0",
        "filename": "mario.nes",
        "size_bytes": 1024,
        "sha256": "c" * 64,
        "provider_rom_identity": "d" * 40,
        "provider_rom_identity_algorithm": "sha1-provider-body-v1",
        "object_uri": "s3://private-bucket/mario.nes",
    }
    resolved.effective["train_config"]["rom_asset_manifest"] = asset
    resolved.base["train_config"]["rom_asset_manifest"] = deepcopy(asset)

    document = build_recipe_document(
        resolved.effective,
        repo_root=Path.cwd(),
        source_commit="a" * 40,
        run_description="portable evaluation asset",
        seed=7,
        runtime_image_ref=RUNTIME,
        base_materialized_recipe=resolved.base,
        canonical_goal=resolved.canonical_goal,
    )

    expected_asset = {
        "schema_version": 2,
        "game": "SuperMarioBros-Nes-v0",
        "filename": "mario.nes",
        "size_bytes": 1024,
        "sha256": "c" * 64,
        "provider_rom_identity": "d" * 40,
        "provider_rom_identity_algorithm": "sha1-provider-body-v1",
    }
    assert evaluation_contract(document)["asset"] == expected_asset
    assert document["provenance"]["asset"] == expected_asset


def test_recipe_provider_is_exact_and_never_falls_back() -> None:
    provider = "supermariobrosnes-turbo"
    resolved = compose_resolved_train_documents(
        GOAL,
        RECIPE,
        env_provider=provider,
        source_sha="a" * 40,
    )
    bind_mario_asset(resolved)
    validated = build_recipe_document(
        resolved.effective,
        repo_root=Path.cwd(),
        source_commit="a" * 40,
        run_description=f"{provider} recipe fixture",
        seed=7,
        runtime_image_ref=RUNTIME,
        base_materialized_recipe=resolved.base,
        canonical_goal=resolved.canonical_goal,
    )

    assert validated["recipe"]["train_config"]["env_provider"] == provider
    assert evaluation_contract(validated)["environment"]["env_provider"] == provider


def test_canonical_recipe_bytes_are_deterministic_and_newline_terminated() -> None:
    document = level1_1_recipe_document()
    assert canonical_json_bytes(document) == canonical_json_bytes(
        json.loads(canonical_json_bytes(document))
    )
    assert canonical_json_bytes(document).endswith(b"\n")
    assert b" " not in canonical_json_bytes({"z": 1, "a": 2})


def test_future_recipe_version_fails_with_source_and_supported_versions(tmp_path: Path) -> None:
    path = tmp_path / "recipe.json"
    path.write_text(
        json.dumps(
            {
                "document_type": "gradlab.recipe",
                "format_version": 999,
                "recipe": {},
                "provenance": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(UnsupportedPolicyDocumentVersion) as error:
        load_recipe_document(path)
    message = str(error.value)
    assert str(path) in message
    assert "999" in message
    assert "[4]" in message
    assert "Regenerate the artifact" in message


def test_divisor_era_recipe_version_is_rejected() -> None:
    document = level1_1_recipe_document()
    document["format_version"] = 3

    with pytest.raises(UnsupportedPolicyDocumentVersion, match="format_version 3"):
        validate_recipe_document(document)


def test_future_model_version_fails_before_checkpoint_access(tmp_path: Path) -> None:
    write_bundle(tmp_path)
    model_path = tmp_path / "model.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    model["format_version"] = 999
    model_path.write_text(json.dumps(model), encoding="utf-8")

    with patch("gradlab.policy_bundle.sha256_file", side_effect=AssertionError("checkpoint read")):
        with pytest.raises(UnsupportedPolicyDocumentVersion) as error:
            load_policy_bundle(tmp_path)
    assert MODEL_DOCUMENT_TYPE in str(error.value)
    assert "format_version 999" in str(error.value)
    assert "[3]" in str(error.value)


def test_model_v3_records_durable_state_archive_summary(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"checkpoint bytes")
    recipe_document = level1_1_recipe_document()
    recipe_path = write_canonical_json(tmp_path / "recipe.json", recipe_document)
    metadata = {
        "kind": "checkpoint",
        "checkpoint_step": 500_000,
        "algorithm_id": "ppo",
        "model_class": "stable_baselines3.ppo.ppo.PPO",
        "training_backend_id": "sb3.ppo",
        "training_backend_config_hash": training_backend_config_hash(
            recipe_document["recipe"]["train_config"]
        ),
        "state_archive_preflight_sha256": "c" * 64,
        "state_archive_summary": {
            "semantic_id": "state-archive-v1",
            "schema_version": 1,
            "persistence": "durable",
            "provider_id": "supermariobrosnes-turbo",
            "codec_id": "supermariobrosnes-turbo.portable-v2",
            "compatibility_id": "sha256:" + "d" * 64,
            "entry_count": 61,
            "blob_count": 17,
            "blob_bytes": 123456,
            "view_ids": ["go-explore"],
        },
    }
    model = build_model_document(checkpoint, recipe_path, metadata)
    write_canonical_json(tmp_path / "model.json", model)

    bundle = load_policy_bundle(tmp_path)

    assert bundle.model["format_version"] == 3
    assert bundle.model["provenance"]["state_archive_summary"] == metadata["state_archive_summary"]


def test_noncurrent_model_schema_is_rejected(tmp_path: Path) -> None:
    write_bundle(tmp_path)
    model_path = tmp_path / "model.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    model["format_version"] = 2
    write_canonical_json(model_path, model)

    with pytest.raises(UnsupportedPolicyDocumentVersion, match="format_version 2"):
        load_policy_bundle(tmp_path)


def test_known_recipe_schema_rejects_unknown_fields_and_urls(tmp_path: Path) -> None:
    document = level1_1_recipe_document()
    document["unknown"] = True
    write_canonical_json(tmp_path / "recipe.json", document)
    with pytest.raises(PolicyDocumentError, match="unknown field"):
        load_recipe_document(tmp_path / "recipe.json")

    document = level1_1_recipe_document()
    document["provenance"]["runtime"]["packages"] = {"bad": "https://example.invalid/policy"}
    write_canonical_json(tmp_path / "recipe.json", document)
    with pytest.raises(PolicyDocumentError, match="URL"):
        load_recipe_document(tmp_path / "recipe.json")


def test_bundle_rejects_checkpoint_and_recipe_hash_mismatches(tmp_path: Path) -> None:
    write_bundle(tmp_path)
    (tmp_path / "model.zip").write_bytes(b"changed")
    with pytest.raises(PolicyDocumentError, match="model.zip hash"):
        load_policy_bundle(tmp_path)

    write_bundle(tmp_path)
    recipe = json.loads((tmp_path / "recipe.json").read_text(encoding="utf-8"))
    recipe["recipe"]["description"] = "changed"
    write_canonical_json(tmp_path / "recipe.json", recipe)
    with pytest.raises(PolicyDocumentError, match="effective_sha256"):
        load_policy_bundle(tmp_path)


def test_all_source_kinds_normalize_to_identical_eval_and_seed_requests(
    tmp_path: Path,
) -> None:
    write_bundle(tmp_path)
    local = load_policy_bundle(tmp_path, source=str(tmp_path))
    sources = (
        local,
        replace(
            local,
            source="https://models.example/runs/gradlab-test/checkpoints/1-a/model.zip",
            revision="a" * 64,
        ),
        replace(local, source="gradlab://run/gradlab-test/checkpoint/a", revision="a" * 64),
        replace(local, source="hf://tsilva/policy", revision="d" * 40),
    )
    requests = [normalized_evaluation_request(bundle, episodes=5, n_envs=1) for bundle in sources]
    assert requests[1:] == requests[:-1]
    assert len(requests[0]["seed_assignments"]) == 5
    assert requests[0]["seed_assignments"][0] == {
        "lane": 0,
        "lane_episode_ordinal": 0,
        "seed": 10_000,
    }
