from __future__ import annotations

import copy
from pathlib import Path

import pytest

from rlab.config_loader import load_mapping_document
from rlab.recipe_documents import compose_train_document, load_goal_contract
from rlab.reward_programs import validate_reward_shape_catalog


MARIO_GOAL = Path("experiments/goals/SuperMarioBros-Nes-v0/Level1-1/_goal.yaml")
MARIO_RECIPE = MARIO_GOAL.parent / "recipes/ppo.yaml"
BREAKOUT_GOAL = Path("experiments/goals/Breakout-Atari2600-v0/_goal.yaml")
BREAKOUT_RECIPE = BREAKOUT_GOAL.parent / "recipes/ppo.yaml"
VIZDOOM_GOAL = Path("experiments/goals/VizdoomBasic-v1/_goal.yaml")
VIZDOOM_RECIPE = VIZDOOM_GOAL.parent / "recipes/ppo.yaml"


def test_mario_reward_shape_defaults_and_cli_override_materialize_both_phases() -> None:
    default = compose_train_document(MARIO_GOAL, MARIO_RECIPE)
    selected = compose_train_document(
        MARIO_GOAL,
        MARIO_RECIPE,
        recipe_overrides=("reward_shape=full-v1",),
    )

    default_config = default["train_config"]
    selected_config = selected["train_config"]
    assert default_config["reward_shape"] == "speedrun-v1"
    assert default_config["reward_shape_is_default"] is True
    assert default_config["task"]["reward"]["reward_mode"] == "additive"
    assert default_config["task"]["reward"]["time_penalty"] == 0.001
    assert default_config["checkpoint_eval_environment"]["task"]["reward"]["time_penalty"] == 0.001
    assert selected_config["reward_shape"] == "full-v1"
    assert selected_config["reward_shape_is_default"] is False
    assert selected_config["task"]["reward"]["reward_mode"] == "score"
    assert selected_config["task"]["reward"]["time_penalty"] == 0.001
    assert (
        selected_config["checkpoint_eval_environment"]["task"]["reward"]["reward_mode"] == "score"
    )
    assert selected_config["checkpoint_eval_environment"]["task"]["reward"]["time_penalty"] == 0.001
    assert default_config["reward_shape_sha256"] != selected_config["reward_shape_sha256"]
    assert default_config["goal_contract_sha256"] == selected_config["goal_contract_sha256"]
    assert (
        default_config["effective_goal_contract_sha256"]
        != selected_config["effective_goal_contract_sha256"]
    )
    assert "reward_shapes" not in selected["goal"]


def test_all_mario_recipes_select_the_speedrun_default() -> None:
    goal_root = Path("experiments/goals/SuperMarioBros-Nes-v0")
    recipes = sorted(goal_root.glob("*/recipes/*.yaml"))
    assert recipes

    for recipe in recipes:
        goal = recipe.parent.parent / "_goal.yaml"
        document = compose_train_document(goal, recipe)
        config = document["train_config"]
        assert config["reward_shape"] == "speedrun-v1", recipe
        assert config["reward_shape_is_default"] is True, recipe
        assert config["task"]["reward"]["reward_mode"] == "additive", recipe
        assert config["task"]["reward"]["time_penalty"] == 0.001, recipe
        assert (
            config["checkpoint_eval_environment"]["task"]["reward"]["reward_mode"] == "additive"
        ), recipe


def test_catalog_selector_and_raw_reward_override_fail_closed() -> None:
    with pytest.raises(ValueError, match="unknown reward_shape"):
        compose_train_document(
            MARIO_GOAL,
            MARIO_RECIPE,
            recipe_overrides=("reward_shape=missing-v1",),
        )
    with pytest.raises(ValueError, match="reject raw reward overrides"):
        compose_train_document(
            MARIO_GOAL,
            MARIO_RECIPE,
            recipe_overrides=("train.environment.task.reward.time_penalty=0.5",),
        )
    with pytest.raises(ValueError, match="reject raw reward overrides"):
        compose_train_document(
            MARIO_GOAL,
            MARIO_RECIPE,
            recipe_overrides=("train.task.reward.time_penalty=0.5",),
        )


def test_policy_reward_override_is_mirrored_and_changes_the_effective_goal() -> None:
    baseline = compose_train_document(VIZDOOM_GOAL, VIZDOOM_RECIPE)
    requested = (
        "train.environment.env_config.env_args.reward_clip=true",
        "eval.environment.env_config.env_args.reward_clip=true",
    )

    document = compose_train_document(
        VIZDOOM_GOAL,
        VIZDOOM_RECIPE,
        recipe_overrides=requested,
    )
    config = document["train_config"]

    assert document["recipe_overrides"] == list(requested)
    assert document["effective_recipe_overrides"] == [
        "train.environment.task.reward.reward_clip=true",
        "eval.environment.task.reward.reward_clip=true",
    ]
    assert "reward_clip" not in config["env_args"]
    assert config["task"]["reward"]["reward_clip"] == [-1.0, 1.0]
    assert "reward_clip" not in config["checkpoint_eval_environment"]["env_args"]
    assert config["checkpoint_eval_environment"]["task"]["reward"]["reward_clip"] == [-1.0, 1.0]
    assert document["policy_environment_hash"] == document["evaluation_environment_hash"]
    assert (
        config["effective_goal_contract_sha256"]
        != baseline["train_config"]["effective_goal_contract_sha256"]
    )


def test_training_policy_override_mirrors_without_an_eval_duplicate() -> None:
    document = compose_train_document(
        VIZDOOM_GOAL,
        VIZDOOM_RECIPE,
        recipe_overrides=("train.environment.task.reward.reward_clip=true",),
    )

    config = document["train_config"]
    assert config["task"]["reward"]["reward_clip"] == [-1.0, 1.0]
    assert config["checkpoint_eval_environment"]["task"]["reward"]["reward_clip"] == [-1.0, 1.0]


def test_eval_only_or_conflicting_policy_overrides_fail_closed() -> None:
    with pytest.raises(ValueError, match="cannot define policy semantics independently"):
        compose_train_document(
            VIZDOOM_GOAL,
            VIZDOOM_RECIPE,
            recipe_overrides=("eval.environment.task.reward.reward_clip=true",),
        )
    with pytest.raises(ValueError, match="train/eval policy environment overrides disagree"):
        compose_train_document(
            VIZDOOM_GOAL,
            VIZDOOM_RECIPE,
            recipe_overrides=(
                "train.environment.task.reward.reward_clip=true",
                "eval.environment.task.reward.reward_clip=false",
            ),
        )


def test_selected_reward_definition_can_be_overridden_for_an_adhoc_run() -> None:
    baseline = compose_train_document(MARIO_GOAL, MARIO_RECIPE)
    overrides = (
        "reward_shapes.definitions.speedrun-v1.progress_reward_scale=0.25",
        "reward_shapes.definitions.speedrun-v1.progress_reward_boost_start_x=640.0",
        "reward_shapes.definitions.speedrun-v1.progress_reward_boost_scale=10.0",
        "reward_shapes.definitions.speedrun-v1.death_penalty=100.0",
        "reward_shapes.definitions.speedrun-v1.completion_reward=250.0",
    )
    document = compose_train_document(
        MARIO_GOAL,
        MARIO_RECIPE,
        recipe_overrides=overrides,
    )
    config = document["train_config"]
    expected = {
        "progress_reward_scale": 0.25,
        "progress_reward_boost_start_x": 640.0,
        "progress_reward_boost_scale": 10.0,
        "death_penalty": 100.0,
        "completion_reward": 250.0,
    }

    assert config["reward_shape"] == "speedrun-v1"
    assert config["reward_shape_is_default"] is False
    assert document["recipe_overrides"] == list(overrides)
    for key, value in expected.items():
        assert config["task"]["reward"][key] == value
        assert config["checkpoint_eval_environment"]["task"]["reward"][key] == value
    assert config["goal_contract_sha256"] == baseline["train_config"]["goal_contract_sha256"]
    assert (
        config["effective_goal_contract_sha256"]
        != baseline["train_config"]["effective_goal_contract_sha256"]
    )
    assert config["reward_shape_sha256"] != baseline["train_config"]["reward_shape_sha256"]


def test_reward_definition_override_must_target_selected_shape() -> None:
    with pytest.raises(ValueError, match="unselected shape"):
        compose_train_document(
            MARIO_GOAL,
            MARIO_RECIPE,
            recipe_overrides=("reward_shapes.definitions.full-v1.death_penalty=100.0",),
        )


def test_non_catalog_goal_remains_compatible() -> None:
    document = compose_train_document(BREAKOUT_GOAL, BREAKOUT_RECIPE)
    assert "reward_shape" not in document["train_config"]
    with pytest.raises(ValueError, match="does not define reward_shapes"):
        compose_train_document(
            BREAKOUT_GOAL,
            BREAKOUT_RECIPE,
            recipe_overrides=("reward_shape=score-v1",),
        )


def test_catalog_definitions_are_complete_strict_and_semantically_unique() -> None:
    goal = load_goal_contract(MARIO_GOAL)
    malformed = copy.deepcopy(goal)
    malformed["reward_shapes"]["definitions"]["full-v1"]["reward_clip"] = [1.0]
    with pytest.raises(ValueError, match="reward_clip must contain exactly"):
        validate_reward_shape_catalog(malformed)

    incomplete = copy.deepcopy(goal)
    del incomplete["reward_shapes"]["definitions"]["full-v1"]["death_penalty"]
    with pytest.raises(ValueError, match="missing required field.*death_penalty"):
        validate_reward_shape_catalog(incomplete)

    duplicate = copy.deepcopy(goal)
    duplicate["reward_shapes"]["definitions"]["alias-v1"] = copy.deepcopy(
        duplicate["reward_shapes"]["definitions"]["full-v1"]
    )
    with pytest.raises(ValueError, match="identical executable semantics"):
        validate_reward_shape_catalog(duplicate)

    mismatched_termination = copy.deepcopy(goal)
    mismatched_termination["eval"]["environment"]["task"]["termination"]["failure"] = []
    with pytest.raises(ValueError, match="including termination"):
        validate_reward_shape_catalog(mismatched_termination)

    missing_stall = copy.deepcopy(goal)
    for phase in ("train", "eval"):
        del missing_stall[phase]["environment"]["task"]["events"]["stalled"]
    with pytest.raises(ValueError, match="must declare stalled"):
        validate_reward_shape_catalog(missing_stall)


def test_yaml_loader_rejects_duplicate_mapping_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text("reward_shape: score-v1\nreward_shape: other-v1\n", encoding="utf-8")
    with pytest.raises(Exception, match="duplicate key 'reward_shape'"):
        load_mapping_document(path)
