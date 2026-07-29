from copy import deepcopy
from pathlib import Path

import pytest

from gradlab.recipe_documents import compose_train_document


GOALS_ROOT = Path("experiments/goals")
EXPECTED_GOALS = {
    "VizdoomBasic-v1": {
        "timesteps": 2_000_000,
        "event": "monster_killed",
        "training_metric": "train/outcome/success/across_starts/window_100/rate/min",
        "acceptance_metric": "eval/full/outcome/success/across_starts/rate/min",
        "acceptance_threshold": 1.0,
        "reward_scale": 100.0,
    },
    "VizdoomDeadlyCorridor-v1": {
        "timesteps": 25_000_000,
        "event": "vest_reached",
        "training_metric": "train/episode/return/shaped/from/target/window_100/mean",
        "acceptance_metric": "eval/full/episode/return/shaped/mean",
        "acceptance_threshold": 0.80,
        "reward_scale": 100.0,
    },
    "VizdoomDefendCenter-v1": {
        "timesteps": 10_000_000,
        "event": "monster_killed",
        "training_metric": "train/episode/return/shaped/from/target/window_100/mean",
        "acceptance_metric": "eval/full/episode/return/shaped/mean",
        "acceptance_threshold": 10.0,
        "reward_scale": 1.0,
    },
    "VizdoomDefendLine-v1": {
        "timesteps": 10_000_000,
        "event": "monster_killed",
        "training_metric": "train/episode/return/shaped/from/target/window_100/mean",
        "acceptance_metric": "eval/full/episode/return/shaped/mean",
        "acceptance_threshold": 5.0,
        "reward_scale": 1.0,
    },
    "VizdoomDefendLine-Plus-v1": {
        "timesteps": 10_000_000,
        "event": "monster_killed",
        "training_metric": "train/episode/return/shaped/from/target/window_100/mean",
        "acceptance_metric": "eval/full/episode/return/shaped/mean",
        "acceptance_threshold": 5.0,
        "reward_scale": 1.0,
    },
    "VizdoomHealthGathering-v1": {
        "timesteps": 10_000_000,
        "event": "survived",
        "training_metric": "train/outcome/success/across_starts/window_100/rate/min",
        "acceptance_metric": "eval/full/outcome/success/across_starts/rate/min",
        "acceptance_threshold": 0.95,
        "reward_scale": 100.0,
    },
    "VizdoomHealthGatheringSupreme-v1": {
        "timesteps": 20_000_000,
        "event": "survived",
        "training_metric": "train/outcome/success/across_starts/window_100/rate/min",
        "acceptance_metric": "eval/full/outcome/success/across_starts/rate/min",
        "acceptance_threshold": 0.95,
        "reward_scale": 100.0,
    },
    "VizdoomMyWayHome-v1": {
        "timesteps": 10_000_000,
        "event": "vest_reached",
        "training_metric": "train/outcome/success/across_starts/window_100/rate/min",
        "acceptance_metric": "eval/full/outcome/success/across_starts/rate/min",
        "acceptance_threshold": 0.95,
        "reward_scale": 1.0,
    },
    "VizdoomPredictPosition-v1": {
        "timesteps": 5_000_000,
        "event": "monster_killed",
        "training_metric": "train/outcome/success/across_starts/window_100/rate/min",
        "acceptance_metric": "eval/full/outcome/success/across_starts/rate/min",
        "acceptance_threshold": 0.95,
        "reward_scale": 1.0,
    },
    "VizdoomTakeCover-v1": {
        "timesteps": 10_000_000,
        "event": "survived",
        "training_metric": "train/outcome/success/across_starts/window_100/rate/min",
        "acceptance_metric": "eval/full/outcome/success/across_starts/rate/min",
        "acceptance_threshold": 0.95,
        "reward_scale": 100.0,
    },
}


@pytest.mark.parametrize(("goal_id", "expected"), EXPECTED_GOALS.items())
def test_vizdoom_goal_has_complete_evaluated_ppo_contract(
    goal_id: str,
    expected: dict[str, object],
) -> None:
    goal_path = GOALS_ROOT / goal_id / "_goal.yaml"
    recipe_path = goal_path.parent / "recipes/ppo.yaml"
    document = compose_train_document(goal_path, recipe_path)

    train_config = document["train_config"]
    goal = document["goal"]
    train_environment = goal["train"]["environment"]
    eval_environment = goal["eval"]["environment"]
    acceptance = goal["eval"]["acceptance"]

    assert goal["goal_id"] == goal_id
    assert document["recipe_id"] == "ppo"
    assert train_config["timesteps"] == expected["timesteps"]
    assert train_config["checkpoint_eval_backend"] == "none"
    assert train_config["stop_on_acceptance"] is False
    assert train_config["checkpoint_eval_acceptance"] == acceptance
    assert train_config["env_provider"] == "vizdoom-turbo"
    assert train_config["game"] == goal_id
    assert train_config["state"] == "default"
    assert train_config["n_envs"] == 32
    assert train_config["env_args"]["num_threads"] == 32
    assert train_config["env_args"]["use_restricted_actions"] == "discrete"
    assert train_config["task"]["id"] == "identity"
    assert expected["event"] in train_config["task"]["events"]
    assert train_config["task"]["reward"]["reward_scale"] == expected["reward_scale"]
    assert train_config["task"] == train_environment["task"]
    assert train_environment["task"] == eval_environment["task"]
    assert eval_environment["env_config"]["n_envs"] == 16
    assert eval_environment["env_config"]["env_args"]["num_threads"] == 16
    assert goal["eval"]["episodes"] == 100
    assert goal["eval"]["policy"] == {"stochastic": True}
    conditions = train_config["early_stop"]["conditions"]
    assert set(conditions) == {"return_plateau", "target_reached"}
    assert conditions["target_reached"] == {
        "metric": expected["training_metric"],
        "trigger": "threshold",
        "operator": ">=",
        "progress_baseline": 0.0,
        "threshold": expected["acceptance_threshold"],
        "patience_steps": 0,
        "outcome": "success",
        "action": "stop",
    }
    assert acceptance == [
        {
            "metric": expected["acceptance_metric"],
            "operator": ">=",
            "threshold": expected["acceptance_threshold"],
        }
    ]
    assert goal["release"] == {"huggingface": {}}


def test_vizdoom_goal_training_target_remains_valid_when_evaluation_is_enabled() -> None:
    goal_path = GOALS_ROOT / "VizdoomBasic-v1" / "_goal.yaml"
    recipe_path = goal_path.parent / "recipes/ppo.yaml"
    document = compose_train_document(
        goal_path,
        recipe_path,
        recipe_overrides=("train.checkpoint_eval_backend=modal",),
    )

    train_config = document["train_config"]
    assert train_config["checkpoint_eval_backend"] == "modal"
    assert train_config["stop_on_acceptance"] is True
    assert train_config["early_stop"]["conditions"]["target_reached"]["outcome"] == "success"
    assert train_config["checkpoint_eval_acceptance"] == document["goal"]["eval"]["acceptance"]


def test_vizdoom_defend_line_plus_differs_only_by_environment_identity() -> None:
    base_goal_path = GOALS_ROOT / "VizdoomDefendLine-v1" / "_goal.yaml"
    plus_goal_path = GOALS_ROOT / "VizdoomDefendLine-Plus-v1" / "_goal.yaml"
    base = compose_train_document(base_goal_path, base_goal_path.parent / "recipes/ppo.yaml")
    plus = compose_train_document(plus_goal_path, plus_goal_path.parent / "recipes/ppo.yaml")

    normalized_plus_goal = deepcopy(plus["goal"])
    normalized_plus_goal["goal_id"] = base["goal"]["goal_id"]
    normalized_plus_goal["tags"] = base["goal"]["tags"]
    normalized_plus_goal["train"]["environment"]["env_config"]["game"] = (
        base["goal"]["train"]["environment"]["env_config"]["game"]
    )
    normalized_plus_goal["eval"]["environment"]["env_config"]["game"] = (
        base["goal"]["eval"]["environment"]["env_config"]["game"]
    )

    assert normalized_plus_goal == base["goal"]
    assert plus["train_config"]["game"] == "VizdoomDefendLine-Plus-v1"
    assert plus["train_config"]["task"] == base["train_config"]["task"]
    assert plus["train_config"]["training_backend"] == base["train_config"]["training_backend"]
