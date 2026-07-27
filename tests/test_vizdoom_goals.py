from pathlib import Path

import pytest

from rlab.recipe_documents import compose_train_document


GOALS_ROOT = Path("experiments/goals")
EXPECTED_GOALS = {
    "VizdoomDeadlyCorridor-v1": {
        "timesteps": 25_000_000,
        "event": "vest_reached",
        "acceptance_metric": "eval/full/outcome/success/rate/min",
        "acceptance_threshold": 0.80,
        "reward_scale": 100.0,
    },
    "VizdoomDefendCenter-v1": {
        "timesteps": 10_000_000,
        "event": "monster_killed",
        "acceptance_metric": "eval/full/episode/return/mean",
        "acceptance_threshold": 10.0,
        "reward_scale": 1.0,
    },
    "VizdoomDefendLine-v1": {
        "timesteps": 10_000_000,
        "event": "monster_killed",
        "acceptance_metric": "eval/full/episode/return/mean",
        "acceptance_threshold": 5.0,
        "reward_scale": 1.0,
    },
    "VizdoomHealthGathering-v1": {
        "timesteps": 10_000_000,
        "event": "survived",
        "acceptance_metric": "eval/full/outcome/success/rate/min",
        "acceptance_threshold": 0.95,
        "reward_scale": 100.0,
    },
    "VizdoomHealthGatheringSupreme-v1": {
        "timesteps": 20_000_000,
        "event": "survived",
        "acceptance_metric": "eval/full/outcome/success/rate/min",
        "acceptance_threshold": 0.95,
        "reward_scale": 100.0,
    },
    "VizdoomMyWayHome-v1": {
        "timesteps": 10_000_000,
        "event": "vest_reached",
        "acceptance_metric": "eval/full/outcome/success/rate/min",
        "acceptance_threshold": 0.95,
        "reward_scale": 1.0,
    },
    "VizdoomPredictPosition-v1": {
        "timesteps": 5_000_000,
        "event": "monster_killed",
        "acceptance_metric": "eval/full/outcome/success/rate/min",
        "acceptance_threshold": 0.95,
        "reward_scale": 1.0,
    },
    "VizdoomTakeCover-v1": {
        "timesteps": 10_000_000,
        "event": "survived",
        "acceptance_metric": "eval/full/outcome/success/rate/min",
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
    assert acceptance == [
        {
            "metric": expected["acceptance_metric"],
            "operator": ">=",
            "threshold": expected["acceptance_threshold"],
        }
    ]
    assert goal["release"] == {"huggingface": {}}
