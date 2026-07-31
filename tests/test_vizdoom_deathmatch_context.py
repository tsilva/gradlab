from pathlib import Path

from gradlab.recipe_documents import compose_train_document


REPO_ROOT = Path(__file__).resolve().parents[1]
GOAL_PATH = REPO_ROOT / "experiments/goals/VizdoomDeathmatch-v1/_goal.yaml"
RECIPE_PATH = GOAL_PATH.parent / "recipes/ppo.yaml"


def test_deathmatch_routes_normalized_health_and_armor_only_to_nonlinear_critic() -> None:
    document = compose_train_document(GOAL_PATH, RECIPE_PATH)
    train_config = document["train_config"]

    assert train_config["task"]["model_inputs"]["context"]["health"] == {
        "signal": "health",
        "update": "transition",
        "encoding": {
            "kind": "continuous",
            "scale": 0.01,
            "offset": 0.0,
            "low": 0.0,
            "high": 2.0,
            "clip": True,
        },
    }
    assert train_config["task"]["model_inputs"]["context"]["armor"] == {
        "signal": "armor",
        "update": "transition",
        "encoding": {
            "kind": "continuous",
            "scale": 0.005,
            "offset": 0.0,
            "low": 0.0,
            "high": 1.0,
            "clip": True,
        },
    }
    assert train_config["policy_model"]["routes"] == {
        "armor": ["state_value"],
        "health": ["state_value"],
    }
    assert train_config["policy_model"]["heads"] == {
        "action": {"hidden_sizes": [], "activation": "tanh"},
        "state_value": {"hidden_sizes": [256], "activation": "tanh"},
    }
