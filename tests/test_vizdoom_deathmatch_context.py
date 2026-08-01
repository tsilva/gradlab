from pathlib import Path

from gradlab.recipe_documents import compose_train_document


REPO_ROOT = Path(__file__).resolve().parents[1]
GOAL_PATH = REPO_ROOT / "experiments/goals/VizdoomDeathmatch-v1/_goal.yaml"
RECIPE_PATH = GOAL_PATH.parent / "recipes/ppo.yaml"


def test_deathmatch_routes_survivability_to_critic_and_weapon_state_to_both_heads() -> None:
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
    assert train_config["task"]["model_inputs"]["context"]["selected_weapon"] == {
        "signal": "selected_weapon",
        "update": "transition",
        "encoding": {
            "kind": "categorical",
            "values": [1, 2, 3, 4, 5, 6],
        },
    }
    assert train_config["task"]["model_inputs"]["context"][
        "selected_weapon_ammo"
    ] == {
        "signal": "selected_weapon_ammo",
        "update": "transition",
        "encoding": {
            "kind": "continuous",
            "scale": 1.0 / 300.0,
            "offset": 0.0,
            "low": 0.0,
            "high": 1.0,
            "clip": True,
        },
    }
    assert train_config["policy_model"]["routes"] == {
        "armor": ["state_value"],
        "health": ["state_value"],
        "selected_weapon": ["action", "state_value"],
        "selected_weapon_ammo": ["action", "state_value"],
    }
    assert train_config["policy_model"]["heads"] == {
        "action": {"hidden_sizes": [], "activation": "tanh"},
        "state_value": {"hidden_sizes": [256], "activation": "tanh"},
    }
