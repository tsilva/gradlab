from pathlib import Path

from gradlab.recipe_documents import compose_train_document


REPO_ROOT = Path(__file__).resolve().parents[1]
GOAL_PATH = REPO_ROOT / "experiments/goals/VizdoomDeathmatch-v1/_goal.yaml"
RECIPE_PATH = GOAL_PATH.parent / "recipes/ppo.yaml"


def test_deathmatch_shares_survivability_and_complete_weapon_state_between_heads() -> None:
    document = compose_train_document(GOAL_PATH, RECIPE_PATH)
    train_config = document["train_config"]

    assert train_config["task"]["model_inputs"]["context"]["health"] == {
        "signal": "health",
        "update": "transition",
        "history": "provider_frame_stack",
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
        "history": "provider_frame_stack",
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
        "history": "provider_frame_stack",
        "encoding": {
            "kind": "categorical",
            "values": [1, 2, 3, 4, 5, 6],
        },
    }
    assert train_config["task"]["model_inputs"]["context"]["selected_weapon_ammo"] == {
        "signal": "selected_weapon_ammo",
        "update": "transition",
        "history": "provider_frame_stack",
        "encoding": {
            "kind": "continuous",
            "scale": 1.0 / 300.0,
            "offset": 0.0,
            "low": 0.0,
            "high": 1.0,
            "clip": True,
        },
    }
    assert train_config["task"]["model_inputs"]["context"]["weapons_owned"] == {
        "signal": "weapons_owned",
        "update": "transition",
        "history": "provider_frame_stack",
        "encoding": {
            "kind": "continuous",
            "scale": 1.0,
            "offset": 0.0,
            "low": 0.0,
            "high": 1.0,
            "clip": True,
        },
    }
    assert train_config["task"]["model_inputs"]["context"]["weapon_ammo"] == {
        "signal": "weapon_ammo",
        "update": "transition",
        "history": "provider_frame_stack",
        "encoding": {
            "kind": "continuous",
            "scale": [1.0, 1.0 / 200.0, 1.0 / 50.0, 1.0 / 200.0, 1.0 / 50.0, 1.0 / 300.0],
            "offset": 0.0,
            "low": 0.0,
            "high": 1.0,
            "clip": True,
        },
    }
    assert train_config["policy_model"] == {
        "schema_version": 2,
        "encoder": {"kind": "nature_cnn", "features_dim": 512},
        "fusion": {"hidden_sizes": [256], "activation": "tanh"},
        "normalize_images": True,
        "orthogonal_init": True,
    }
