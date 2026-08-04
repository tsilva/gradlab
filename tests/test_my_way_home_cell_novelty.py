from pathlib import Path

from gradlab.recipe_documents import compose_train_document
from gradlab.training.sb3_on_policy import active_reward_components


GOAL_ROOT = Path("experiments/goals/VizdoomMyWayHome-v1")


def test_cell_novelty_recipe_shapes_training_without_changing_evaluation() -> None:
    document = compose_train_document(
        GOAL_ROOT / "_goal.yaml",
        GOAL_ROOT / "recipes/ppo-cell-novelty.yaml",
    )
    train = document["train_config"]
    evaluation = train["checkpoint_eval_environment"]

    assert train["checkpoint_eval_backend"] == "none"
    assert train["timesteps"] == 10_000_000
    assert document["seeds"] == [123]
    assert train["env_args"]["game_variables"] == [
        "ARMOR",
        "POSITION_X",
        "POSITION_Y",
    ]
    assert train["env_args"]["info_filter"] == {
        "mode": "all",
        "keys": ["armor", "position_x", "position_y"],
    }
    assert train["task"]["signals"] == {
        "armor": "armor",
        "native_timeout": "provider_truncated",
        "position_x": "position_x",
        "position_y": "position_y",
    }
    assert train["task"]["reward"] == {
        "reward_mode": "native",
        "reward_scale": 1.0,
        "reward_clip": False,
        "cell_novelty": {
            "cell": {
                "dimensions": [
                    {"signal": "position_x", "bucket_size": 64.0},
                    {"signal": "position_y", "bucket_size": 64.0},
                ]
            },
            "first_visit_bonus": 0.005,
            "episode_bonus_cap": 0.2,
        },
    }
    assert train["early_stop"]["conditions"]["return_plateau"]["action"] == "observe"
    assert active_reward_components(train["task"]) == ("native", "cell_novelty")
    assert "model_inputs" not in train["task"]

    assert evaluation["env_args"]["game_variables"] == ["ARMOR"]
    assert evaluation["env_args"]["info_filter"] == {
        "mode": "all",
        "keys": ["armor"],
    }
    assert evaluation["task"]["signals"] == {
        "armor": "armor",
        "native_timeout": "provider_truncated",
    }
    assert evaluation["task"]["reward"] == {
        "reward_mode": "native",
        "reward_scale": 1.0,
        "reward_clip": False,
    }
    assert document["policy_environment_hash"] != document["evaluation_environment_hash"]


def test_existing_my_way_home_ppo_recipe_remains_unshaped() -> None:
    document = compose_train_document(
        GOAL_ROOT / "_goal.yaml",
        GOAL_ROOT / "recipes/ppo.yaml",
    )
    train = document["train_config"]

    assert train["env_args"]["game_variables"] == ["ARMOR"]
    assert "cell_novelty" not in train["task"]["reward"]
    assert document["policy_environment_hash"] == document["evaluation_environment_hash"]
