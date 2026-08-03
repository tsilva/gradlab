from copy import deepcopy
from pathlib import Path

import pytest

from gradlab.checkpoint_acceptance import CheckpointEvalContractCompiler
from gradlab.recipe_documents import compose_train_document


GOALS_ROOT = Path("experiments/goals")
DEATHMATCH_ACTIONS = [
    [],
    ["ATTACK"],
    ["MOVE_FORWARD"],
    ["MOVE_BACKWARD"],
    ["MOVE_LEFT"],
    ["MOVE_RIGHT"],
    ["TURN_LEFT"],
    ["TURN_RIGHT"],
    ["SPEED", "MOVE_FORWARD"],
    ["ATTACK", "MOVE_FORWARD"],
    ["ATTACK", "MOVE_BACKWARD"],
    ["ATTACK", "MOVE_LEFT"],
    ["ATTACK", "MOVE_RIGHT"],
    ["ATTACK", "TURN_LEFT"],
    ["ATTACK", "TURN_RIGHT"],
    ["SELECT_NEXT_WEAPON"],
    ["SELECT_PREV_WEAPON"],
]
EXPECTED_GOALS = {
    "VizdoomBasic-v1": {
        "timesteps": 2_000_000,
        "event": "goal_reached",
        "training_metric": "train/outcome/success/across_starts/window_100/rate/min",
        "acceptance_metric": "eval/full/outcome/success/across_starts/rate/min",
        "acceptance_threshold": 1.0,
        "reward_scale": 100.0,
        "max_episode_steps": None,
    },
    "VizdoomBasic-Plus-v1": {
        "timesteps": 2_000_000,
        "event": "goal_reached",
        "training_metric": "train/outcome/success/across_starts/window_100/rate/min",
        "acceptance_metric": "eval/full/outcome/success/across_starts/rate/min",
        "acceptance_threshold": 1.0,
        "reward_scale": 100.0,
        "max_episode_steps": None,
    },
    "VizdoomDeadlyCorridor-v1": {
        "timesteps": 25_000_000,
        "event": "goal_reached",
        "training_metric": "train/episode/return/shaped/from/target/window_100/mean",
        "acceptance_metric": "eval/full/episode/return/shaped/mean",
        "acceptance_threshold": 0.80,
        "reward_scale": 100.0,
        "max_episode_steps": None,
    },
    "VizdoomDeathmatch-v1": {
        "timesteps": 100_000_000,
        "n_envs": 128,
        "num_threads": 32,
        "n_steps": 32,
        "event": "monster_killed",
        "training_metric": "train/episode/return/shaped/from/target/window_100/mean",
        "acceptance_metric": "eval/full/episode/return/shaped/mean",
        "acceptance_threshold": 10.0,
        "reward_scale": 1.0,
        "actions": DEATHMATCH_ACTIONS,
        "max_episode_steps": 1050,
    },
    "VizdoomDefendCenter-v1": {
        "timesteps": 10_000_000,
        "event": "monster_killed",
        "training_metric": "train/episode/return/shaped/from/target/window_100/mean",
        "acceptance_metric": "eval/full/episode/return/shaped/mean",
        "acceptance_threshold": 10.0,
        "reward_scale": 1.0,
        "max_episode_steps": None,
    },
    "VizdoomDefendLine-v1": {
        "timesteps": 10_000_000,
        "event": "monster_killed",
        "training_metric": "train/episode/return/shaped/from/target/window_100/mean",
        "acceptance_metric": "eval/full/episode/return/shaped/mean",
        "acceptance_threshold": 5.0,
        "reward_scale": 1.0,
        "max_episode_steps": None,
    },
    "VizdoomDefendLine-Plus-v1": {
        "timesteps": 10_000_000,
        "event": "monster_killed",
        "training_metric": "train/episode/return/shaped/from/target/window_100/mean",
        "acceptance_metric": "eval/full/episode/return/shaped/mean",
        "acceptance_threshold": 5.0,
        "reward_scale": 1.0,
        "max_episode_steps": None,
    },
    "VizdoomHealthGathering-v1": {
        "timesteps": 10_000_000,
        "event": "goal_reached",
        "training_metric": "train/outcome/success/across_starts/window_100/rate/min",
        "acceptance_metric": "eval/full/outcome/success/across_starts/rate/min",
        "acceptance_threshold": 0.95,
        "reward_scale": 100.0,
        "max_episode_steps": None,
    },
    "VizdoomHealthGathering-Plus-v1": {
        "game": "VizdoomHealthGathering-v1",
        "timesteps": 10_000_000,
        "event": "goal_reached",
        "training_metric": "train/episode/return/shaped/from/target/window_100/mean",
        "acceptance_metric": "eval/full/episode/return/shaped/mean",
        "acceptance_threshold": 200.0,
        "reward_scale": 100.0,
        "max_episode_steps": 10500,
    },
    "VizdoomHealthGatheringSupreme-v1": {
        "timesteps": 20_000_000,
        "event": "goal_reached",
        "training_metric": "train/outcome/success/across_starts/window_100/rate/min",
        "acceptance_metric": "eval/full/outcome/success/across_starts/rate/min",
        "acceptance_threshold": 0.95,
        "reward_scale": 100.0,
        "max_episode_steps": None,
    },
    "VizdoomMyWayHome-v1": {
        "timesteps": 10_000_000,
        "event": "goal_reached",
        "training_metric": "train/outcome/success/across_starts/window_100/rate/min",
        "acceptance_metric": "eval/full/outcome/success/across_starts/rate/min",
        "acceptance_threshold": 0.95,
        "reward_scale": 1.0,
        "max_episode_steps": None,
    },
    "VizdoomPredictPosition-v1": {
        "timesteps": 5_000_000,
        "event": "goal_reached",
        "training_metric": "train/outcome/success/across_starts/window_100/rate/min",
        "acceptance_metric": "eval/full/outcome/success/across_starts/rate/min",
        "acceptance_threshold": 0.95,
        "reward_scale": 1.0,
        "max_episode_steps": None,
    },
    "VizdoomTakeCover-v1": {
        "timesteps": 10_000_000,
        "event": "goal_reached",
        "training_metric": "train/outcome/success/across_starts/window_100/rate/min",
        "acceptance_metric": "eval/full/outcome/success/across_starts/rate/min",
        "acceptance_threshold": 0.95,
        "reward_scale": 100.0,
        "max_episode_steps": 1024,
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
    assert "stop_on_acceptance" not in train_config
    assert train_config["checkpoint_eval_acceptance"] == acceptance
    assert train_config["env_provider"] == "vizdoom-turbo"
    assert train_config["game"] == expected.get("game", goal_id)
    assert train_config["state"] == "default"
    assert train_config["frame_skip"] == 2
    assert train_environment["preprocessing"]["frame_skip"] == 2
    assert eval_environment["preprocessing"]["frame_skip"] == 2
    expected_n_envs = expected.get("n_envs", 32)
    assert train_config["n_envs"] == expected_n_envs
    assert train_config["env_args"]["num_threads"] == expected.get(
        "num_threads", expected_n_envs
    )
    assert train_config["env_args"]["doom_skill"] == 1
    assert train_environment["env_config"]["env_args"]["doom_skill"] == 1
    assert eval_environment["env_config"]["env_args"]["doom_skill"] == 1
    if "n_steps" in expected:
        assert train_config["training_backend"]["config"]["n_steps"] == expected["n_steps"]
    assert train_config["env_args"]["use_restricted_actions"] == expected.get("actions", "discrete")
    assert train_config["task"]["id"] == "identity"
    assert expected["event"] in train_config["task"]["events"]
    assert train_config["task"]["reward"]["reward_scale"] == expected["reward_scale"]
    termination = train_config["task"]["termination"]
    if termination.get("success"):
        assert termination["success"] == ["goal_reached"]
    if expected["max_episode_steps"] is None:
        assert "max_episode_steps" not in termination
    else:
        assert termination["max_episode_steps"] == expected["max_episode_steps"]
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
        "action": "observe",
    }
    assert acceptance == [
        {
            "metric": expected["acceptance_metric"],
            "operator": ">=",
            "threshold": expected["acceptance_threshold"],
        }
    ]
    assert goal["release"] == {"huggingface": {}}


@pytest.mark.parametrize(
    "goal_id",
    ["VizdoomDefendLine-v1", "VizdoomDefendLine-Plus-v1"],
)
def test_vizdoom_defend_line_uses_a_native_tic_horizon(goal_id: str) -> None:
    goal_path = GOALS_ROOT / goal_id / "_goal.yaml"
    recipe_path = goal_path.parent / "recipes/ppo.yaml"
    document = compose_train_document(goal_path, recipe_path)

    train_config = document["train_config"]
    assert train_config["env_args"]["vizdoom_config"] == {"episode_timeout": 2100}
    assert "max_episode_steps" not in train_config["task"]["termination"]
    assert document["goal"]["eval"]["environment"]["env_config"]["max_steps"] == 2100


@pytest.mark.parametrize(
    ("goal_id", "expected_max_steps"),
    [
        ("VizdoomBasic-v1", 150),
        ("VizdoomBasic-Plus-v1", 150),
        ("VizdoomDeadlyCorridor-v1", 1050),
        ("VizdoomDeathmatch-v1", 1050),
        ("VizdoomDefendLine-v1", 2100),
        ("VizdoomDefendLine-Plus-v1", 2100),
        ("VizdoomDefendCenter-v1", 1050),
        ("VizdoomHealthGathering-v1", 1050),
        ("VizdoomHealthGathering-Plus-v1", 10500),
        ("VizdoomHealthGatheringSupreme-v1", 1050),
        ("VizdoomMyWayHome-v1", 1050),
        ("VizdoomPredictPosition-v1", 150),
        ("VizdoomTakeCover-v1", 1024),
    ],
)
def test_sequential_vizdoom_goals_materialize_finite_checkpoint_eval_bounds(
    goal_id: str,
    expected_max_steps: int,
) -> None:
    goal_path = GOALS_ROOT / goal_id / "_goal.yaml"
    recipe_path = goal_path.parent / "recipes/ppo.yaml"
    document = compose_train_document(goal_path, recipe_path)

    contract = CheckpointEvalContractCompiler.from_train_config(
        document["train_config"],
        require_asset=False,
        materialize_seed_defaults=True,
    )

    assert contract.max_steps == expected_max_steps


@pytest.mark.parametrize(
    "goal_id",
    ["VizdoomHealthGathering-v1", "VizdoomHealthGatheringSupreme-v1"],
)
def test_vizdoom_health_gathering_uses_native_provider_truncation_for_survival(
    goal_id: str,
) -> None:
    goal_path = GOALS_ROOT / goal_id / "_goal.yaml"
    recipe_path = goal_path.parent / "recipes/ppo.yaml"
    document = compose_train_document(goal_path, recipe_path)

    train_config = document["train_config"]
    assert train_config["env_args"]["info_filter"] == {
        "mode": "all",
        "keys": ["player_dead", "health"],
    }
    expected_signals = {
        "player_dead": "player_dead",
        "health": "health",
        "native_timeout": "provider_truncated",
    }
    assert train_config["task"]["signals"] == expected_signals
    assert train_config["task"]["events"]["goal_reached"] == {
        "signal": "native_timeout",
        "operation": "equals_for",
        "value": 1,
        "steps": 1,
    }
    assert train_config["task"]["termination"] == {
        "failure": ["player_died"],
        "success": ["goal_reached"],
    }


def test_vizdoom_health_gathering_plus_is_long_horizon_and_time_unaware() -> None:
    goal_path = GOALS_ROOT / "VizdoomHealthGathering-Plus-v1" / "_goal.yaml"
    recipe_path = goal_path.parent / "recipes/ppo.yaml"
    document = compose_train_document(goal_path, recipe_path)

    train_config = document["train_config"]
    goal = document["goal"]
    assert train_config["game"] == "VizdoomHealthGathering-v1"
    assert train_config["env_args"]["vizdoom_config"] == {"episode_timeout": 0}
    assert train_config["task"]["termination"]["max_episode_steps"] == 10500
    assert goal["eval"]["environment"]["env_config"]["env_args"]["vizdoom_config"] == {
        "episode_timeout": 0
    }
    assert set(train_config["task"]["model_inputs"]["context"]) == {"health"}
    assert train_config["policy_model"]["schema_version"] == 2
    assert train_config["policy_model"]["fusion"] == {
        "hidden_sizes": [256],
        "activation": "tanh",
    }
    assert "remaining_time" not in str(train_config)
    assert goal["eval"]["environment"]["env_config"]["max_steps"] == 10500
    assert goal["eval"]["environment"]["task"]["termination"]["max_episode_steps"] == 10500
    assert goal["eval"]["acceptance"] == [
        {
            "metric": "eval/full/episode/return/shaped/mean",
            "operator": ">=",
            "threshold": 200.0,
        }
    ]
    assert set(goal["eval"]["environment"]["task"]["model_inputs"]["context"]) == {"health"}


@pytest.mark.parametrize(
    "goal_id",
    [
        "VizdoomHealthGathering-v1",
        "VizdoomHealthGathering-Plus-v1",
        "VizdoomHealthGatheringSupreme-v1",
    ],
)
def test_vizdoom_health_gathering_ppo_uses_confirmed_training_defaults(
    goal_id: str,
) -> None:
    goal_path = GOALS_ROOT / goal_id / "_goal.yaml"
    recipe_path = goal_path.parent / "recipes/ppo.yaml"
    document = compose_train_document(goal_path, recipe_path)

    backend_config = document["train_config"]["training_backend"]["config"]
    assert backend_config["n_steps"] == 256
    assert backend_config["learning_rate"] == 2.5e-4
    assert document["train_config"]["env_args"]["game_variables"] == ["HEALTH"]
    assert document["train_config"]["task"]["model_inputs"]["context"]["health"] == {
        "signal": "health",
        "update": "transition",
        "encoding": {
            "kind": "continuous",
            "scale": 0.01,
            "offset": 0.0,
            "low": -1.0,
            "high": 2.0,
        },
    }
    if goal_id == "VizdoomHealthGathering-Plus-v1":
        assert set(document["train_config"]["task"]["model_inputs"]["context"]) == {"health"}
        return
    assert document["train_config"]["task"]["model_inputs"]["context"]["remaining_time"] == {
        "signal": "episode_step",
        "update": "transition",
        "encoding": {
            "kind": "continuous",
            "scale": -1.0 / 1050.0,
            "offset": 1.0,
            "low": 0.0,
            "high": 1.0,
        },
    }
    assert document["train_config"]["policy_model"]["fusion"] == {
        "hidden_sizes": [256],
        "activation": "tanh",
    }


def test_vizdoom_deathmatch_declares_complete_single_player_combat_semantics() -> None:
    goal_path = GOALS_ROOT / "VizdoomDeathmatch-v1" / "_goal.yaml"
    recipe_path = goal_path.parent / "recipes/ppo.yaml"
    document = compose_train_document(goal_path, recipe_path)

    train_config = document["train_config"]
    eval_environment = document["goal"]["eval"]["environment"]

    assert train_config["env_args"]["players"] == 1
    assert train_config["env_args"]["use_restricted_actions"] == DEATHMATCH_ACTIONS
    assert train_config["env_args"]["game_variables"] == [
        "KILLCOUNT",
        "HEALTH",
        "ARMOR",
        "SELECTED_WEAPON",
        "SELECTED_WEAPON_AMMO",
        "WEAPON1",
        "WEAPON2",
        "WEAPON3",
        "WEAPON4",
        "WEAPON5",
        "WEAPON6",
        "AMMO1",
        "AMMO2",
        "AMMO3",
        "AMMO4",
        "AMMO5",
        "AMMO6",
    ]
    assert train_config["env_args"]["info_filter"] == {
        "mode": "all",
        "keys": [
            "killcount",
            "health",
            "armor",
            "selected_weapon",
            "selected_weapon_ammo",
            "weapon1",
            "weapon2",
            "weapon3",
            "weapon4",
            "weapon5",
            "weapon6",
            "ammo1",
            "ammo2",
            "ammo3",
            "ammo4",
            "ammo5",
            "ammo6",
            "player_dead",
            "pending_reset",
        ],
    }
    assert train_config["task"]["signals"] == {
        "kills": "killcount",
        "health": "health",
        "armor": "armor",
        "selected_weapon": "selected_weapon",
        "selected_weapon_ammo": "selected_weapon_ammo",
        "weapons_owned": ["weapon1", "weapon2", "weapon3", "weapon4", "weapon5", "weapon6"],
        "weapon_ammo": ["ammo1", "ammo2", "ammo3", "ammo4", "ammo5", "ammo6"],
        "player_dead": "player_dead",
        "episode_done": "pending_reset",
    }
    assert train_config["task"]["model_inputs"]["context"]["selected_weapon"] == {
        "signal": "selected_weapon",
        "update": "transition",
        "encoding": {
            "kind": "categorical",
            "values": [1, 2, 3, 4, 5, 6],
        },
    }
    assert train_config["task"]["model_inputs"]["context"]["selected_weapon_ammo"] == {
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
    assert train_config["task"]["model_inputs"]["context"]["weapons_owned"] == {
        "signal": "weapons_owned",
        "update": "transition",
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
    assert train_config["task"]["termination"] == {
        "failure": ["player_died"],
        "timeout": ["episode_ended"],
        "max_episode_steps": 1050,
    }
    assert (
        eval_environment["env_config"]["env_args"]["use_restricted_actions"] == DEATHMATCH_ACTIONS
    )
    assert eval_environment["task"] == train_config["task"]


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
    assert "stop_on_acceptance" not in train_config
    target = train_config["early_stop"]["conditions"]["target_reached"]
    assert target["outcome"] == "success"
    assert target["action"] == "observe"
    assert train_config["checkpoint_eval_acceptance"] == document["goal"]["eval"]["acceptance"]


def test_vizdoom_defend_line_plus_differs_only_by_environment_identity() -> None:
    base_goal_path = GOALS_ROOT / "VizdoomDefendLine-v1" / "_goal.yaml"
    plus_goal_path = GOALS_ROOT / "VizdoomDefendLine-Plus-v1" / "_goal.yaml"
    base = compose_train_document(base_goal_path, base_goal_path.parent / "recipes/ppo.yaml")
    plus = compose_train_document(plus_goal_path, plus_goal_path.parent / "recipes/ppo.yaml")

    normalized_plus_goal = deepcopy(plus["goal"])
    normalized_plus_goal["goal_id"] = base["goal"]["goal_id"]
    normalized_plus_goal["tags"] = base["goal"]["tags"]
    normalized_plus_goal["train"]["environment"]["env_config"]["game"] = base["goal"]["train"][
        "environment"
    ]["env_config"]["game"]
    normalized_plus_goal["eval"]["environment"]["env_config"]["game"] = base["goal"]["eval"][
        "environment"
    ]["env_config"]["game"]

    assert normalized_plus_goal == base["goal"]
    assert plus["train_config"]["game"] == "VizdoomDefendLine-Plus-v1"
    assert plus["train_config"]["task"] == base["train_config"]["task"]
    assert plus["train_config"]["training_backend"] == base["train_config"]["training_backend"]


def test_vizdoom_basic_plus_differs_only_by_environment_identity() -> None:
    base_goal_path = GOALS_ROOT / "VizdoomBasic-v1" / "_goal.yaml"
    plus_goal_path = GOALS_ROOT / "VizdoomBasic-Plus-v1" / "_goal.yaml"
    base = compose_train_document(base_goal_path, base_goal_path.parent / "recipes/ppo.yaml")
    plus = compose_train_document(plus_goal_path, plus_goal_path.parent / "recipes/ppo.yaml")

    normalized_plus_goal = deepcopy(plus["goal"])
    normalized_plus_goal["goal_id"] = base["goal"]["goal_id"]
    normalized_plus_goal["tags"] = base["goal"]["tags"]
    normalized_plus_goal["train"]["environment"]["env_config"]["game"] = base["goal"]["train"][
        "environment"
    ]["env_config"]["game"]
    normalized_plus_goal["eval"]["environment"]["env_config"]["game"] = base["goal"]["eval"][
        "environment"
    ]["env_config"]["game"]

    assert normalized_plus_goal == base["goal"]
    assert plus["train_config"]["game"] == "VizdoomBasic-Plus-v1"
    assert plus["train_config"]["task"] == base["train_config"]["task"]
    assert plus["train_config"]["training_backend"] == base["train_config"]["training_backend"]
