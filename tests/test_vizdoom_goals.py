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
NATIVE_HORIZONS = {
    "VizdoomBasic-v1": 300,
    "VizdoomBasic-Plus-v1": 300,
    "VizdoomDeadlyCorridor-v1": 2100,
    "VizdoomDeathmatch-v1": 4200,
    "VizdoomDefendCenter-v1": 2100,
    "VizdoomDefendLine-v1": 2100,
    "VizdoomDefendLine-Plus-v1": 2100,
    "VizdoomHealthGathering-v1": 2100,
    "VizdoomHealthGathering-Plus-v1": 2100,
    "VizdoomHealthGatheringSupreme-v1": 2100,
    "VizdoomMyWayHome-v1": 2100,
    "VizdoomPredictPosition-v1": 300,
    "VizdoomTakeCover-v1": 2048,
}
SUCCESSFUL_HORIZON_GOALS = {
    "VizdoomDeathmatch-v1",
    "VizdoomDefendLine-v1",
    "VizdoomDefendLine-Plus-v1",
    "VizdoomHealthGathering-v1",
    "VizdoomHealthGathering-Plus-v1",
    "VizdoomHealthGatheringSupreme-v1",
    "VizdoomTakeCover-v1",
}
EXPECTED_GOALS = {
    "VizdoomBasic-v1": {
        "timesteps": 2_000_000,
        "event": "monster_hit",
        "training_metric": "train/outcome/success/starts/all/rolling/rate/min",
        "acceptance_metric": "eval/full/outcome/success/starts/rate/min",
        "acceptance_threshold": 1.0,
        "reward_scale": 0.01,
        "max_episode_steps": None,
    },
    "VizdoomBasic-Plus-v1": {
        "timesteps": 2_000_000,
        "event": "monster_hit",
        "training_metric": "train/outcome/success/starts/all/rolling/rate/min",
        "acceptance_metric": "eval/full/outcome/success/starts/rate/min",
        "acceptance_threshold": 1.0,
        "reward_scale": 0.01,
        "max_episode_steps": None,
    },
    "VizdoomDeadlyCorridor-v1": {
        "timesteps": 25_000_000,
        "event": "goal_reached",
        "training_metric": "train/outcome/success/starts/all/rolling/rate/min",
        "training_threshold": 1.0,
        "acceptance_metric": "eval/full/episode/return/shaped/mean",
        "acceptance_threshold": 0.80,
        "reward_scale": 0.01,
        "max_episode_steps": None,
    },
    "VizdoomDeathmatch-v1": {
        "timesteps": 500_000_000,
        "n_envs": 128,
        "num_threads": 32,
        "n_steps": 32,
        "event": "monster_killed",
        "training_metric": "train/outcome/success/starts/all/rolling/rate/min",
        "training_threshold": 1.0,
        "acceptance_metric": "eval/full/progress/kills/mean",
        "acceptance_threshold": 10.0,
        "reward_scale": 1.0,
        "actions": DEATHMATCH_ACTIONS,
        "max_episode_steps": None,
    },
    "VizdoomDefendCenter-v1": {
        "timesteps": 10_000_000,
        "event": "monster_killed",
        "training_metric": "train/outcome/success/starts/all/rolling/rate/min",
        "training_threshold": 1.0,
        "acceptance_metric": "eval/full/episode/return/shaped/mean",
        "acceptance_threshold": 10.0,
        "reward_scale": 1.0,
        "max_episode_steps": None,
    },
    "VizdoomDefendLine-v1": {
        "timesteps": 10_000_000,
        "event": "monster_killed",
        "training_metric": "train/outcome/success/starts/all/rolling/rate/min",
        "training_threshold": 1.0,
        "acceptance_metric": "eval/full/episode/return/shaped/mean",
        "acceptance_threshold": 5.0,
        "reward_scale": 1.0,
        "max_episode_steps": None,
    },
    "VizdoomDefendLine-Plus-v1": {
        "timesteps": 10_000_000,
        "event": "monster_killed",
        "training_metric": "train/outcome/success/starts/all/rolling/rate/min",
        "training_threshold": 1.0,
        "acceptance_metric": "eval/full/episode/return/shaped/mean",
        "acceptance_threshold": 5.0,
        "reward_scale": 1.0,
        "max_episode_steps": None,
    },
    "VizdoomHealthGathering-v1": {
        "timesteps": 10_000_000,
        "event": "time_limit_reached",
        "training_metric": "train/outcome/success/starts/all/rolling/rate/min",
        "training_threshold": 1.0,
        "acceptance_metric": "eval/full/outcome/success/starts/rate/min",
        "acceptance_threshold": 0.95,
        "reward_scale": 0.01,
        "max_episode_steps": None,
    },
    "VizdoomHealthGathering-Plus-v1": {
        "game": "VizdoomHealthGathering-v1",
        "timesteps": 10_000_000,
        "event": "time_limit_reached",
        "training_metric": "train/outcome/success/starts/all/rolling/rate/min",
        "training_threshold": 1.0,
        "acceptance_metric": "eval/full/outcome/success/starts/rate/min",
        "acceptance_threshold": 0.95,
        "reward_scale": 0.01,
        "max_episode_steps": None,
    },
    "VizdoomHealthGatheringSupreme-v1": {
        "timesteps": 20_000_000,
        "event": "time_limit_reached",
        "training_metric": "train/outcome/success/starts/all/rolling/rate/min",
        "training_threshold": 1.0,
        "acceptance_metric": "eval/full/outcome/success/starts/rate/min",
        "acceptance_threshold": 0.95,
        "reward_scale": 0.01,
        "max_episode_steps": None,
    },
    "VizdoomMyWayHome-v1": {
        "timesteps": 10_000_000,
        "event": "goal_reached",
        "training_metric": "train/outcome/success/starts/all/rolling/rate/min",
        "training_threshold": 1.0,
        "acceptance_metric": "eval/full/outcome/success/starts/rate/min",
        "acceptance_threshold": 0.95,
        "reward_scale": 1.0,
        "max_episode_steps": None,
    },
    "VizdoomPredictPosition-v1": {
        "timesteps": 5_000_000,
        "event": "goal_reached",
        "training_metric": "train/outcome/success/starts/all/rolling/rate/min",
        "training_threshold": 1.0,
        "acceptance_metric": "eval/full/outcome/success/starts/rate/min",
        "acceptance_threshold": 0.95,
        "reward_scale": 1.0,
        "max_episode_steps": None,
    },
    "VizdoomTakeCover-v1": {
        "timesteps": 10_000_000,
        "event": "time_limit_reached",
        "training_metric": "train/outcome/success/starts/all/rolling/rate/min",
        "training_threshold": 1.0,
        "acceptance_metric": "eval/full/outcome/success/starts/rate/min",
        "acceptance_threshold": 0.95,
        "reward_scale": 0.01,
        "max_episode_steps": None,
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
    assert train_config["obs_crop"] == [0, 32, 0, 0]
    assert train_config["obs_crop_mode"] == "mask"
    assert train_config["obs_crop_fill"] == 0
    for environment in (train_environment, eval_environment):
        assert environment["preprocessing"]["obs_crop"] == [0, 32, 0, 0]
        assert environment["preprocessing"]["obs_crop_mode"] == "mask"
        assert environment["preprocessing"]["obs_crop_fill"] == 0
    expected_n_envs = expected.get("n_envs", 32)
    assert train_config["n_envs"] == expected_n_envs
    assert train_config["env_args"]["num_threads"] == expected.get("num_threads", expected_n_envs)
    assert train_config["env_args"]["doom_skill"] == 1
    assert train_environment["env_config"]["env_args"]["doom_skill"] == 1
    assert eval_environment["env_config"]["env_args"]["doom_skill"] == 1
    if "n_steps" in expected:
        assert train_config["training_backend"]["config"]["n_steps"] == expected["n_steps"]
    assert train_config["env_args"]["use_restricted_actions"] == expected.get("actions", "discrete")
    assert train_config["task"]["id"] == "identity"
    assert expected["event"] in train_config["task"]["events"]
    assert train_config["task"]["events"]["time_limit_reached"] == {
        "signal": "native_timeout",
        "operation": "equals_for",
        "value": 1,
        "steps": 1,
    }
    expected_vizdoom_config = {"episode_timeout": NATIVE_HORIZONS[goal_id]}
    expected_vizdoom_config["render_hud"] = True
    assert train_config["env_args"]["vizdoom_config"] == expected_vizdoom_config
    assert train_config["task"]["reward"]["reward_scale"] == expected["reward_scale"]
    termination = train_config["task"]["termination"]
    expected_horizon_outcome = "success" if goal_id in SUCCESSFUL_HORIZON_GOALS else "timeout"
    assert "time_limit_reached" in termination[expected_horizon_outcome]
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
        "threshold": expected.get("training_threshold", expected["acceptance_threshold"]),
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


def test_vizdoom_defend_center_uses_the_perfect_score_as_success() -> None:
    goal_path = GOALS_ROOT / "VizdoomDefendCenter-v1" / "_goal.yaml"
    recipe_path = goal_path.parent / "recipes/ppo.yaml"
    document = compose_train_document(goal_path, recipe_path)

    train_config = document["train_config"]
    assert train_config["task"]["events"]["perfect_score_reached"] == {
        "signal": "kills",
        "operation": "equals_for",
        "value": 52,
        "steps": 1,
    }
    assert train_config["task"]["termination"] == {
        "failure": ["player_died"],
        "success": ["perfect_score_reached"],
        "timeout": ["time_limit_reached"],
    }
    assert train_config["early_stop"]["conditions"]["target_reached"] == {
        "metric": "train/outcome/success/starts/all/rolling/rate/min",
        "trigger": "threshold",
        "operator": ">=",
        "progress_baseline": 0.0,
        "threshold": 1.0,
        "patience_steps": 0,
        "outcome": "success",
        "action": "stop",
    }


@pytest.mark.parametrize(
    "goal_id",
    ["VizdoomDefendLine-v1", "VizdoomDefendLine-Plus-v1"],
)
def test_vizdoom_defend_line_uses_a_native_tic_horizon(goal_id: str) -> None:
    goal_path = GOALS_ROOT / goal_id / "_goal.yaml"
    recipe_path = goal_path.parent / "recipes/ppo.yaml"
    document = compose_train_document(goal_path, recipe_path)

    train_config = document["train_config"]
    assert train_config["env_args"]["vizdoom_config"] == {
        "episode_timeout": 2100,
        "render_hud": True,
    }
    assert "max_episode_steps" not in train_config["task"]["termination"]
    assert "max_steps" not in document["goal"]["eval"]["environment"]["env_config"]


@pytest.mark.parametrize(
    ("goal_id", "expected_watchdog_steps"),
    [
        ("VizdoomBasic-v1", 150),
        ("VizdoomBasic-Plus-v1", 150),
        ("VizdoomDeadlyCorridor-v1", 1050),
        ("VizdoomDeathmatch-v1", 2100),
        ("VizdoomDefendLine-v1", 1050),
        ("VizdoomDefendLine-Plus-v1", 1050),
        ("VizdoomDefendCenter-v1", 1050),
        ("VizdoomHealthGathering-v1", 1050),
        ("VizdoomHealthGathering-Plus-v1", 1050),
        ("VizdoomHealthGatheringSupreme-v1", 1050),
        ("VizdoomMyWayHome-v1", 1050),
        ("VizdoomPredictPosition-v1", 150),
        ("VizdoomTakeCover-v1", 1024),
    ],
)
def test_sequential_vizdoom_goals_materialize_finite_checkpoint_eval_bounds(
    goal_id: str,
    expected_watchdog_steps: int,
) -> None:
    goal_path = GOALS_ROOT / goal_id / "_goal.yaml"
    recipe_path = goal_path.parent / "recipes/ppo.yaml"
    document = compose_train_document(goal_path, recipe_path)

    contract = CheckpointEvalContractCompiler.from_train_config(
        document["train_config"],
        require_asset=False,
        materialize_seed_defaults=True,
    )

    assert contract.watchdog_steps == expected_watchdog_steps
    assert "max_episode_steps" not in contract.environment["task"]["termination"]


@pytest.mark.parametrize(
    ("frame_skip", "expected_watchdog_steps"),
    [(1, 300), (2, 150), (3, 100), (4, 75), (8, 38)],
)
def test_vizdoom_watchdog_scales_with_effective_frame_skip(
    frame_skip: int,
    expected_watchdog_steps: int,
) -> None:
    goal_path = GOALS_ROOT / "VizdoomBasic-v1" / "_goal.yaml"
    recipe_path = goal_path.parent / "recipes/ppo.yaml"
    document = compose_train_document(
        goal_path,
        recipe_path,
        recipe_overrides=(f"train.environment.preprocessing.frame_skip={frame_skip}",),
    )

    contract = CheckpointEvalContractCompiler.from_train_config(
        document["train_config"],
        require_asset=False,
        materialize_seed_defaults=True,
    )

    assert document["train_config"]["frame_skip"] == frame_skip
    assert contract.environment["frame_skip"] == frame_skip
    assert contract.environment["env_args"]["vizdoom_config"] == {
        "episode_timeout": 300,
        "render_hud": True,
    }
    assert contract.watchdog_steps == expected_watchdog_steps


@pytest.mark.parametrize(
    "goal_id",
    [
        "VizdoomHealthGathering-v1",
        "VizdoomHealthGathering-Plus-v1",
        "VizdoomHealthGatheringSupreme-v1",
    ],
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
    assert train_config["task"]["events"]["time_limit_reached"] == {
        "signal": "native_timeout",
        "operation": "equals_for",
        "value": 1,
        "steps": 1,
    }
    assert train_config["task"]["termination"] == {
        "failure": ["player_died"],
        "success": ["time_limit_reached"],
    }


def test_vizdoom_health_gathering_plus_only_varies_the_surface_identity() -> None:
    regular_root = GOALS_ROOT / "VizdoomHealthGathering-v1"
    plus_root = GOALS_ROOT / "VizdoomHealthGathering-Plus-v1"
    regular = compose_train_document(
        regular_root / "_goal.yaml",
        regular_root / "recipes/ppo.yaml",
    )
    plus = compose_train_document(
        plus_root / "_goal.yaml",
        plus_root / "recipes/ppo.yaml",
    )

    assert plus["train_config"]["game"] == "VizdoomHealthGathering-v1"
    assert plus["train_config"]["env_args"]["vizdoom_config"] == {
        "episode_timeout": 2100,
        "render_hud": True,
    }
    assert plus["train_config"]["task"] == regular["train_config"]["task"]
    assert plus["train_config"]["early_stop"] == regular["train_config"]["early_stop"]
    assert plus["goal"]["objective"] == regular["goal"]["objective"]
    assert plus["goal"]["eval"]["acceptance"] == regular["goal"]["eval"]["acceptance"]
    assert (
        plus["goal"]["eval"]["environment"]["task"]
        == regular["goal"]["eval"]["environment"]["task"]
    )


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
    assert document["train_config"]["env_args"]["game_variables"] == ["health"]
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
    assert document["train_config"]["task"]["model_inputs"]["context"]["remaining_time"] == {
        "signal": "native_time_remaining",
        "update": "transition",
        "encoding": {
            "kind": "continuous",
            "scale": 1.0,
            "offset": 0.0,
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
        "native_timeout": "provider_truncated",
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
    assert train_config["task"]["termination"] == {
        "failure": ["player_died"],
        "success": ["time_limit_reached"],
        "bootstrap": ["time_limit_reached"],
    }
    assert train_config["episode_progress_fields"] == ["kills"]
    assert train_config["env_args"]["vizdoom_config"] == {
        "episode_timeout": 4200,
        "render_hud": True,
    }
    assert (
        eval_environment["env_config"]["env_args"]["use_restricted_actions"] == DEATHMATCH_ACTIONS
    )
    assert eval_environment["task"] == train_config["task"]


def test_vizdoom_basic_training_target_still_stops_when_evaluation_is_enabled() -> None:
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
    assert target["action"] == "stop"
    assert train_config["checkpoint_eval_acceptance"] == document["goal"]["eval"]["acceptance"]


def test_vizdoom_basic_ends_on_the_first_shot_for_training_and_evaluation() -> None:
    goal_path = GOALS_ROOT / "VizdoomBasic-v1" / "_goal.yaml"
    recipe_path = goal_path.parent / "recipes/ppo.yaml"
    document = compose_train_document(goal_path, recipe_path)

    for environment in (
        document["goal"]["train"]["environment"],
        document["goal"]["eval"]["environment"],
    ):
        env_args = environment["env_config"]["env_args"]
        assert env_args["game_variables"] == ["hitcount", "ammo2"]
        assert env_args["info_filter"] == {
            "mode": "all",
            "keys": ["hitcount", "ammo2"],
        }
    assert document["train_config"]["env_args"]["game_variables"] == [
        "hitcount",
        "ammo2",
    ]
    assert document["train_config"]["env_args"]["info_filter"] == {
        "mode": "all",
        "keys": ["hitcount", "ammo2"],
    }
    assert document["train_config"]["task"]["signals"] == {
        "native_timeout": "provider_truncated",
        "hits": "hitcount",
        "ammo": "ammo2",
    }
    assert document["train_config"]["task"]["events"]["monster_hit"] == {
        "signal": "hits",
        "operation": "increase",
    }
    assert document["train_config"]["task"]["events"]["shot_fired"] == {
        "signal": "ammo",
        "operation": "decrease",
    }
    assert document["train_config"]["task"]["termination"] == {
        "success": ["monster_hit"],
        "failure": ["shot_fired"],
        "outcome_precedence": ["success", "failure", "timeout"],
        "timeout": ["time_limit_reached"],
    }
    assert document["train_config"]["env_args"]["vizdoom_config"] == {
        "episode_timeout": 300,
        "render_hud": True,
    }
    assert document["train_config"]["obs_crop"] == [0, 32, 0, 0]
    assert document["train_config"]["obs_crop_mode"] == "mask"
    assert document["train_config"]["obs_crop_fill"] == 0
    eval_preprocessing = document["goal"]["eval"]["environment"]["preprocessing"]
    assert eval_preprocessing["obs_crop"] == [0, 32, 0, 0]
    assert eval_preprocessing["obs_crop_mode"] == "mask"
    assert eval_preprocessing["obs_crop_fill"] == 0


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
