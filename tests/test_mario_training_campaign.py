from pathlib import Path
from types import SimpleNamespace

import pytest

from gradlab.action_contract import declared_action_contract
from gradlab.early_stop import MetricEarlyStopStateMachine, MetricSample
from gradlab.env_providers import provider_native_vec_kwargs
from gradlab.recipe_documents import compose_resolved_train_documents
from gradlab.training_metrics import EpisodeMetricsReducer
from gradlab.checkpoint_schedule import CheckpointSchedule


ROOT = Path(__file__).resolve().parents[1]
GOALS = ROOT / "experiments/goals/SuperMarioBros-Nes-v0"
LEVELS = tuple(
    f"Level{world}-{stage}"
    for world in range(1, 9)
    for stage in range(1, 5)
    if (world, stage) != (8, 4)
)


def campaign_recipe(level):
    directory = GOALS / level / "recipes"
    name = "ppo-train-clear-80" if level == "Level4-4" else "ppo"
    return directory / f"{name}.yaml"


@pytest.mark.parametrize("level", LEVELS)
def test_campaign_recipe_preserves_full_start_and_separate_acceptance(level):
    resolved = compose_resolved_train_documents(
        GOALS / level / "_goal.yaml",
        campaign_recipe(level),
        source_sha="a" * 40,
    )
    document = resolved.effective
    config = document["train_config"]
    assert document["goal"]["evaluation_mode"] == "evaluated"
    assert document["goal"]["eval"]["episodes"] == 100
    assert document["goal"]["eval"]["acceptance"] == [
        {"metric": "eval/success/min", "operator": ">=", "threshold": 1.0}
    ]
    for phase in ("train", "eval"):
        env = document["goal"][phase]["environment"]
        assert env["env_config"]["state"] == level
        assert env["task"]["events"]["level_change"] == {"signal": "level", "operation": "change"}
        assert env["task"]["termination"]["success"] == ["level_change"]
        assert env["task"]["termination"]["failure"] == ["life_loss"]
    assert config["env_provider"] == "env-supermariobrosnes-turbo-emu"
    assert config["state"] == level
    assert config["n_envs"] == 16
    assert config["tracking"]["backend"] == "mlflow"
    assert config["tracking"]["delivery"] == "online"
    assert config["checkpoint_eval_backend"] == "none"
    assert config["checkpoint_monitoring"]["enabled"] is False
    assert not config.get("state_archive")
    assert not config.get("initialization")
    conditions = config["early_stop"]["conditions"]
    assert conditions["clear_100"]["action"] == "stop"
    assert {key for key, value in conditions.items() if value["action"] == "stop"} == {
        "clear_100",
        "qualified_plateau",
    }
    assert conditions["clear_80"]["action"] == "observe"
    assert conditions["qualified_plateau"]["activation_threshold"] == 0.8
    assert conditions["qualified_plateau"]["outcome"] == "neutral"
    assert config["checkpoint_candidates"]["thresholds"] == [0.8, 0.9, 0.95, 1.0]
    assert conditions["clear_80"]["metric"] == "train/success/min"
    assert conditions["clear_80"]["threshold"] == 0.8
    assert conditions["clear_80"]["operator"] == ">="
    assert conditions["clear_80"]["patience_steps"] == 0
    backend = config["training_backend"]
    assert backend["id"] == "gradlab.ppo"
    assert backend["config"]["device"] == "cuda"
    uses_sil = level in {
        "Level1-3", "Level2-2", "Level2-4", "Level3-3", "Level4-2", "Level4-3", "Level4-4",
        "Level6-1", "Level6-3", "Level7-2", "Level7-3", "Level7-4",
    }
    assert bool(backend["config"].get("sil", {}).get("enabled")) == uses_sil
    assert config["timesteps"] == (16_777_216 if uses_sil or level == "Level1-1" else 50_000_000)


def test_campaign_covers_exactly_31_ordinary_levels_without_final_level():
    recipes = [campaign_recipe(level) for level in LEVELS]
    final_goal = GOALS / "Level8-4"
    assert set(GOALS.glob("Level[1-8]-[1-4]/recipes/*.yaml")) == {
        *recipes, final_goal / "recipes/ppo.yaml", final_goal / "recipes/ppo-sil.yaml",
    }
    assert {path.parent.name for path in GOALS.glob("Level[1-8]-[1-4]/_goal.yaml")} == {
        *LEVELS, "Level8-4",
    }
    assert {path.parent.parent.name for path in recipes} == set(LEVELS)
    assert len(recipes) == len(LEVELS) == 31


@pytest.mark.parametrize("name, uses_sil", [("ppo", False), ("ppo-sil", True)])
def test_final_castle_recipes_share_pipe_controls_and_qualified_victory(name, uses_sil):
    goal = GOALS / "Level8-4"
    resolved = compose_resolved_train_documents(goal / "_goal.yaml", goal / "recipes" / f"{name}.yaml")
    config = resolved.effective["train_config"]
    assert config["state"] == "Level8-4"
    assert config["checkpoint_eval_backend"] == "none"
    assert config["reward_shape"] == "sft-ppo-v1"
    assert config["task"]["reward"]["completion_reward"] > 0
    assert "progress_milestones" not in config["task"]["reward"]
    assert ["DOWN"] in declared_action_contract(config)["table"]
    assert bool(config["training_backend"]["config"].get("sil", {}).get("enabled")) == uses_sil
    for phase in ("train", "eval"):
        environment = resolved.effective["goal"][phase]["environment"]
        assert environment["env_config"]["state"] == "Level8-4"
        task = environment["task"]
        assert task["signals"]["game_mode"] == "game_mode"
        assert task["termination"]["success"] == ["game_complete"]
        assert task["events"]["game_complete"] == {
            "signal": "game_mode",
            "operation": "equals",
            "value": 2,
            "when": {"signal": "level", "value": [7, 3]},
        }


def test_final_castle_named_route_shape_adds_only_the_milestone_rewards():
    goal = GOALS / "Level8-4" / "_goal.yaml"
    recipe = goal.parent / "recipes/ppo-sil.yaml"
    baseline = compose_resolved_train_documents(goal, recipe)
    route = compose_resolved_train_documents(
        goal, recipe, recipe_overrides=("reward_shape=sft-ppo-8-4-route-v1",),
    )
    control = baseline.effective["train_config"]
    candidate = route.effective["train_config"]
    assert candidate["reward_shape"] == "sft-ppo-8-4-route-v1"
    assert candidate["reward_shape_sha256"] != control["reward_shape_sha256"]
    assert candidate["training_backend"] == control["training_backend"]
    original_task = control["task"]
    route_task = candidate["task"]
    reward = dict(route_task["reward"])
    milestones = reward.pop("progress_milestones")
    assert [item["reward"] for item in milestones] == [300, 350, 650, 550, 300, 750]
    assert {**route_task, "reward": reward} == original_task

    def info_keys(documents):
        kwargs = provider_native_vec_kwargs(
            documents.effective_training.environment, n_envs=1,
            native_obs_crop=lambda _: None, state_weight_mapping=lambda _: {},
        )
        return set(kwargs["info_filter"]["keys"])

    assert info_keys(route) - info_keys(baseline) == {"area_type"}
    assert "coins" in info_keys(route)


def episode(success, *, origin="target"):
    return SimpleNamespace(
        start_id="Level1-1",
        start_origin=origin,
        episode_return=100000.0,
        episode_length=10,
        metrics={},
        outcome="success" if success else "failure",
    )


def test_full_genuine_80_window_preserves_fallback_then_continues_to_100():
    config = compose_resolved_train_documents(
        GOALS / "Level1-1/_goal.yaml",
        campaign_recipe("Level1-1"),
    ).effective["train_config"]
    reducer = EpisodeMetricsReducer(configured_starts=("Level1-1",))
    assert reducer.window_size == 100
    machine = MetricEarlyStopStateMachine(config["early_stop"])

    def observe(records, step):
        values = reducer.consume(records)
        return values, machine.update(
            {key: MetricSample(step=step, value=value) for key, value in values.items()}
        ).stop_decision

    # Even 99/99 actual clears cannot qualify during warm-up.
    warmup = EpisodeMetricsReducer(configured_starts=("Level1-1",))
    assert "train/success/min" not in warmup.consume([episode(True) for _ in range(99)])
    # Rewards/progress and non-episode records cannot fill the denominator.
    records = [episode(False) for _ in range(21)] + [episode(True) for _ in range(78)]
    values, decision = observe(records, 1)
    assert "train/success/min" not in values and decision is None
    values, decision = observe([SimpleNamespace(reward=1e9, progress=1.0)], 2)
    assert values["train/episodes/count"] == 99 and decision is None
    values, decision = observe([episode(True)], 3)
    assert values["train/success/min"] == 0.79 and decision is None
    values, decision = observe([episode(True, origin="archive") for _ in range(100)], 4)
    assert values["train/episodes/count"] == 200
    assert values["train/success/Level1-1/count"] == 79
    assert values["train/success/min"] == 0.79 and decision is None
    # The next genuine clear evicts a failure from the rolling window.
    values, decision = observe([episode(True)], 5)
    assert values["train/success/min"] == 0.8
    assert len(reducer.success_windows["Level1-1"]) == 100
    assert sum(reducer.success_windows["Level1-1"]) == 80
    assert decision is None
    schedule = CheckpointSchedule(config)
    assert schedule.candidate_due(5, (values["train/success/min"], 5))
    values, decision = observe([episode(True) for _ in range(20)], 6)
    assert decision["condition_id"] == "clear_100"
    assert decision["metric_step"] == 6
    assert decision["value"] == 1.0


def test_a_perfect_first_window_selects_clear_100_stop():
    config = compose_resolved_train_documents(
        GOALS / "Level1-1/_goal.yaml",
        campaign_recipe("Level1-1"),
    ).effective["train_config"]
    machine = MetricEarlyStopStateMachine(config["early_stop"])
    decision = machine.update(
        {"train/success/min": MetricSample(step=100, value=1.0)}
    ).stop_decision
    assert decision["condition_id"] == "clear_100"
