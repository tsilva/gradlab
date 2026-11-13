from pathlib import Path
from types import SimpleNamespace

import pytest

from gradlab.early_stop import MetricEarlyStopStateMachine, MetricSample
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
    name = "ppo-sil-train-clear-80" if level in {"Level1-3", "Level2-2"} else "ppo-train-clear-80"
    return GOALS / level / "recipes" / f"{name}.yaml"


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
    uses_sil = level in {"Level1-3", "Level2-2"}
    assert bool(backend["config"].get("sil", {}).get("enabled")) == uses_sil
    assert config["timesteps"] == (16_777_216 if uses_sil else 50_000_000)


def test_campaign_covers_exactly_31_ordinary_levels_without_final_level():
    recipes = sorted(GOALS.glob("*/recipes/*train-clear-80.yaml"))
    assert {path.parent.parent.name for path in recipes} == set(LEVELS)
    assert len(recipes) == len(LEVELS) == 31
    assert not (GOALS / "Level8-4").exists()


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
