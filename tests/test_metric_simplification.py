from types import SimpleNamespace

import numpy as np
import pytest

from gradlab.callbacks import RewardStatsAccumulator
from gradlab.metric_names import (
    METRIC_DEFINITIONS,
    metric_definition,
    metric_relationship,
    training_proxy_metric,
)
from gradlab.metric_inventory import resolve_metric_inventory
from gradlab.training_metrics import EpisodeMetricsReducer
from gradlab.wandb_leaders import rank_run_leaders, run_score


def test_reward_moments_match_finite_concatenated_samples_across_unequal_batches():
    accumulator = RewardStatsAccumulator(active_components=("event",))
    chunks = [np.array([1.0, np.nan, -3.0]), np.array([0.0]), np.array([7.0, np.inf])]
    for values in chunks:
        accumulator.consume({"shaped_reward": values, "raw_reward": values, "event_reward_component": values}, reserve=100)
        assert not accumulator._pending_shaped
        assert not accumulator._pending_raw
    actual = accumulator.flush()
    expected = np.array([1.0, -3.0, 0.0, 7.0])
    assert actual["train/reward/shaped/mean"] == pytest.approx(expected.mean())
    assert actual["train/reward/shaped/std"] == pytest.approx(expected.std())
    assert actual["train/reward/component/event/share"] == 1.0
    assert "train/reward/pre_transform/mean" not in actual
    assert accumulator.flush() == {}


def test_reward_equality_tracks_sequence_not_just_moments_or_batch_boundaries():
    accumulator = RewardStatsAccumulator()
    accumulator.consume({"shaped_reward": [1, 2, 3], "raw_reward": [1]}, reserve=3)
    accumulator.consume({"raw_reward": [2, 3]}, reserve=3)
    assert "train/reward/pre_transform/mean" not in accumulator.flush()
    accumulator.consume({"shaped_reward": [1, 2], "raw_reward": [2, 1]}, reserve=2)
    assert accumulator.flush()["train/reward/pre_transform/mean"] == 1.5


def test_cached_episode_snapshot_is_invalidated_and_cannot_be_mutated_by_callers(monkeypatch):
    reducer = EpisodeMetricsReducer(track_success=False)
    calls = []
    original = reducer._build_snapshot
    monkeypatch.setattr(reducer, "_build_snapshot", lambda: (calls.append(1), original())[1])
    reducer.consume(())["train/all/episodes_total"] = 999
    assert reducer.consume(())["train/all/episodes_total"] == 0
    assert len(calls) == 1
    reducer.consume([SimpleNamespace(episode_return=2, episode_length=3, outcome="success")])
    assert reducer.snapshot()["train/all/episodes_total"] == 1
    assert len(calls) == 2


def test_registry_relationships_and_axes_resolve_with_custom_progress_names():
    for definition in METRIC_DEFINITIONS:
        if definition.axis != "-":
            assert metric_definition(definition.axis) is not None
        concrete = definition.name.format(algorithm="ppo", progress="return", start="Start", reason="event", event="event", component="event", condition="stop")
        for relationship in ("leader", "training_proxy"):
            target = metric_relationship(concrete, relationship)
            assert target is None or metric_definition(target) is not None
    assert training_proxy_metric("eval/progress/return/mean", progress_fields=frozenset()) is None
    assert training_proxy_metric("eval/progress/return/mean", progress_fields=frozenset({"return"})) == "train/target/progress/return/mean"


def test_run_order_uses_all_goal_criteria_and_rejects_missing_tiebreakers():
    rank = ["max(train/target/progress/bricks_destroyed/mean)", "max(train/target/progress/bricks_destroyed/max)", "min(train/all/episode_steps_mean)"]
    def score(name, best, length, step):
        return run_score(SimpleNamespace(
            id=name, name=name, url="", config={"goal_slug": "goal", "recipe_slug": "recipe", "selection_rank": rank},
            summary={"train/target/progress/bricks_destroyed/mean": 100, "train/target/progress/bricks_destroyed/max": best, "train/all/episode_steps_mean": length, "train/global_step": step},
        ), objective_keys=())
    scores = [score("A", 110, 100, 1), score("B", 120, 200, 10), score("C", 120, 150, 100)]
    assert [run.run_id for run in rank_run_leaders(scores)[0].runs] == ["C", "B", "A"]
    assert score("missing", None, 100, 1) is None


def test_inventory_keeps_required_metrics_and_filters_inapplicable_families():
    inventory = resolve_metric_inventory({
        "training_backend": {"id": "sb3.a2c"}, "checkpoint_eval_backend": "none",
        "episode_progress_fields": ["return"], "task": {"reward": {"reward_mode": "native"}},
        "selection_rank": ["max(train/target/progress/return/mean)"],
    })
    assert inventory.matches("train/target/progress/{progress}/mean")
    assert inventory.matches("train/a2c/value_loss")
    assert not inventory.matches("train/ppo/approx_kl")
    assert not inventory.matches("train/target/success/start_rate_min")
    assert not inventory.matches("train/curriculum/archive/cell/count")
    assert not inventory.matches("eval/return_mean")


def test_minimizing_primary_ranks_individual_runs_and_cohorts_consistently():
    def score(name, recipe, length):
        return run_score(SimpleNamespace(
            id=name, name=name, url="", config={"goal_slug": "goal", "recipe_slug": recipe,
                "selection_rank": ["min(train/all/episode_steps_mean)"]},
            summary={"train/all/episode_steps_mean": length},
        ), objective_keys=())
    cohorts = rank_run_leaders([score("slow", "a", 20), score("fast", "a", 10), score("other", "b", 30)])
    assert [cohort.recipe_slug for cohort in cohorts] == ["a", "b"]
    assert [run.run_id for run in cohorts[0].runs] == ["fast", "slow"]
    assert (cohorts[0].worst_seed, cohorts[0].best_seed) == (20, 10)


def test_dashboard_uses_goal_order_and_rejects_conflicting_shared_goals(tmp_path):
    from gradlab.wandb_workspace_declarations import WandbWorkspaceSpec, _resolve_project_metrics
    spec = WandbWorkspaceSpec(identity="test", view_id="test", project="test", profile_id="test",
                             display_name="test", run_scope="all", max_runs=5, sections=(), primary_metrics="goal_rank")
    ranks = ["max(train/target/return_max)", "max(train/target/return_mean)", "min(train/all/episode_steps_mean)"]
    goal = {"objective": {"rank": ranks}}
    result = _resolve_project_metrics(spec, [(tmp_path / "_goal.yaml", goal)])
    assert [panel.y[0] for panel in result.sections[0].panels] == [
        "train/target/return_max", "train/target/return_mean", "train/all/episode_steps_mean"]
    with pytest.raises(ValueError, match="different goal rankings"):
        _resolve_project_metrics(spec, [(tmp_path / "a.yaml", goal),
            (tmp_path / "b.yaml", {"objective": {"rank": list(reversed(ranks))}})])


@pytest.mark.parametrize("representation", ["flat", "nested", "scalar"])
def test_run_ranking_reads_wandb_summary_reducer_representations(representation):
    metrics = {"train/target/progress/bricks_destroyed/mean": (32.08, "last"),
               "train/target/progress/bricks_destroyed/max": (66.0, "last"),
               "train/all/episode_steps_mean": (2277.98, "last"),
               "train/global_step": (4915200, "max")}
    summary = {(f"{name}.{reducer}" if representation == "flat" else name):
               ({reducer: value} if representation == "nested" else value)
               for name, (value, reducer) in metrics.items()}
    run = SimpleNamespace(id="run", name="run", url="", summary=summary, config={
        "goal_slug": "goal", "recipe_slug": "recipe", "selection_rank": [
            "max(train/target/progress/bricks_destroyed/mean)",
            "max(train/target/progress/bricks_destroyed/max)",
            "min(train/all/episode_steps_mean)"]})
    result = run_score(run, objective_keys=())
    assert result.objective == 32.08
    assert result.rank_values == (32.08, 66.0, -2277.98)
    assert result.steps == 4915200
