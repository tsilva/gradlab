from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pytest

from gradlab.callbacks import RewardStatsAccumulator
from gradlab.metric_inventory import resolve_metric_inventory
from gradlab.metric_names import (
    METRIC_DEFINITIONS,
    metric_definition,
    metric_relationship,
    training_proxy_metric,
)
from gradlab.training_metrics import EpisodeMetricsReducer
from gradlab.wandb_leaders import rank_run_leaders, run_score


def test_reward_moments_match_finite_concatenated_samples_across_unequal_batches():
    accumulator = RewardStatsAccumulator(active_components=("event",), task={"reward": {}})
    chunks = [np.array([1.0, np.nan, -3.0]), np.array([0.0]), np.array([7.0, np.inf])]
    for values in chunks:
        accumulator.consume(
            {"shaped_reward": values, "raw_reward": values, "event_reward_component": values},
            reserve=100,
        )
    actual = accumulator.flush()
    expected = np.array([1.0, -3.0, 0.0, 7.0])
    assert actual["train/reward/mean"] == pytest.approx(expected.mean())
    assert actual["train/reward/std"] == pytest.approx(expected.std())
    assert actual["train/reward/part/event/share"] == 1.0
    assert "train/reward/task/mean" not in actual
    assert accumulator.flush() == {}


def test_active_transform_keeps_raw_statistics_even_when_a_rollout_matches():
    accumulator = RewardStatsAccumulator(task={"reward": {"reward_clip": True}})
    accumulator.consume({"shaped_reward": [1, 2, 3], "raw_reward": [1]}, reserve=3)
    accumulator.consume({"raw_reward": [2, 3]}, reserve=3)
    assert accumulator.flush()["train/reward/task/mean"] == 2.0
    accumulator.consume({"shaped_reward": [1, 2], "raw_reward": [2, 1]}, reserve=2)
    assert accumulator.flush()["train/reward/task/mean"] == 1.5


def test_single_component_and_fixed_event_deduplication_preserves_required_metrics():
    task = {"id": "identity", "reward": {"reward_mode": "events", "event_rewards": {"hit": 2.0}}}
    values = {
        "shaped_reward": [0, 2, 0, 2],
        "raw_reward": [0, 2, 0, 2],
        "event_reward_component": [0, 2, 0, 2],
        "event_reward_component/hit": [0, 2, 0, 2],
    }
    accumulator = RewardStatsAccumulator(active_components=("event",), task=task)
    accumulator.consume(values, reserve=4)
    result = accumulator.flush()
    assert result["train/reward/mean"] == 1.0
    assert result["train/reward/event/hit/fraction"] == 0.5
    assert not any("/part/" in name or "/task/" in name for name in result)
    assert "train/reward/event/hit/mean" not in result
    required = RewardStatsAccumulator(
        active_components=("event",), task=task, required_metrics=("train/reward/event/hit/mean",)
    )
    required.consume(values, reserve=4)
    assert required.flush()["train/reward/event/hit/mean"] == 1.0


def test_cached_episode_snapshot_is_invalidated_and_cannot_be_mutated_by_callers(monkeypatch):
    reducer = EpisodeMetricsReducer(track_success=False)
    calls = []
    original = reducer._build_snapshot
    monkeypatch.setattr(reducer, "_build_snapshot", lambda: (calls.append(1), original())[1])
    reducer.consume(())["train/episodes/count"] = 999
    assert reducer.consume(())["train/episodes/count"] == 0
    assert len(calls) == 1
    reducer.consume([SimpleNamespace(episode_return=2, episode_length=3, outcome="success")])
    assert reducer.snapshot()["train/episodes/count"] == 1
    assert len(calls) == 2


@pytest.mark.parametrize("required", [False, True])
def test_single_start_deduplication_keeps_maturity_and_selected_series(required):
    selected = ("train/success/mean", "train/success/Start/fraction") if required else ()
    reducer = EpisodeMetricsReducer(
        configured_starts=("Start",), track_success=True, required_metrics=selected
    )

    def episode(outcome):
        return SimpleNamespace(
            episode_return=1.0, episode_length=2, outcome=outcome, start_id="Start", metrics={}
        )

    immature = reducer.consume([episode("success") for _ in range(99)])
    assert immature["train/success/Start/count"] == 99
    assert "train/success/min" not in immature
    assert not any(name in immature for name in selected)
    mature = reducer.consume([episode("failure")])
    assert mature["train/success/min"] == 0.99
    assert ("train/success/mean" in mature) == required
    assert ("train/success/Start/fraction" in mature) == required
    for name in selected:
        assert mature[name] == 0.99
    assert reducer.local_progress()["completion"] == 0.99
    assert "completion" not in mature


def test_registry_relationships_and_axes_resolve_with_custom_progress_names():
    for definition in METRIC_DEFINITIONS:
        if definition.axis != "-":
            assert metric_definition(definition.axis) is not None
        concrete = definition.name.format(
            algorithm="ppo",
            progress="return",
            start="Start",
            reason="event",
            event="event",
            component="event",
            condition="stop",
        )
        for relationship in ("leader", "training_proxy"):
            target = metric_relationship(concrete, relationship)
            assert target is None or metric_definition(target) is not None
    assert training_proxy_metric("eval/progress/return/mean", progress_fields=frozenset()) is None
    assert (
        training_proxy_metric("eval/progress/return/mean", progress_fields=frozenset({"return"}))
        == "train/progress/return/mean"
    )


def test_registry_templates_have_no_namespace_collisions():
    for left, right in combinations(METRIC_DEFINITIONS, 2):
        a, b = left.name.split("/"), right.name.split("/")
        if len(a) == len(b):
            assert any(
                x != y and "{" not in x and "{" not in y for x, y in zip(a, b, strict=True)
            ), (left.name, right.name)


@pytest.mark.parametrize("mode,action_frequency", [("discrete", True), ("multi_discrete", False)])
def test_inventory_filters_diagnostics_using_declared_action_encoding(mode, action_frequency):
    inventory = resolve_metric_inventory(
        {
            "training_backend": {"id": "sb3.ppo"},
            "env_args": {"use_restricted_actions": mode},
            "task": {"action": {"set": "native"}, "reward": {"reward_mode": "native"}},
        }
    )
    assert "train/noise/std/mean" not in inventory.names
    assert ("train/action/fraction/max" in inventory.names) == action_frequency


def test_run_order_uses_all_goal_criteria_and_rejects_missing_tiebreakers():
    rank = [
        "max(train/progress/bricks_destroyed/mean)",
        "max(train/progress/bricks_destroyed/max)",
        "min(train/episode_steps/mean)",
    ]

    def score(name, best, length, step):
        return run_score(
            SimpleNamespace(
                id=name,
                name=name,
                url="",
                config={"goal_slug": "goal", "recipe_slug": "recipe", "selection_rank": rank},
                summary={
                    "train/progress/bricks_destroyed/mean": 100,
                    "train/progress/bricks_destroyed/max": best,
                    "train/episode_steps/mean": length,
                    "train/step": step,
                },
            ),
            objective_keys=(),
        )

    scores = [score("A", 110, 100, 1), score("B", 120, 200, 10), score("C", 120, 150, 100)]
    assert [run.run_id for run in rank_run_leaders(scores)[0].runs] == ["C", "B", "A"]
    assert score("missing", None, 100, 1) is None


def test_inventory_keeps_required_metrics_and_filters_inapplicable_families():
    inventory = resolve_metric_inventory(
        {
            "training_backend": {"id": "sb3.a2c"},
            "checkpoint_eval_backend": "none",
            "episode_progress_fields": ["return"],
            "task": {"reward": {"reward_mode": "native"}},
            "selection_rank": ["max(train/progress/return/mean)"],
        }
    )
    assert inventory.matches("train/progress/{progress}/mean")
    assert inventory.matches("train/value_loss/mean")
    assert not inventory.matches("train/kl/mean")
    assert not inventory.matches("train/success/min")
    assert not inventory.matches("train/curriculum/cells/count")
    assert not inventory.matches("eval/return/mean")


def test_minimizing_primary_ranks_individual_runs_and_cohorts_consistently():
    def score(name, recipe, length):
        return run_score(
            SimpleNamespace(
                id=name,
                name=name,
                url="",
                config={
                    "goal_slug": "goal",
                    "recipe_slug": recipe,
                    "selection_rank": ["min(train/episode_steps/mean)"],
                },
                summary={"train/episode_steps/mean": length},
            ),
            objective_keys=(),
        )

    cohorts = rank_run_leaders(
        [score("slow", "a", 20), score("fast", "a", 10), score("other", "b", 30)]
    )
    assert [cohort.recipe_slug for cohort in cohorts] == ["a", "b"]
    assert [run.run_id for run in cohorts[0].runs] == ["fast", "slow"]
    assert (cohorts[0].worst_seed, cohorts[0].best_seed) == (20, 10)


def test_dashboard_uses_goal_order_and_rejects_conflicting_shared_goals(tmp_path):
    from gradlab.wandb_workspace_declarations import WandbWorkspaceSpec, _resolve_project_metrics

    spec = WandbWorkspaceSpec(
        identity="test",
        view_id="test",
        project="test",
        profile_id="test",
        display_name="test",
        run_scope="all",
        max_runs=5,
        sections=(),
        primary_metrics="goal_rank",
    )
    ranks = ["max(train/return/max)", "max(train/return/mean)", "min(train/episode_steps/mean)"]
    goal = {"objective": {"rank": ranks}}
    result = _resolve_project_metrics(spec, [(tmp_path / "_goal.yaml", goal)])
    assert [panel.y[0] for panel in result.sections[0].panels] == [
        "train/return/max",
        "train/return/mean",
        "train/episode_steps/mean",
    ]
    with pytest.raises(ValueError, match="different goal rankings"):
        _resolve_project_metrics(
            spec,
            [
                (tmp_path / "a.yaml", goal),
                (tmp_path / "b.yaml", {"objective": {"rank": list(reversed(ranks))}}),
            ],
        )


@pytest.mark.parametrize("representation", ["flat", "nested", "scalar"])
def test_run_ranking_reads_wandb_summary_reducer_representations(representation):
    metrics = {
        "train/progress/bricks_destroyed/mean": (32.08, "last"),
        "train/progress/bricks_destroyed/max": (66.0, "last"),
        "train/episode_steps/mean": (2277.98, "last"),
        "train/step": (4915200, "max"),
    }
    summary = {
        (f"{name}.{reducer}" if representation == "flat" else name): (
            {reducer: value} if representation == "nested" else value
        )
        for name, (value, reducer) in metrics.items()
    }
    run = SimpleNamespace(
        id="run",
        name="run",
        url="",
        summary=summary,
        config={
            "goal_slug": "goal",
            "recipe_slug": "recipe",
            "selection_rank": [
                "max(train/progress/bricks_destroyed/mean)",
                "max(train/progress/bricks_destroyed/max)",
                "min(train/episode_steps/mean)",
            ],
        },
    )
    result = run_score(run, objective_keys=())
    assert result.objective == 32.08
    assert result.rank_values == (32.08, 66.0, -2277.98)
    assert result.steps == 4915200
