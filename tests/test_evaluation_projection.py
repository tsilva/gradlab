from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from gradlab.evaluation_projection import (
    evaluation_wandb_projection,
    metrics_schema_version_from_recipe_document,
    validate_evaluation_metric_payload,
)
from gradlab.metric_names import (
    EVAL_ACCEPTANCE_EPISODE_COMPLETED_COUNT,
    EVAL_ACCEPTANCE_EPISODE_PLANNED_COUNT,
    EVAL_CHECKPOINT_STEP,
    V13_EVAL_ACCEPTANCE_EPISODE_COMPLETED_COUNT,
    V13_EVAL_ACCEPTANCE_EPISODE_PLANNED_COUNT,
    V13_EVAL_CHECKPOINT_STEP,
    V13_LEADER_CHECKPOINT_ARTIFACT_REF,
    V13_LEADER_CHECKPOINT_RETURN_SHAPED_MEAN,
    leader_checkpoint_progress_metric,
)
from gradlab.metric_store import MetricStore
from gradlab.wandb_leaders import checkpoint_leader
from gradlab.wandb_publisher import publish_pending_frames, publish_promotion_summary


def recipe_document(version: int) -> dict[str, object]:
    return {
        "recipe": {
            "train_config": {
                "metrics_schema_version": version,
            }
        }
    }


@pytest.mark.parametrize("version", [13, 14])
def test_recipe_owned_evaluation_schema_accepts_only_supported_versions(version: int) -> None:
    assert metrics_schema_version_from_recipe_document(recipe_document(version)) == version


@pytest.mark.parametrize("version", [12, 15])
def test_recipe_owned_evaluation_schema_rejects_unknown_versions(version: int) -> None:
    with pytest.raises(ValueError, match="unsupported metrics schema"):
        metrics_schema_version_from_recipe_document(recipe_document(version))


def test_v14_projection_keeps_one_bounded_eval_surface() -> None:
    projection = evaluation_wandb_projection(
        {
            "eval/full/episode/return/shaped/mean": 4.0,
            "eval/full/episode/return/shaped/std": 1.0,
            "eval/full/outcome/success/across_starts/rate/min": 0.5,
            "eval/full/outcome/success/from/Start/rate": 0.5,
            "eval/full/outcome/reason/timeout/rate": 0.5,
            "eval/full/duration/seconds": 2.0,
            "failure_count": 1,
        },
        schema_version=14,
        checkpoint_step=100,
        accepted=False,
        episodes_planned=2,
        episodes_completed=2,
        duration_seconds=2.0,
    )

    assert projection[EVAL_CHECKPOINT_STEP] == 100
    assert projection[EVAL_ACCEPTANCE_EPISODE_PLANNED_COUNT] == 2.0
    assert projection[EVAL_ACCEPTANCE_EPISODE_COMPLETED_COUNT] == 2.0
    assert "eval/full/outcome/success/from/Start/rate" not in projection
    assert "eval/full/outcome/reason/timeout/rate" not in projection
    assert "eval/full/duration/seconds" not in projection
    assert "failure_count" not in projection


def test_v13_projection_and_promotion_stay_in_the_finite_legacy_namespace() -> None:
    metrics = {
        "eval/full/episode/return/mean": 4.0,
        "eval/full/episode/return/best": 7.0,
        "eval/full/outcome/success/rate/min": 0.5,
        "eval/full/outcome/success/rate/mean": 0.75,
    }
    projection = evaluation_wandb_projection(
        metrics,
        schema_version=13,
        checkpoint_step=100,
        accepted=True,
        episodes_planned=4,
        episodes_completed=4,
        duration_seconds=2.0,
    )
    assert projection[V13_EVAL_CHECKPOINT_STEP] == 100
    assert projection[V13_EVAL_ACCEPTANCE_EPISODE_PLANNED_COUNT] == 4.0
    assert projection[V13_EVAL_ACCEPTANCE_EPISODE_COMPLETED_COUNT] == 4.0
    assert EVAL_CHECKPOINT_STEP not in projection

    run = SimpleNamespace(summary={})
    publish_promotion_summary(
        run,
        checkpoint_step=100,
        checkpoint_url="https://models.example/model.zip",
        metrics=metrics,
        updated_at="2026-07-29T00:00:00Z",
        selection_rank=[
            "max(eval/full/episode/return/mean)",
            "min(leader/checkpoint/step)",
        ],
        evaluation_source="modal:manual",
        metrics_schema_version=13,
    )
    assert run.summary[V13_LEADER_CHECKPOINT_RETURN_SHAPED_MEAN] == 4.0
    assert (
        run.summary[V13_LEADER_CHECKPOINT_ARTIFACT_REF]
        == "https://models.example/model.zip"
    )
    assert "leader/checkpoint/objective" not in run.summary
    assert "leader/checkpoint/rank_values" not in run.summary

    leader = checkpoint_leader(
        SimpleNamespace(
            config={
                "metrics_schema_version": 13,
                "selection_rank": [
                    "max(eval/full/episode/return/mean)",
                    "min(leader/checkpoint/step)",
                ],
            },
            summary=run.summary,
            id="run",
            name="run",
            url="https://wandb.example/run",
        )
    )
    assert leader is not None
    assert leader.objective == 4.0
    assert leader.rank_score == (4.0, -100.0)


def test_legacy_projection_validation_rejects_names_outside_allowlist() -> None:
    with pytest.raises(ValueError, match="does not allow evaluation metric"):
        validate_evaluation_metric_payload(
            {"eval/full/outcome/reason/timeout/rate": 1.0},
            schema_version=13,
        )


def test_v14_promotion_projects_only_finite_semantic_leader_fields() -> None:
    run = SimpleNamespace(summary={})
    publish_promotion_summary(
        run,
        checkpoint_step=200,
        checkpoint_url="https://models.example/model.zip",
        metrics={
            "eval/full/episode/return/shaped/mean": 8.0,
            "eval/full/progress/x/max": 42.0,
            "eval/full/episode/return/shaped/std": 3.0,
        },
        updated_at="2026-07-29T00:00:00Z",
        selection_rank=[
            "max(eval/full/progress/x/max)",
            "max(eval/full/episode/return/shaped/mean)",
            "min(leader/checkpoint/step)",
        ],
        evaluation_source="modal:automatic",
    )

    assert run.summary[leader_checkpoint_progress_metric("x")] == 42.0
    assert run.summary["leader/checkpoint/episode/return/shaped/mean"] == 8.0
    assert "leader/checkpoint/episode/return/shaped/std" not in run.summary
    assert "leader/checkpoint/objective" not in run.summary
    assert "leader/checkpoint/rank_values" not in run.summary
    assert "leader/checkpoint/acceptance_pass" not in run.summary


def test_promotion_rejects_a_missing_rank_input_instead_of_fabricating_zero() -> None:
    with pytest.raises(ValueError, match="missing finite rank metric"):
        publish_promotion_summary(
            SimpleNamespace(summary={}),
            checkpoint_step=200,
            checkpoint_url="https://models.example/model.zip",
            metrics={"eval/full/episode/return/shaped/mean": 8.0},
            updated_at="2026-07-29T00:00:00Z",
            selection_rank=[
                "max(eval/full/outcome/success/across_starts/rate/min)",
                "min(leader/checkpoint/step)",
            ],
            evaluation_source="modal:automatic",
        )


def test_v13_history_uses_the_v13_checkpoint_axis() -> None:
    class Run:
        def __init__(self) -> None:
            self.definitions: list[tuple[str, dict[str, object]]] = []
            self.rows: list[dict[str, object]] = []

        def define_metric(self, name: str, **kwargs: object) -> None:
            self.definitions.append((name, kwargs))

        def log(self, payload: dict[str, object], *, step: int) -> None:
            self.rows.append({**payload, "_step": step})

    with tempfile.TemporaryDirectory() as temporary:
        store = MetricStore(Path(temporary) / "metrics.sqlite")
        store.init()
        store.append_metrics(
            {"eval/full/episode/return/mean": 4.0},
            step=100,
            source="eval:manual:legacy",
            metrics_schema_version=13,
        )
        run = Run()
        assert (
            publish_pending_frames(
                store,
                run,
                limit=10,
                metrics_schema_version=13,
            )
            == 1
        )

    assert run.rows[0][V13_EVAL_CHECKPOINT_STEP] == 100
    assert EVAL_CHECKPOINT_STEP not in run.rows[0]
    assert (
        "eval/full/episode/return/mean",
        {"step_metric": V13_EVAL_CHECKPOINT_STEP},
    ) in run.definitions
