from __future__ import annotations

from types import SimpleNamespace

import pytest

from gradlab.evaluation_projection import (
    evaluation_wandb_projection,
    metrics_schema_version_from_recipe_document,
)
from gradlab.metric_names import (
    EVAL_ACCEPTANCE_EPISODE_COMPLETED_COUNT,
    EVAL_ACCEPTANCE_EPISODE_PLANNED_COUNT,
    EVAL_CHECKPOINT_STEP,
    LEADER_CHECKPOINT_ARTIFACT_REF,
    LEADER_CHECKPOINT_EVALUATION_SOURCE,
    LEADER_CHECKPOINT_PROJECTION_TIMESTAMP,
    LEADER_CHECKPOINT_RETURN_SHAPED_MAX,
    LEADER_CHECKPOINT_STEP,
    METRICS_SCHEMA_VERSION,
    leader_checkpoint_progress_metric,
)
from gradlab.wandb_publisher import (
    promotion_summary_matches,
    publish_promotion_summary,
)


def recipe_document(version: int) -> dict[str, object]:
    return {
        "recipe": {
            "train_config": {
                "metrics_schema_version": version,
            }
        }
    }


def test_recipe_owned_evaluation_schema_accepts_current_version() -> None:
    assert (
        metrics_schema_version_from_recipe_document(
            recipe_document(METRICS_SCHEMA_VERSION)
        )
        == METRICS_SCHEMA_VERSION
    )


@pytest.mark.parametrize("version", [16, 17, 18, 19, 21])
def test_recipe_owned_evaluation_schema_rejects_unknown_versions(version: int) -> None:
    with pytest.raises(ValueError, match="unsupported metrics schema"):
        metrics_schema_version_from_recipe_document(recipe_document(version))


def test_v20_projection_keeps_one_bounded_eval_surface() -> None:
    projection = evaluation_wandb_projection(
        {
            "eval/full/episode/return/shaped/mean": 4.0,
            "eval/full/episode/return/shaped/std": 1.0,
            "eval/full/outcome/success/starts/rate/min": 0.5,
            "eval/full/outcome/success/from/Start/rate": 0.5,
            "eval/full/outcome/reason/timeout/rate": 0.5,
            "eval/full/duration/seconds": 2.0,
            "failure_count": 1,
        },
        schema_version=METRICS_SCHEMA_VERSION,
        checkpoint_step=100,
        accepted=False,
        episodes_planned=2,
        episodes_completed=2,
    )

    assert projection[EVAL_CHECKPOINT_STEP] == 100
    assert projection[EVAL_ACCEPTANCE_EPISODE_PLANNED_COUNT] == 2.0
    assert projection[EVAL_ACCEPTANCE_EPISODE_COMPLETED_COUNT] == 2.0
    assert "eval/full/outcome/success/from/Start/rate" not in projection
    assert "eval/full/outcome/reason/timeout/rate" not in projection
    assert "eval/full/duration/seconds" not in projection
    assert "failure_count" not in projection


def test_v20_promotion_projects_only_configured_finite_leader_fields() -> None:
    run = SimpleNamespace(summary={})
    publish_promotion_summary(
        run,
        checkpoint_step=200,
        checkpoint_url="https://models.example/model.zip",
        metrics={
            "eval/full/episode/return/shaped/mean": 8.0,
            "eval/full/episode/return/shaped/max": 10.0,
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
    assert LEADER_CHECKPOINT_RETURN_SHAPED_MAX not in run.summary
    assert "leader/checkpoint/episode/return/shaped/std" not in run.summary
    assert "leader/checkpoint/objective" not in run.summary
    assert "leader/checkpoint/rank_values" not in run.summary
    assert "leader/checkpoint/acceptance_pass" not in run.summary


def test_v20_promotion_projects_progress_mean_and_max_for_deathmatch_rank() -> None:
    run = SimpleNamespace(summary={})
    publish_promotion_summary(
        run,
        checkpoint_step=300,
        checkpoint_url="https://models.example/deathmatch.zip",
        metrics={
            "eval/full/progress/kills/mean": 12.5,
            "eval/full/progress/kills/max": 20.0,
        },
        updated_at="2026-08-06T00:00:00Z",
        selection_rank=[
            "max(eval/full/progress/kills/mean)",
            "max(eval/full/progress/kills/max)",
            "min(leader/checkpoint/step)",
        ],
        evaluation_source="modal:automatic",
    )

    assert run.summary[leader_checkpoint_progress_metric("kills", "mean")] == 12.5
    assert run.summary[leader_checkpoint_progress_metric("kills", "max")] == 20.0


def test_promotion_rejects_a_missing_rank_input_instead_of_fabricating_zero() -> None:
    with pytest.raises(ValueError, match="missing finite rank metric"):
        publish_promotion_summary(
            SimpleNamespace(summary={}),
            checkpoint_step=200,
            checkpoint_url="https://models.example/model.zip",
            metrics={"eval/full/episode/return/shaped/mean": 8.0},
            updated_at="2026-07-29T00:00:00Z",
            selection_rank=[
                "max(eval/full/outcome/success/starts/rate/min)",
                "min(leader/checkpoint/step)",
            ],
            evaluation_source="modal:automatic",
        )


def test_remote_promotion_fence_requires_complete_receipt_projection() -> None:
    run = SimpleNamespace(summary={})
    selection_rank = [
        "max(eval/full/progress/x/max)",
        "max(eval/full/episode/return/shaped/mean)",
        "min(leader/checkpoint/step)",
    ]
    publish_promotion_summary(
        run,
        checkpoint_step=200,
        checkpoint_url="https://models.example/model.zip",
        metrics={
            "eval/full/progress/x/max": 42.0,
            "eval/full/episode/return/shaped/mean": 8.0,
        },
        updated_at="2026-08-07T00:00:00Z",
        selection_rank=selection_rank,
        evaluation_source="modal:automatic",
    )

    assert promotion_summary_matches(
        run.summary,
        checkpoint_step=200,
        checkpoint_url="https://models.example/model.zip",
        updated_at="2026-08-07T00:00:00Z",
        selection_rank=selection_rank,
    )
    assert not any(str(key).startswith("gradlab/") for key in run.summary)

    for required_key in (
        LEADER_CHECKPOINT_ARTIFACT_REF,
        LEADER_CHECKPOINT_STEP,
        LEADER_CHECKPOINT_PROJECTION_TIMESTAMP,
        LEADER_CHECKPOINT_EVALUATION_SOURCE,
    ):
        incomplete = dict(run.summary)
        incomplete.pop(required_key)
        assert not promotion_summary_matches(
            incomplete,
            checkpoint_step=200,
            checkpoint_url="https://models.example/model.zip",
            updated_at="2026-08-07T00:00:00Z",
            selection_rank=selection_rank,
        )

    incomplete = dict(run.summary)
    incomplete.pop(leader_checkpoint_progress_metric("x"))
    assert not promotion_summary_matches(
        incomplete,
        checkpoint_step=200,
        checkpoint_url="https://models.example/model.zip",
        updated_at="2026-08-07T00:00:00Z",
        selection_rank=selection_rank,
    )
