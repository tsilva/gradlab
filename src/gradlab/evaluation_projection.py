from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from gradlab.metric_names import (
    EVAL_ACCEPTANCE_EPISODE_COMPLETED_COUNT,
    EVAL_ACCEPTANCE_EPISODE_PLANNED_COUNT,
    EVAL_ACCEPTANCE_PASS,
    EVAL_CHECKPOINT_STEP,
    EVAL_FULL_EPISODE_RETURN_SHAPED_MAX,
    EVAL_FULL_EPISODE_RETURN_SHAPED_MEAN,
    EVAL_FULL_OUTCOME_SUCCESS_STARTS_RATE_MEAN,
    EVAL_FULL_OUTCOME_SUCCESS_STARTS_RATE_MIN,
    metric_definition,
    require_current_metrics_schema,
)


_CURRENT_FIXED_FULL_METRICS = frozenset(
    {
        EVAL_FULL_EPISODE_RETURN_SHAPED_MEAN,
        EVAL_FULL_EPISODE_RETURN_SHAPED_MAX,
        EVAL_FULL_OUTCOME_SUCCESS_STARTS_RATE_MIN,
        EVAL_FULL_OUTCOME_SUCCESS_STARTS_RATE_MEAN,
    }
)
_PROGRESS_METRIC_RE = re.compile(r"^eval/full/progress/[A-Za-z0-9_.-]+/(?:mean|max)$")


def metrics_schema_version_from_recipe_document(document: Mapping[str, Any]) -> int:
    recipe = document.get("recipe")
    if not isinstance(recipe, Mapping):
        raise ValueError("checkpoint recipe document is missing recipe")
    train_config = recipe.get("train_config")
    if not isinstance(train_config, Mapping):
        raise ValueError("checkpoint recipe document is missing recipe.train_config")
    value = train_config.get("metrics_schema_version")
    try:
        version = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint recipe has no valid metrics_schema_version") from exc
    return require_current_metrics_schema(version)


def _allowed_full_metric(name: str, *, schema_version: int) -> bool:
    require_current_metrics_schema(schema_version)
    if _PROGRESS_METRIC_RE.fullmatch(name):
        return True
    return name in _CURRENT_FIXED_FULL_METRICS and metric_definition(name) is not None


def validate_evaluation_scientific_metric(
    name: str,
    *,
    schema_version: int,
) -> str:
    if not _allowed_full_metric(name, schema_version=schema_version):
        raise ValueError(
            f"metrics schema v{schema_version} does not allow evaluation metric: {name}"
        )
    return name


def validate_evaluation_metric_payload(
    payload: Mapping[str, Any],
    *,
    schema_version: int,
) -> None:
    """Validate the finite W&B projection allowed for one eval schema."""

    require_current_metrics_schema(schema_version)
    fixed = {
        EVAL_CHECKPOINT_STEP,
        EVAL_ACCEPTANCE_EPISODE_PLANNED_COUNT,
        EVAL_ACCEPTANCE_EPISODE_COMPLETED_COUNT,
        EVAL_ACCEPTANCE_PASS,
    }
    invalid = sorted(
        name
        for raw_name in payload
        if (name := str(raw_name)) not in fixed
        and not _allowed_full_metric(name, schema_version=schema_version)
    )
    if invalid:
        raise ValueError(
            f"metrics schema v{schema_version} does not allow evaluation metric: "
            f"{invalid[0]}"
        )


def evaluation_wandb_projection(
    aggregates: Mapping[str, Any],
    *,
    schema_version: int,
    checkpoint_step: int,
    accepted: bool,
    episodes_planned: int,
    episodes_completed: int,
) -> dict[str, Any]:
    require_current_metrics_schema(schema_version)
    projected = {
        str(name): value
        for name, value in aggregates.items()
        if _allowed_full_metric(str(name), schema_version=schema_version)
    }
    projected.update(
        {
            EVAL_CHECKPOINT_STEP: int(checkpoint_step),
            EVAL_ACCEPTANCE_PASS: 1.0 if accepted else 0.0,
            EVAL_ACCEPTANCE_EPISODE_PLANNED_COUNT: float(episodes_planned),
            EVAL_ACCEPTANCE_EPISODE_COMPLETED_COUNT: float(episodes_completed),
        }
    )
    return projected
