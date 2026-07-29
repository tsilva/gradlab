from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from gradlab.metric_names import (
    EVAL_ACCEPTANCE_DURATION_SECONDS,
    EVAL_ACCEPTANCE_PASS,
    LEGACY_METRICS_SCHEMA_VERSION,
    METRICS_SCHEMA_VERSION,
    V13_EVAL_FULL_EPISODE_COMPLETED_COUNT,
    V13_EVAL_FULL_EPISODE_RETURN_SHAPED_MAX,
    V13_EVAL_FULL_EPISODE_RETURN_SHAPED_MEAN,
    V13_EVAL_FULL_SUCCESS_ACROSS_STARTS_RATE_MEAN,
    V13_EVAL_FULL_SUCCESS_ACROSS_STARTS_RATE_MIN,
    evaluation_metric_schema,
    metric_definition,
)


_CURRENT_FIXED_FULL_METRICS = frozenset(
    {
        "eval/full/episode/return/shaped/mean",
        "eval/full/episode/return/shaped/std",
        "eval/full/episode/return/shaped/median",
        "eval/full/episode/return/shaped/max",
        "eval/full/episode/length/mean",
        "eval/full/episode/completed/count",
        "eval/full/outcome/success/across_starts/rate/min",
        "eval/full/outcome/success/across_starts/rate/mean",
    }
)
_LEGACY_FIXED_FULL_METRICS = frozenset(
    {
        V13_EVAL_FULL_EPISODE_RETURN_SHAPED_MEAN,
        "eval/full/episode/return/std",
        "eval/full/episode/return/median",
        V13_EVAL_FULL_EPISODE_RETURN_SHAPED_MAX,
        "eval/full/episode/length/mean",
        V13_EVAL_FULL_EPISODE_COMPLETED_COUNT,
        V13_EVAL_FULL_SUCCESS_ACROSS_STARTS_RATE_MIN,
        V13_EVAL_FULL_SUCCESS_ACROSS_STARTS_RATE_MEAN,
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
    evaluation_metric_schema(version)
    return version


def _allowed_full_metric(name: str, *, schema_version: int) -> bool:
    if _PROGRESS_METRIC_RE.fullmatch(name):
        return True
    if schema_version == METRICS_SCHEMA_VERSION:
        return name in _CURRENT_FIXED_FULL_METRICS and metric_definition(name) is not None
    if schema_version == LEGACY_METRICS_SCHEMA_VERSION:
        return name in _LEGACY_FIXED_FULL_METRICS
    raise ValueError(f"unsupported metrics schema version: {schema_version}")


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

    schema = evaluation_metric_schema(schema_version)
    fixed = {
        schema.checkpoint_step,
        schema.acceptance_episode_planned_count,
        schema.acceptance_episode_completed_count,
        EVAL_ACCEPTANCE_PASS,
        EVAL_ACCEPTANCE_DURATION_SECONDS,
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
    raw_metrics: Mapping[str, Any],
    *,
    schema_version: int,
    checkpoint_step: int,
    accepted: bool,
    episodes_planned: int,
    episodes_completed: int,
    duration_seconds: float,
) -> dict[str, Any]:
    schema = evaluation_metric_schema(schema_version)
    projected = {
        str(name): value
        for name, value in raw_metrics.items()
        if _allowed_full_metric(str(name), schema_version=schema_version)
    }
    projected.update(
        {
            schema.checkpoint_step: int(checkpoint_step),
            EVAL_ACCEPTANCE_PASS: 1.0 if accepted else 0.0,
            schema.acceptance_episode_planned_count: float(episodes_planned),
            schema.acceptance_episode_completed_count: float(episodes_completed),
            EVAL_ACCEPTANCE_DURATION_SECONDS: float(duration_seconds),
        }
    )
    return projected
