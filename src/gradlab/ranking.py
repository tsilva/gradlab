from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from gradlab.metric_names import (
    LEADER_CHECKPOINT_STEP,
    METRICS_SCHEMA_VERSION,
    require_current_metrics_schema,
    validate_metric_name,
)


_RANK_RE = re.compile(r"^(max|min)\(([^()]+)\)$")

@dataclass(frozen=True)
class RankCriterion:
    direction: str
    metric: str


def parse_objective_rank(
    value: Any,
    *,
    metrics_schema_version: int = METRICS_SCHEMA_VERSION,
) -> tuple[RankCriterion, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return ()
    try:
        require_current_metrics_schema(metrics_schema_version)
    except ValueError:
        return ()
    criteria: list[RankCriterion] = []
    for item in value:
        match = _RANK_RE.fullmatch(str(item).strip())
        if match is None:
            return ()
        metric = match.group(2).strip()
        try:
            validate_metric_name(metric)
        except ValueError:
            return ()
        criteria.append(RankCriterion(match.group(1), metric))
    return tuple(criteria)


def require_objective_rank(
    value: Any,
    *,
    metrics_schema_version: int = METRICS_SCHEMA_VERSION,
) -> tuple[RankCriterion, ...]:
    criteria = parse_objective_rank(
        value,
        metrics_schema_version=metrics_schema_version,
    )
    if not criteria:
        raise ValueError(
            "objective.rank must contain valid "
            f"schema-v{metrics_schema_version} metric criteria"
        )
    return criteria


def objective_rank_strings(criteria: Sequence[RankCriterion]) -> tuple[str, ...]:
    return tuple(f"{criterion.direction}({criterion.metric})" for criterion in criteria)


def _metric_value(metrics: Mapping[str, Any], metric: str) -> Any:
    if metric == LEADER_CHECKPOINT_STEP:
        return metrics.get(metric, metrics.get("checkpoint_step"))
    return metrics.get(metric)


def rank_metric_values(
    metrics: Mapping[str, Any], criteria: Sequence[RankCriterion]
) -> tuple[float | None, ...]:
    values: list[float | None] = []
    for criterion in criteria:
        value = _metric_value(metrics, criterion.metric)
        try:
            values.append(float(value) if value is not None else None)
        except TypeError, ValueError:
            values.append(None)
    return tuple(values)


def rank_score(metrics: Mapping[str, Any], criteria: Sequence[RankCriterion]) -> tuple[float, ...]:
    score: list[float] = []
    for criterion, value in zip(criteria, rank_metric_values(metrics, criteria), strict=True):
        score.append(
            float("-inf") if value is None else value if criterion.direction == "max" else -value
        )
    return tuple(score)
