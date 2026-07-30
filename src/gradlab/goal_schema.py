from __future__ import annotations

from collections.abc import Mapping
from typing import Any


GOAL_FIELDS = frozenset(
    {
        "eval",
        "evaluation_mode",
        "goal_id",
        "objective",
        "release",
        "reward_shapes",
        "tags",
        "title",
        "train",
    }
)
GOAL_EVALUATION_MODES = frozenset({"evaluated", "training_only"})
GOAL_OBJECTIVE_FIELDS = frozenset({"rank", "states"})
GOAL_TRAIN_FIELDS = frozenset(
    {
        "checkpoint_freq",
        "early_stop",
        "environment",
    }
)
GOAL_EVAL_FIELDS = frozenset({"acceptance", "environment", "episodes", "policy"})
GOAL_EVAL_POLICY_FIELDS = frozenset({"stochastic"})


def _reject_unknown_fields(
    value: Any,
    *,
    allowed: frozenset[str],
    label: str,
) -> None:
    if not isinstance(value, Mapping):
        return
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        raise ValueError(f"{label} has unknown field(s): {', '.join(unknown)}")


def goal_evaluation_mode(
    document: Mapping[str, Any],
    *,
    label: str,
) -> str:
    value = document.get("evaluation_mode")
    if not isinstance(value, str) or value not in GOAL_EVALUATION_MODES:
        choices = ", ".join(sorted(GOAL_EVALUATION_MODES))
        raise ValueError(f"{label}.evaluation_mode must be one of: {choices}")
    return value


def validate_goal_document_shape(
    document: Mapping[str, Any],
    *,
    label: str,
) -> None:
    """Reject misspelled goal-owned fields without importing runtime orchestration."""

    _reject_unknown_fields(document, allowed=GOAL_FIELDS, label=label)
    goal_evaluation_mode(document, label=label)
    _reject_unknown_fields(
        document.get("objective"),
        allowed=GOAL_OBJECTIVE_FIELDS,
        label=f"{label}.objective",
    )
    _reject_unknown_fields(
        document.get("train"),
        allowed=GOAL_TRAIN_FIELDS,
        label=f"{label}.train",
    )
    _reject_unknown_fields(
        document.get("eval"),
        allowed=GOAL_EVAL_FIELDS,
        label=f"{label}.eval",
    )
    evaluation = document.get("eval")
    if isinstance(evaluation, Mapping):
        _reject_unknown_fields(
            evaluation.get("policy"),
            allowed=GOAL_EVAL_POLICY_FIELDS,
            label=f"{label}.eval.policy",
        )
