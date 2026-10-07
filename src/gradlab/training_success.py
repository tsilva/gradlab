"""Goal-owned training success, independent of stopping and Acceptance."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from typing import Any

from gradlab.early_stop import EARLY_STOP_OPERATORS, normalize_metric_threshold_rule


def training_success_criterion(value: Any, *, label: str) -> dict[str, Any]:
    criterion = normalize_metric_threshold_rule(value, label=label)
    if not criterion["metric"].startswith("train/"):
        raise ValueError(f"{label}.metric must use a train/* metric")
    return criterion


def training_success_evidence(
    criterion: Mapping[str, Any],
    samples: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    normalized = training_success_criterion(criterion, label="training_success")
    operator = normalized["operator"]
    ascending = operator in {">", ">="}
    best: dict[str, int | float] | None = None
    first_met: dict[str, int | float] | None = None
    for sample in samples:
        step = sample.get("step")
        value = sample.get("value")
        if (
            sample.get("status") != "published"
            or not isinstance(step, int)
            or isinstance(step, bool)
            or not isinstance(value, int | float)
            or isinstance(value, bool)
            or step < 0
            or not math.isfinite(float(value))
        ):
            continue
        observed = {"step": step, "value": float(value)}
        if best is None or (
            observed["value"] > best["value"] if ascending else observed["value"] < best["value"]
        ):
            best = observed
        if first_met is None and EARLY_STOP_OPERATORS[operator](
            observed["value"], normalized["threshold"]
        ):
            first_met = observed
    return {
        "criterion": normalized,
        "status": "met" if first_met is not None else "not_met" if best is not None else "unavailable",
        "best": best,
        "first_met": first_met,
    }


def validate_training_success_evidence(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "criterion", "status", "best", "first_met"
    }:
        raise ValueError("training success evidence has invalid fields")
    criterion = training_success_criterion(value["criterion"], label="training_success.criterion")

    def sample(raw: Any, *, label: str) -> dict[str, int | float] | None:
        if raw is None:
            return None
        if not isinstance(raw, Mapping) or set(raw) != {"step", "value"}:
            raise ValueError(f"{label} must have step and value")
        step, observed = raw["step"], raw["value"]
        if (
            not isinstance(step, int)
            or isinstance(step, bool)
            or step < 0
            or not isinstance(observed, int | float)
            or isinstance(observed, bool)
            or not math.isfinite(float(observed))
        ):
            raise ValueError(f"{label} has invalid sample")
        return {"step": step, "value": float(observed)}

    best = sample(value["best"], label="training_success.best")
    first_met = sample(value["first_met"], label="training_success.first_met")
    status = value["status"]
    if status not in {"met", "not_met", "unavailable"}:
        raise ValueError("training success status is invalid")
    if (status == "met") != (first_met is not None) or (status == "unavailable") != (best is None):
        raise ValueError("training success status disagrees with samples")
    if first_met is not None and not EARLY_STOP_OPERATORS[criterion["operator"]](
        first_met["value"], criterion["threshold"]
    ):
        raise ValueError("training success first_met does not satisfy the criterion")
    if best is not None and (status == "not_met") and EARLY_STOP_OPERATORS[
        criterion["operator"]
    ](best["value"], criterion["threshold"]):
        raise ValueError("training success best contradicts not_met")
    return {"criterion": criterion, "status": status, "best": best, "first_met": first_met}
