from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from gradlab.json_utils import canonical_json_sha256
from gradlab.metric_names import metric_path_segment, validate_metric_name
from gradlab.validation import label_path as _label_path
from gradlab.validation import require_mapping as _require_mapping


EARLY_STOP_OPERATORS = {
    ">": lambda value, threshold: value > threshold,
    ">=": lambda value, threshold: value >= threshold,
    "<": lambda value, threshold: value < threshold,
    "<=": lambda value, threshold: value <= threshold,
}
METRIC_THRESHOLD_RULE_KEYS = frozenset({"metric", "operator", "threshold"})
METRIC_EARLY_STOP_CONFIG_KEYS = frozenset({"conditions"})
METRIC_EARLY_STOP_COMMON_KEYS = frozenset(
    {
        "action",
        "metric",
        "outcome",
        "patience_steps",
        "start_after_steps",
        "trigger",
    }
)
METRIC_EARLY_STOP_THRESHOLD_KEYS = frozenset({"operator", "progress_baseline", "threshold"})
METRIC_EARLY_STOP_NO_IMPROVEMENT_KEYS = frozenset({"delta_mode", "direction", "min_delta"})
METRIC_EARLY_STOP_TRIGGERS = frozenset({"no_improvement", "threshold"})
METRIC_EARLY_STOP_OUTCOMES = frozenset({"failure", "success"})
METRIC_EARLY_STOP_ACTIONS = frozenset({"observe", "stop"})
METRIC_EARLY_STOP_DIRECTIONS = frozenset({"maximize", "minimize"})
METRIC_EARLY_STOP_DELTA_MODES = frozenset({"absolute", "relative"})
METRIC_EARLY_STOP_DECISION_KEYS = frozenset(
    {
        "action",
        "best_value",
        "condition",
        "condition_id",
        "early_stop_config_sha256",
        "elapsed_steps",
        "kind",
        "matched_condition_ids",
        "metric",
        "metric_step",
        "outcome",
        "patience_progress",
        "schema_version",
        "trigger",
        "value",
    }
)


def _require_non_empty_string(document: Mapping[str, Any], key: str, *, label: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{_label_path(label, key)} must be a non-empty string")
    return value.strip()


def _require_choice(
    document: Mapping[str, Any],
    key: str,
    *,
    choices: frozenset[str],
    label: str,
) -> str:
    value = _require_non_empty_string(document, key, label=label)
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{_label_path(label, key)} must be one of {allowed}")
    return value


def _require_finite_number(document: Mapping[str, Any], key: str, *, label: str) -> float:
    value = document.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{_label_path(label, key)} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{_label_path(label, key)} must be finite")
    return value


def _non_negative_integer(
    document: Mapping[str, Any],
    key: str,
    *,
    label: str,
    default: int | None = None,
) -> int:
    value = document.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{_label_path(label, key)} must be a non-negative integer")
    return int(value)


def normalize_metric_threshold_rule(
    value: Any,
    *,
    label: str,
    metric_validator: Callable[[str], object] = validate_metric_name,
) -> dict[str, Any]:
    """Normalize one stateless rule used by checkpoint acceptance."""

    node = _require_mapping(value, label=label)
    extra_keys = sorted(set(node) - METRIC_THRESHOLD_RULE_KEYS)
    if extra_keys:
        raise ValueError(f"{label} has unexpected keys: {extra_keys}")
    metric = _require_non_empty_string(node, "metric", label=label)
    try:
        metric_validator(metric)
    except ValueError as exc:
        raise ValueError(
            f"{_label_path(label, 'metric')} is not a registered metric: {metric}"
        ) from exc
    operator = _require_non_empty_string(node, "operator", label=label)
    if operator not in EARLY_STOP_OPERATORS:
        allowed = ", ".join(sorted(EARLY_STOP_OPERATORS))
        raise ValueError(f"{_label_path(label, 'operator')} must be one of {allowed}")
    threshold = _require_finite_number(node, "threshold", label=label)
    return {"metric": metric, "operator": operator, "threshold": threshold}


def normalize_metric_threshold_rules(
    value: Any,
    *,
    label: str,
    metric_validator: Callable[[str], object] = validate_metric_name,
) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{label} must be a non-empty list")
    if not value:
        raise ValueError(f"{label} must be a non-empty list")
    return [
        normalize_metric_threshold_rule(
            rule,
            label=f"{label}[{index}]",
            metric_validator=metric_validator,
        )
        for index, rule in enumerate(value)
    ]


def evaluate_metric_threshold_rules(
    config: Sequence[Mapping[str, Any]],
    value_lookup: Callable[[str], float | None],
) -> tuple[bool | None, dict[str, float]]:
    """Evaluate AND-combined stateless checkpoint-acceptance rules."""

    values: dict[str, float] = {}
    pending = False
    for rule in config:
        metric = str(rule["metric"])
        value = value_lookup(metric)
        if value is None:
            pending = True
            continue
        values[metric] = value
        operator = str(rule["operator"])
        threshold = float(rule["threshold"])
        if not EARLY_STOP_OPERATORS[operator](value, threshold):
            return False, values
    return (None if pending else True), values


def _normalize_metric_early_stop_condition(
    condition_id: str,
    value: Any,
    *,
    label: str,
) -> dict[str, Any]:
    node = _require_mapping(value, label=label)
    trigger = _require_choice(
        node,
        "trigger",
        choices=METRIC_EARLY_STOP_TRIGGERS,
        label=label,
    )
    trigger_keys = (
        METRIC_EARLY_STOP_THRESHOLD_KEYS
        if trigger == "threshold"
        else METRIC_EARLY_STOP_NO_IMPROVEMENT_KEYS
    )
    extra_keys = sorted(set(node) - METRIC_EARLY_STOP_COMMON_KEYS - trigger_keys)
    if extra_keys:
        raise ValueError(f"{label} has unexpected keys: {extra_keys}")

    metric = _require_non_empty_string(node, "metric", label=label)
    try:
        validate_metric_name(metric)
    except ValueError as exc:
        raise ValueError(
            f"{_label_path(label, 'metric')} is not a registered metric: {metric}"
        ) from exc
    if not metric.startswith("train/"):
        raise ValueError(f"{_label_path(label, 'metric')} must use a train/* metric")

    normalized: dict[str, Any] = {
        "metric": metric,
        "trigger": trigger,
        "outcome": _require_choice(
            node,
            "outcome",
            choices=METRIC_EARLY_STOP_OUTCOMES,
            label=label,
        ),
        "action": _require_choice(
            node,
            "action",
            choices=METRIC_EARLY_STOP_ACTIONS,
            label=label,
        ),
        "start_after_steps": _non_negative_integer(
            node,
            "start_after_steps",
            label=label,
            default=0,
        ),
        "patience_steps": _non_negative_integer(node, "patience_steps", label=label),
    }
    if trigger == "threshold":
        operator = _require_non_empty_string(node, "operator", label=label)
        if operator not in EARLY_STOP_OPERATORS:
            allowed = ", ".join(sorted(EARLY_STOP_OPERATORS))
            raise ValueError(f"{_label_path(label, 'operator')} must be one of {allowed}")
        threshold = _require_finite_number(node, "threshold", label=label)
        normalized.update(
            {
                "operator": operator,
                "threshold": threshold,
            }
        )
        if "progress_baseline" in node:
            progress_baseline = _require_finite_number(
                node,
                "progress_baseline",
                label=label,
            )
            direction = "maximize" if operator in {">", ">="} else "minimize"
            invalid_baseline = (
                progress_baseline >= threshold
                if direction == "maximize"
                else progress_baseline <= threshold
            )
            if invalid_baseline:
                relation = "below" if direction == "maximize" else "above"
                raise ValueError(
                    f"{_label_path(label, 'progress_baseline')} must be {relation} "
                    f"threshold for operator {operator}"
                )
            normalized["progress_baseline"] = progress_baseline
    else:
        min_delta = _require_finite_number(node, "min_delta", label=label)
        if min_delta < 0:
            raise ValueError(f"{_label_path(label, 'min_delta')} must be non-negative")
        normalized.update(
            {
                "direction": _require_choice(
                    node,
                    "direction",
                    choices=METRIC_EARLY_STOP_DIRECTIONS,
                    label=label,
                ),
                "min_delta": min_delta,
                "delta_mode": _require_choice(
                    node,
                    "delta_mode",
                    choices=METRIC_EARLY_STOP_DELTA_MODES,
                    label=label,
                ),
            }
        )
    metric_path_segment(condition_id)
    return normalized


def normalize_metric_early_stop_config(
    value: Any,
    *,
    label: str = "early_stop",
) -> dict[str, Any]:
    root = _require_mapping(value, label=label)
    extra_keys = sorted(set(root) - METRIC_EARLY_STOP_CONFIG_KEYS)
    if extra_keys:
        raise ValueError(f"{label} has unexpected keys: {extra_keys}")
    conditions = _require_mapping(root.get("conditions"), label=f"{label}.conditions")
    if not conditions:
        raise ValueError(f"{label}.conditions must be a non-empty object")
    normalized: dict[str, dict[str, Any]] = {}
    for raw_condition_id in sorted(conditions, key=str):
        if not isinstance(raw_condition_id, str):
            raise ValueError(
                f"{label}.conditions has non-string condition id: {raw_condition_id!r}"
            )
        condition_id = raw_condition_id.strip()
        try:
            metric_path_segment(condition_id)
        except ValueError as exc:
            raise ValueError(
                f"{label}.conditions has invalid condition id: {raw_condition_id!r}"
            ) from exc
        normalized[condition_id] = _normalize_metric_early_stop_condition(
            condition_id,
            conditions[raw_condition_id],
            label=f"{label}.conditions.{condition_id}",
        )
    return {"conditions": normalized}


def validate_metric_early_stop_decision(
    value: Any,
    config: Any,
    *,
    label: str = "early_stop_decision",
) -> dict[str, Any]:
    decision = _require_mapping(value, label=label)
    extra_keys = sorted(set(decision) - METRIC_EARLY_STOP_DECISION_KEYS)
    missing_keys = sorted(METRIC_EARLY_STOP_DECISION_KEYS - set(decision))
    if extra_keys or missing_keys:
        raise ValueError(
            f"{label} keys do not match the decision contract; "
            f"missing={missing_keys} unexpected={extra_keys}"
        )
    normalized_config = normalize_metric_early_stop_config(config, label="early_stop")
    expected_hash = canonical_json_sha256(normalized_config)
    if str(decision["early_stop_config_sha256"]) != expected_hash:
        raise ValueError(f"{label}.early_stop_config_sha256 does not match the train config")
    if decision["schema_version"] != 1 or decision["kind"] != "metric_early_stop":
        raise ValueError(f"{label} has an unsupported schema or kind")
    if decision["action"] != "stop":
        raise ValueError(f"{label}.action must be stop")

    conditions: Mapping[str, Mapping[str, Any]] = normalized_config["conditions"]
    condition_id = str(decision["condition_id"])
    condition = conditions.get(condition_id)
    if condition is None:
        raise ValueError(f"{label}.condition_id is not configured: {condition_id}")
    matched_value = decision["matched_condition_ids"]
    if (
        not isinstance(matched_value, Sequence)
        or isinstance(matched_value, str | bytes)
        or not matched_value
    ):
        raise ValueError(f"{label}.matched_condition_ids must be a non-empty list")
    matched = [str(item) for item in matched_value]
    if matched != sorted(set(matched)) or condition_id not in matched:
        raise ValueError(
            f"{label}.matched_condition_ids must be sorted, unique, and contain condition_id"
        )
    for matched_id in matched:
        matched_condition = conditions.get(matched_id)
        if matched_condition is None or str(matched_condition["action"]) != "stop":
            raise ValueError(f"{label} references a non-stopping condition: {matched_id}")
    expected_selected = sorted(
        matched,
        key=lambda item: (
            0 if str(conditions[item]["outcome"]) == "success" else 1,
            item,
        ),
    )[0]
    if condition_id != expected_selected:
        raise ValueError(f"{label}.condition_id does not follow success-first precedence")

    for key in ("outcome", "trigger", "metric"):
        if str(decision[key]) != str(condition[key]):
            raise ValueError(f"{label}.{key} does not match the configured condition")
    if dict(_require_mapping(decision["condition"], label=f"{label}.condition")) != dict(
        condition
    ):
        raise ValueError(f"{label}.condition does not match the normalized train config")
    metric_step = decision["metric_step"]
    elapsed_steps = decision["elapsed_steps"]
    if (
        not isinstance(metric_step, int)
        or isinstance(metric_step, bool)
        or metric_step < 0
        or not isinstance(elapsed_steps, int)
        or isinstance(elapsed_steps, bool)
        or elapsed_steps < 0
    ):
        raise ValueError(f"{label} step fields must be non-negative integers")
    for key in ("value", "best_value", "patience_progress"):
        raw = decision[key]
        if (
            not isinstance(raw, int | float)
            or isinstance(raw, bool)
            or not math.isfinite(float(raw))
        ):
            raise ValueError(f"{label}.{key} must be finite")
    if not math.isclose(float(decision["patience_progress"]), 1.0):
        raise ValueError(f"{label}.patience_progress must be one for a stop decision")
    return {
        **dict(decision),
        "condition_id": condition_id,
        "matched_condition_ids": matched,
        "condition": dict(condition),
        "metric_step": int(metric_step),
        "elapsed_steps": int(elapsed_steps),
        "value": float(decision["value"]),
        "best_value": float(decision["best_value"]),
        "patience_progress": float(decision["patience_progress"]),
    }


@dataclass(frozen=True)
class MetricSample:
    value: float
    step: int


@dataclass(frozen=True)
class MetricEarlyStopObservation:
    condition_id: str
    metric: str
    metric_step: int
    value: float
    best_value: float
    elapsed_steps: int
    patience_progress: float
    target_progress: float | None
    eligible: bool
    would_trigger: bool


@dataclass(frozen=True)
class MetricEarlyStopUpdate:
    observations: Mapping[str, MetricEarlyStopObservation]
    would_trigger_condition_ids: tuple[str, ...]
    stop_decision: Mapping[str, Any] | None


@dataclass
class _MetricEarlyStopRuntime:
    last_sample_step: int | None = None
    best_value: float | None = None
    patience_reference_value: float | None = None
    last_improvement_step: int | None = None
    predicate_since_step: int | None = None


class MetricEarlyStopStateMachine:
    """Deterministic stateful evaluator for training-owned metric conditions."""

    def __init__(self, config: Any, *, label: str = "early_stop") -> None:
        self.config = normalize_metric_early_stop_config(config, label=label)
        self.conditions: Mapping[str, Mapping[str, Any]] = self.config["conditions"]
        self.states = {
            condition_id: _MetricEarlyStopRuntime() for condition_id in self.conditions
        }
        self.config_sha256 = canonical_json_sha256(self.config)

    @staticmethod
    def _valid_sample(value: Any) -> MetricSample | None:
        if not isinstance(value, MetricSample):
            return None
        if (
            not isinstance(value.step, int)
            or isinstance(value.step, bool)
            or value.step < 0
            or not isinstance(value.value, int | float)
            or isinstance(value.value, bool)
            or not math.isfinite(float(value.value))
        ):
            return None
        return MetricSample(value=float(value.value), step=int(value.step))

    @staticmethod
    def _threshold_direction(operator: str) -> Literal["maximize", "minimize"]:
        return "maximize" if operator in {">", ">="} else "minimize"

    @staticmethod
    def _best_value(
        current_best: float | None,
        value: float,
        *,
        direction: str,
    ) -> float:
        if current_best is None:
            return value
        if direction == "maximize":
            return max(current_best, value)
        return min(current_best, value)

    @staticmethod
    def _meaningful_improvement(
        condition: Mapping[str, Any],
        *,
        best_value: float,
        value: float,
    ) -> bool:
        direction = str(condition["direction"])
        improvement = value - best_value if direction == "maximize" else best_value - value
        delta = float(condition["min_delta"])
        if str(condition["delta_mode"]) == "relative":
            delta *= max(abs(best_value), 1.0)
        return improvement > 0.0 and improvement >= delta

    @staticmethod
    def _progress(*, eligible: bool, elapsed: int, patience: int, triggered: bool) -> float:
        if not eligible:
            return 0.0
        if patience == 0:
            return 1.0 if triggered else 0.0
        return min(1.0, max(0.0, elapsed / patience))

    @staticmethod
    def _target_progress(
        condition: Mapping[str, Any],
        *,
        value: float,
    ) -> float | None:
        if "progress_baseline" not in condition:
            return None
        baseline = float(condition["progress_baseline"])
        threshold = float(condition["threshold"])
        progress = (value - baseline) / (threshold - baseline)
        return min(1.0, max(0.0, progress))

    def _update_threshold(
        self,
        *,
        condition_id: str,
        condition: Mapping[str, Any],
        state: _MetricEarlyStopRuntime,
        sample: MetricSample,
    ) -> MetricEarlyStopObservation:
        direction = self._threshold_direction(str(condition["operator"]))
        state.best_value = self._best_value(
            state.best_value,
            sample.value,
            direction=direction,
        )
        matched = EARLY_STOP_OPERATORS[str(condition["operator"])](
            sample.value,
            float(condition["threshold"]),
        )
        eligible = sample.step >= int(condition["start_after_steps"])
        if not matched:
            state.predicate_since_step = None
            elapsed = 0
            triggered = False
        else:
            if state.predicate_since_step is None:
                state.predicate_since_step = max(
                    sample.step,
                    int(condition["start_after_steps"]),
                )
            elapsed = max(0, sample.step - state.predicate_since_step)
            patience = int(condition["patience_steps"])
            triggered = eligible and (patience == 0 or elapsed >= patience)
        patience = int(condition["patience_steps"])
        return MetricEarlyStopObservation(
            condition_id=condition_id,
            metric=str(condition["metric"]),
            metric_step=sample.step,
            value=sample.value,
            best_value=float(state.best_value),
            elapsed_steps=elapsed,
            patience_progress=self._progress(
                eligible=eligible,
                elapsed=elapsed,
                patience=patience,
                triggered=triggered,
            ),
            target_progress=self._target_progress(condition, value=sample.value),
            eligible=eligible,
            would_trigger=triggered,
        )

    def _update_no_improvement(
        self,
        *,
        condition_id: str,
        condition: Mapping[str, Any],
        state: _MetricEarlyStopRuntime,
        sample: MetricSample,
    ) -> MetricEarlyStopObservation:
        initialized = state.best_value is not None
        if not initialized:
            state.best_value = sample.value
            state.patience_reference_value = sample.value
            state.last_improvement_step = sample.step
        elif self._meaningful_improvement(
            condition,
            best_value=float(state.patience_reference_value),
            value=sample.value,
        ):
            state.patience_reference_value = sample.value
            state.last_improvement_step = sample.step
        state.best_value = self._best_value(
            state.best_value,
            sample.value,
            direction=str(condition["direction"]),
        )

        eligible = sample.step >= int(condition["start_after_steps"])
        anchor = max(
            int(condition["start_after_steps"]),
            int(state.last_improvement_step if state.last_improvement_step is not None else sample.step),
        )
        elapsed = max(0, sample.step - anchor) if eligible else 0
        patience = int(condition["patience_steps"])
        triggered = initialized and eligible and (patience == 0 or elapsed >= patience)
        return MetricEarlyStopObservation(
            condition_id=condition_id,
            metric=str(condition["metric"]),
            metric_step=sample.step,
            value=sample.value,
            best_value=float(state.best_value),
            elapsed_steps=elapsed,
            patience_progress=self._progress(
                eligible=eligible,
                elapsed=elapsed,
                patience=patience,
                triggered=triggered,
            ),
            target_progress=None,
            eligible=eligible,
            would_trigger=triggered,
        )

    def update(self, samples: Mapping[str, MetricSample | None]) -> MetricEarlyStopUpdate:
        observations: dict[str, MetricEarlyStopObservation] = {}
        for condition_id, condition in self.conditions.items():
            sample = self._valid_sample(samples.get(str(condition["metric"])))
            if sample is None:
                continue
            state = self.states[condition_id]
            if state.last_sample_step is not None and sample.step <= state.last_sample_step:
                continue
            state.last_sample_step = sample.step
            if str(condition["trigger"]) == "threshold":
                observation = self._update_threshold(
                    condition_id=condition_id,
                    condition=condition,
                    state=state,
                    sample=sample,
                )
            else:
                observation = self._update_no_improvement(
                    condition_id=condition_id,
                    condition=condition,
                    state=state,
                    sample=sample,
                )
            observations[condition_id] = observation

        would_trigger = tuple(
            sorted(
                condition_id
                for condition_id, observation in observations.items()
                if observation.would_trigger
            )
        )
        stop_matches = [
            condition_id
            for condition_id in would_trigger
            if str(self.conditions[condition_id]["action"]) == "stop"
        ]
        stop_matches.sort(
            key=lambda condition_id: (
                0 if str(self.conditions[condition_id]["outcome"]) == "success" else 1,
                condition_id,
            )
        )
        decision: dict[str, Any] | None = None
        if stop_matches:
            selected_id = stop_matches[0]
            condition = dict(self.conditions[selected_id])
            observation = observations[selected_id]
            decision = {
                "schema_version": 1,
                "kind": "metric_early_stop",
                "condition_id": selected_id,
                "matched_condition_ids": sorted(stop_matches),
                "outcome": str(condition["outcome"]),
                "action": "stop",
                "trigger": str(condition["trigger"]),
                "metric": observation.metric,
                "metric_step": observation.metric_step,
                "value": observation.value,
                "best_value": observation.best_value,
                "elapsed_steps": observation.elapsed_steps,
                "patience_progress": observation.patience_progress,
                "condition": condition,
                "early_stop_config_sha256": self.config_sha256,
            }
        return MetricEarlyStopUpdate(
            observations=observations,
            would_trigger_condition_ids=would_trigger,
            stop_decision=decision,
        )
