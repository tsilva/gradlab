from __future__ import annotations

import copy

import pytest

from rlab.early_stop import (
    MetricEarlyStopStateMachine,
    MetricSample,
    normalize_metric_early_stop_config,
    validate_metric_early_stop_decision,
)
from rlab.metric_names import (
    TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN,
    TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MIN,
)


def plateau_condition(
    *,
    direction: str = "maximize",
    delta_mode: str = "relative",
    min_delta: float = 0.01,
    start_after_steps: int = 100,
    patience_steps: int = 100,
    outcome: str = "failure",
    action: str = "stop",
) -> dict:
    return {
        "metric": TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN,
        "trigger": "no_improvement",
        "direction": direction,
        "min_delta": min_delta,
        "delta_mode": delta_mode,
        "start_after_steps": start_after_steps,
        "patience_steps": patience_steps,
        "outcome": outcome,
        "action": action,
    }


def threshold_condition(
    *,
    operator: str = ">=",
    threshold: float = 1.0,
    patience_steps: int = 0,
    outcome: str = "success",
    action: str = "stop",
) -> dict:
    return {
        "metric": TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MIN,
        "trigger": "threshold",
        "operator": operator,
        "threshold": threshold,
        "patience_steps": patience_steps,
        "outcome": outcome,
        "action": action,
    }


def update(machine: MetricEarlyStopStateMachine, metric: str, value: float, step: int):
    return machine.update({metric: MetricSample(value=value, step=step)})


def test_threshold_zero_patience_fires_immediately() -> None:
    machine = MetricEarlyStopStateMachine(
        {"conditions": {"clear": threshold_condition()}}
    )

    result = update(machine, TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MIN, 1.0, 10)

    assert result.stop_decision is not None
    assert result.stop_decision["condition_id"] == "clear"
    assert result.stop_decision["outcome"] == "success"


def test_threshold_patience_requires_continuous_truth_and_resets() -> None:
    machine = MetricEarlyStopStateMachine(
        {
            "conditions": {
                "clear": threshold_condition(patience_steps=100),
            }
        }
    )

    assert update(machine, TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MIN, 1.0, 100).stop_decision is None
    assert update(machine, TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MIN, 0.5, 150).stop_decision is None
    assert update(machine, TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MIN, 1.0, 200).stop_decision is None
    result = update(machine, TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MIN, 1.0, 300)

    assert result.stop_decision is not None
    assert result.observations["clear"].elapsed_steps == 100


def test_plateau_tracks_relative_improvement_after_warmup() -> None:
    machine = MetricEarlyStopStateMachine(
        {"conditions": {"plateau": plateau_condition()}}
    )

    update(machine, TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN, 100.0, 50)
    update(machine, TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN, 100.5, 100)
    improvement = update(
        machine,
        TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN,
        101.0,
        150,
    )
    assert improvement.observations["plateau"].best_value == 101.0
    assert improvement.observations["plateau"].elapsed_steps == 0

    assert update(
        machine,
        TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN,
        101.5,
        200,
    ).stop_decision is None
    result = update(
        machine,
        TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN,
        101.5,
        250,
    )

    assert result.stop_decision is not None
    assert result.stop_decision["outcome"] == "failure"


def test_minimize_plateau_uses_absolute_improvement() -> None:
    machine = MetricEarlyStopStateMachine(
        {
            "conditions": {
                "loss": plateau_condition(
                    direction="minimize",
                    delta_mode="absolute",
                    min_delta=2.0,
                    start_after_steps=0,
                    patience_steps=10,
                )
            }
        }
    )

    update(machine, TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN, 10.0, 0)
    small = update(machine, TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN, 9.0, 5)
    assert small.observations["loss"].best_value == 9.0
    assert small.observations["loss"].elapsed_steps == 5
    improved = update(machine, TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN, 8.0, 8)
    assert improved.observations["loss"].best_value == 8.0
    result = update(machine, TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN, 8.5, 18)
    assert result.stop_decision is not None


def test_invalid_duplicate_and_out_of_order_samples_do_not_advance_patience() -> None:
    machine = MetricEarlyStopStateMachine(
        {
            "conditions": {
                "plateau": plateau_condition(
                    start_after_steps=0,
                    patience_steps=10,
                )
            }
        }
    )

    update(machine, TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN, 10.0, 0)
    assert not update(
        machine,
        TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN,
        10.0,
        0,
    ).observations
    assert not update(
        machine,
        TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN,
        10.0,
        -1,
    ).observations
    assert not update(
        machine,
        TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN,
        float("nan"),
        20,
    ).observations
    assert update(
        machine,
        TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN,
        10.0,
        9,
    ).stop_decision is None


def test_observe_mode_reports_and_can_recover_without_stopping() -> None:
    machine = MetricEarlyStopStateMachine(
        {
            "conditions": {
                "plateau": plateau_condition(
                    start_after_steps=0,
                    patience_steps=10,
                    action="observe",
                )
            }
        }
    )

    update(machine, TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN, 10.0, 0)
    triggered = update(
        machine,
        TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN,
        10.0,
        10,
    )
    assert triggered.stop_decision is None
    assert triggered.observations["plateau"].would_trigger

    recovered = update(
        machine,
        TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN,
        11.0,
        20,
    )
    assert not recovered.observations["plateau"].would_trigger
    assert recovered.observations["plateau"].patience_progress == 0.0


def test_success_wins_when_success_and_failure_stop_together() -> None:
    machine = MetricEarlyStopStateMachine(
        {
            "conditions": {
                "failure": threshold_condition(outcome="failure"),
                "success": threshold_condition(outcome="success"),
            }
        }
    )

    result = update(machine, TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MIN, 1.0, 10)

    assert result.stop_decision is not None
    assert result.stop_decision["condition_id"] == "success"
    assert result.stop_decision["matched_condition_ids"] == ["failure", "success"]


def test_decision_validation_rejects_tampering() -> None:
    config = {"conditions": {"clear": threshold_condition()}}
    machine = MetricEarlyStopStateMachine(config)
    result = update(machine, TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MIN, 1.0, 10)
    assert result.stop_decision is not None
    assert validate_metric_early_stop_decision(result.stop_decision, config)

    tampered = copy.deepcopy(result.stop_decision)
    tampered["outcome"] = "failure"
    with pytest.raises(ValueError, match="does not match"):
        validate_metric_early_stop_decision(tampered, config)


@pytest.mark.parametrize(
    "config, message",
    [
        ([], "must be an object"),
        ({"conditions": {}}, "non-empty object"),
        ({"conditions": {1: threshold_condition()}}, "non-string condition id"),
        (
            {"conditions": {"bad/id": threshold_condition()}},
            "invalid condition id",
        ),
        (
            {
                "conditions": {
                    "bad": {
                        **threshold_condition(),
                        "metric": "eval/full/outcome/success/rate/min",
                    }
                }
            },
            "train/\\*",
        ),
        (
            {
                "conditions": {
                    "bad": {
                        **plateau_condition(),
                        "operator": ">=",
                    }
                }
            },
            "unexpected keys",
        ),
    ],
)
def test_config_validation_rejects_malformed_conditions(config, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_metric_early_stop_config(config)
