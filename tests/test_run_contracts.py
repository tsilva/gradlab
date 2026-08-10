from __future__ import annotations

import copy

import pytest

from gradlab.early_stop import (
    MetricEarlyStopStateMachine,
    MetricSample,
    validate_metric_early_stop_decision,
)
from gradlab.json_utils import canonical_json_sha256
from gradlab.run_contracts import (
    EarlyStopReceipt,
    TerminalReceipt,
    new_attempt_id,
    new_run_id,
    utc_now,
)


METRIC = "train/episode/return/shaped/origin/target/rolling/mean"


def _neutral_decision() -> tuple[dict, dict]:
    config = {
        "conditions": {
            "return_plateau": {
                "metric": METRIC,
                "trigger": "no_improvement",
                "direction": "maximize",
                "min_delta": 0.01,
                "delta_mode": "relative",
                "start_after_steps": 0,
                "patience_steps": 10,
                "outcome": "neutral",
                "action": "stop",
            }
        }
    }
    machine = MetricEarlyStopStateMachine(config)
    machine.update({METRIC: MetricSample(value=100.0, step=0)})
    update = machine.update({METRIC: MetricSample(value=100.0, step=10)})
    assert update.stop_decision is not None
    decision = validate_metric_early_stop_decision(update.stop_decision, config)
    assert machine.config_sha256 == canonical_json_sha256(machine.config)
    return machine.config, decision


def _stopped_receipt(*, evaluated: bool = True) -> TerminalReceipt:
    _config, decision = _neutral_decision()
    run_id = new_run_id()
    attempt_id = new_attempt_id()
    early_stop = EarlyStopReceipt(
        run_id=run_id,
        attempt_id=attempt_id,
        condition_id=str(decision["condition_id"]),
        matched_condition_ids=tuple(decision["matched_condition_ids"]),
        outcome="neutral",
        trigger="no_improvement",
        metric=METRIC,
        metric_step=int(decision["metric_step"]),
        value=float(decision["value"]),
        best_value=float(decision["best_value"]),
        elapsed_steps=int(decision["elapsed_steps"]),
        patience_progress=float(decision["patience_progress"]),
        condition=dict(decision["condition"]),
        early_stop_config_sha256=str(decision["early_stop_config_sha256"]),
        decision_sha256=canonical_json_sha256(decision),
        recorded_at=utc_now(),
    )
    checkpoint_id = "checkpoint-10-" + "a" * 16
    return TerminalReceipt(
        run_id=run_id,
        attempt_id=attempt_id,
        state="stopped",
        acceptance_required=evaluated,
        stop_reason="early_stop_neutral:return_plateau",
        final_step=10,
        checkpoint_inventory=(
            {"checkpoint_id": checkpoint_id, "step": 10, "purpose": "final"},
        ),
        eval_inventory=(
            ({"checkpoint_id": checkpoint_id, "checkpoint_step": 10, "status": "rejected"},)
            if evaluated
            else ()
        ),
        wandb_high_water_mark=4,
        drain={
            "complete": True,
            "metric_segment_high_water": 4,
            "wandb_remote_high_water_mark": 4,
        },
        completed_at=utc_now(),
        early_stop=early_stop.to_dict(),
    )


def test_neutral_plateau_normalization_hashing_and_stopped_receipt_validation() -> None:
    receipt = _stopped_receipt()

    assert TerminalReceipt.from_dict(receipt.to_dict()) == receipt
    assert receipt.early_stop["outcome"] == "neutral"
    assert receipt.state == "stopped"


def test_terminal_receipt_validates_durable_learner_log_evidence() -> None:
    receipt = _stopped_receipt()
    document = receipt.to_dict()
    raw_sha256 = "b" * 64
    object_key = (
        f"runs/{receipt.run_id}/attempts/{receipt.attempt_id}/evidence/learner-{raw_sha256}.log.gz"
    )
    document["drain"]["learner_log"] = {
        "path": "learner.log",
        "size_bytes": 123,
        "sha256": raw_sha256,
        "tail": "latest native failure",
        "archive": {
            "state": "complete",
            "attempts": 1,
            "object_key": object_key,
            "content_encoding": "gzip",
            "size_bytes": 80,
            "sha256": "c" * 64,
            "failure": None,
        },
    }

    assert TerminalReceipt.from_dict(document).drain["learner_log"]["archive"]["state"] == (
        "complete"
    )

    failed_archive = copy.deepcopy(document)
    failed_archive["drain"]["learner_log"]["archive"] = {
        "state": "failed",
        "attempts": 3,
        "object_key": None,
        "content_encoding": "gzip",
        "size_bytes": None,
        "sha256": None,
        "failure": {"type": "RuntimeError", "message": "R2 archive unavailable"},
    }
    assert (
        TerminalReceipt.from_dict(failed_archive).drain["learner_log"]["archive"]["state"]
        == "failed"
    )

    document["drain"]["learner_log"]["archive"]["object_key"] = "wrong/key"
    with pytest.raises(ValueError, match="archive key is invalid"):
        TerminalReceipt.from_dict(document)


def test_canceled_receipt_requires_complete_drain_and_final_checkpoint() -> None:
    run_id = new_run_id()
    attempt_id = new_attempt_id()
    checkpoint_id = "checkpoint-20-" + "a" * 16
    receipt = TerminalReceipt(
        run_id=run_id,
        attempt_id=attempt_id,
        state="canceled",
        acceptance_required=True,
        stop_reason="canceled",
        final_step=20,
        checkpoint_inventory=(
            {"checkpoint_id": checkpoint_id, "step": 20, "purpose": "final"},
        ),
        eval_inventory=(),
        wandb_high_water_mark=2,
        drain={
            "complete": True,
            "metric_segment_high_water": 2,
            "wandb_remote_high_water_mark": 2,
        },
        completed_at=utc_now(),
    )

    assert TerminalReceipt.from_dict(receipt.to_dict()) == receipt
    missing_final = copy.deepcopy(receipt.to_dict())
    missing_final["checkpoint_inventory"][0]["purpose"] = "periodic"
    with pytest.raises(ValueError, match="requires a final checkpoint"):
        TerminalReceipt.from_dict(missing_final)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda row: row.update(stop_reason="early_stop_failure:return_plateau"),
            "stop_reason does not match",
        ),
        (
            lambda row: row["early_stop"]["condition"].update(outcome="failure"),
            "condition is not a neutral stop",
        ),
        (
            lambda row: row["drain"].update(wandb_remote_high_water_mark=3),
            "not remotely visible",
        ),
        (
            lambda row: row["eval_inventory"][0].update(status="failed"),
            "valid rejection",
        ),
    ],
)
def test_stopped_receipt_rejects_tampering(mutate, message: str) -> None:
    document = copy.deepcopy(_stopped_receipt().to_dict())
    mutate(document)

    with pytest.raises(ValueError, match=message):
        TerminalReceipt.from_dict(document)


def test_training_only_stopped_receipt_needs_no_evaluation_inventory() -> None:
    receipt = _stopped_receipt(evaluated=False)

    receipt.validate()


@pytest.mark.parametrize(
    ("status", "step"),
    [("open", 10), ("closed", 9)],
)
def test_stopped_receipt_requires_state_archive_closed_at_final_step(
    status: str,
    step: int,
) -> None:
    document = _stopped_receipt(evaluated=False).to_dict()
    document["state_archive"] = {
        "semantic_id": "state-archive-publication-v1",
        "schema_version": 1,
        "run_id": document["run_id"],
        "attempt_id": document["attempt_id"],
        "generation_sha256": "c" * 64,
        "inventory_sha256": "d" * 64,
        "generation_key": "runs/archive/generation.json",
        "step": step,
        "file_count": 1,
        "size_bytes": 1,
        "status": status,
    }

    with pytest.raises(ValueError, match="state_archive"):
        TerminalReceipt.from_dict(document)
