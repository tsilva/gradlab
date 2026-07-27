from __future__ import annotations

import json
import socket
from pathlib import Path

from gradlab.experiment_cli import main as experiment_main
from gradlab.lifecycle_certification import (
    DEFAULT_SCENARIOS,
    SCENARIOS,
    replay_simulated_certification,
    run_simulated_certification,
)


def test_complete_tier1_report_is_passing_and_byte_deterministic() -> None:
    first = run_simulated_certification()
    second = run_simulated_certification()

    assert first == second
    assert first["status"] == "passed"
    assert first["network_access"] == "denied"
    assert first["credential_requirement"] == "none"
    assert [row["name"] for row in first["scenarios"]] == list(DEFAULT_SCENARIOS)
    assert all(row["status"] == "passed" for row in first["scenarios"])


def test_full_lifecycle_covers_checkpoint_eval_stop_wandb_and_terminal() -> None:
    report = run_simulated_certification(scenarios=["full-lifecycle"])
    scenario = report["scenarios"][0]
    invariants = {row["name"] for row in scenario["invariants"]}

    assert scenario["status"] == "passed"
    assert {
        "all-one-hundred-episodes-accepted",
        "checkpoint-to-eval-causality",
        "single-wandb-writer",
        "training-and-eval-metrics-share-run",
        "delivery-high-water",
        "eval-driven-stop",
        "lowest-step-accepted-promotion",
        "scientific-terminal-receipt",
    } <= invariants
    assert scenario["evidence"]["eval_statuses"] == ["accepted", "rejected"]
    assert scenario["evidence"]["stop_reason"] == "eval_acceptance"
    assert scenario["evidence"]["final_step"] < 50_000_000


def test_parallel_scenario_interleaves_independent_launches() -> None:
    report = run_simulated_certification(scenarios=["parallel-run-isolation"])
    scenario = report["scenarios"][0]
    invariants = {row["name"] for row in scenario["invariants"]}

    assert scenario["status"] == "passed"
    assert {
        "independent-run-leases-coexist",
        "parallel-storage-prefixes-isolated",
        "parallel-wandb-writers-isolated",
        "parallel-modal-dispatches-isolated",
    } <= invariants
    assert len(scenario["evidence"]["interleaving"]) == 2


def test_early_stop_scenario_covers_failure_success_tamper_and_promotion_race() -> None:
    report = run_simulated_certification(scenarios=["early-stop-outcomes"])
    scenario = report["scenarios"][0]
    invariants = {row["name"] for row in scenario["invariants"]}

    assert scenario["status"] == "passed"
    assert {
        "failure-stop-is-designed-non-resumable-failure",
        "training-only-success-stop-succeeds",
        "evaluation-promotion-overrides-simultaneous-failure-stop",
        "early-stop-receipt-corruption-rejected",
    } <= invariants


def test_report_keeps_raw_evidence_and_replays_scenario_set(tmp_path: Path) -> None:
    artifacts = tmp_path / "first"
    report = run_simulated_certification(
        scenarios=["full-lifecycle", "wandb-retry-deduplication"],
        artifact_root=artifacts,
    )

    assert json.loads((artifacts / "report.json").read_text()) == report
    assert (artifacts / "full-lifecycle" / "evidence" / "transcript.json").is_file()
    assert (artifacts / "full-lifecycle" / "evidence" / "wandb-events.json").is_file()
    replayed = replay_simulated_certification(
        artifacts / "replay.json",
        artifact_root=tmp_path / "replayed",
    )
    assert replayed == report


def test_network_attempt_fails_with_replayable_evidence(tmp_path: Path) -> None:
    def network_probe(_root: Path):
        socket.create_connection(("example.com", 443))
        return {"invariants": [], "evidence": {}}

    SCENARIOS["network-probe"] = network_probe
    try:
        report = run_simulated_certification(
            scenarios=["network-probe"],
            artifact_root=tmp_path,
        )
    finally:
        SCENARIOS.pop("network-probe")

    assert report["status"] == "failed"
    assert report["scenarios"][0]["failure"]["type"] == "AssertionError"
    assert "network access" in report["scenarios"][0]["failure"]["message"]
    assert (tmp_path / "report.json").is_file()
    assert (tmp_path / "replay.json").is_file()


def test_experiment_certify_cli_lists_and_emits_json(capsys) -> None:
    assert experiment_main(["certify", "--list"]) == 0
    listed = capsys.readouterr().out.splitlines()
    assert listed == list(DEFAULT_SCENARIOS)

    assert (
        experiment_main(
            [
                "certify",
                "--tier",
                "simulated",
                "--scenario",
                "same-run-lease-fencing",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["failure_bundle"] is None
    assert payload["report"]["status"] == "passed"
