import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gradlab.experiment_cli import _audit_training_run, build_parser, cmd_status
from gradlab.metric_names import ORCHESTRATION_EVENT_SEQUENCE
from gradlab.mlflow_delivery import RUN_ID_TAG
from gradlab.env_identity import environment_hash, environment_identity_from_train_config
from gradlab.run_contracts import new_attempt_id, new_run_id
from tests import test_run_authority as authority_fixtures
from tests.test_policy_bundle import MARIO_ASSET, level1_1_recipe_document


@pytest.fixture
def audit_case():
    fixture = authority_fixtures.RunAuthorityTests()
    fixture.setUp()
    authority = fixture.authority
    document = level1_1_recipe_document(seed=12)
    recipe_sha = authority.put_recipe_document(document)
    config = document["recipe"]["train_config"]
    manifest = fixture.manifest(new_run_id(), new_attempt_id())
    manifest = replace(
        manifest,
        seed=12,
        source_sha=document["provenance"]["source_commit"],
        image_digest=document["provenance"]["runtime"]["image_ref"],
        recipe_sha256=recipe_sha,
        environment_sha256=environment_hash(
            environment_identity_from_train_config(config)
        ).removeprefix("sha256:"),
        goal_variant=document["recipe"]["goal_variant"],
        goal_sha256=config["effective_goal_contract_sha256"],
        tracking={**config["tracking"], "auth_mode": "network"},
        wandb={},
        compute={**manifest.compute, "checkpoint_eval_backend": "none"},
        modal={**manifest.modal, "enabled": False, "rom_asset_manifest": dict(MARIO_ASSET)},
    )
    events = [
        {
            "event_seq": i,
            "event_id": f"event-{i}",
            "kind": "history",
            "source": "train",
            "step": i * 1000,
            "created_at": float(i),
            "payload": {
                "train/success/min": fraction,
                "train/episodes/count": float(count),
                "train/success/Level1-1/count": float(clears),
            },
        }
        for i, (fraction, count, clears) in enumerate([(0.8, 200, 100), (0.95, 400, 280)], 1)
    ]
    prefix = f"runs/{manifest.run_id}/attempts/{manifest.attempt_id}/metric-segments/"
    checks = []
    for i in (1, 2):
        body = f"synthetic model {i}".encode()
        digest = hashlib.sha256(body).hexdigest()
        key = f"runs/{manifest.run_id}/checkpoints/{i * 1000}-{digest}/model.zip"
        authority.models.put_bytes(key, body)
        checks.append(
            {
                "run_id": manifest.run_id,
                "checkpoint_id": f"checkpoint-{i * 1000}-{digest[:16]}",
                "step": i * 1000,
                "sha256": digest,
                "size_bytes": len(body),
                "public_url": authority.models.public_url(key),
                "recipe_sha256": recipe_sha,
                "goal_sha256": manifest.goal_sha256,
                "environment_sha256": manifest.environment_sha256,
            }
        )
    authority.control.put_json(
        f"runs/{manifest.run_id}/metrics-binding.json",
        {
            "run_id": manifest.run_id,
            "backend": "mlflow",
            "service_run_id": "service-fixture",
            "service_url": "https://metrics.example.test/run/service-fixture",
        },
    )
    remote = SimpleNamespace(
        data=SimpleNamespace(
            tags={RUN_ID_TAG: manifest.run_id},
            metrics={ORCHESTRATION_EVENT_SEQUENCE: 2.0},
        )
    )
    client = Mock(get_run=Mock(return_value=remote))
    client.get_metric_history.side_effect = lambda run, key: [
        SimpleNamespace(step=e["step"], value=e["payload"][key])
        for e in events
        if key in e["payload"]
    ]
    status = {
        "completed": True,
        "dstack": {"terminal": True},
        "semantic": {"manifest": manifest.to_dict()},
        "attempt_terminal": {
            "attempt_id": manifest.attempt_id,
            "checkpoint_inventory": checks,
            "state": "stopped",
            "stop_reason": "early_stop_neutral:qualified_plateau",
            "final_step": 2000,
            "drain": {
                "complete": True,
                "metric_segment_high_water": 2,
                "service_remote_high_water_mark": 2,
            },
        },
    }
    case = SimpleNamespace(
        authority=authority,
        manifest=manifest,
        events=events,
        status=status,
        client=client,
        remote=remote,
        prefix=prefix,
    )
    try:
        yield case
    finally:
        fixture.tearDown()


def seal(case):
    payload = b"".join(json.dumps(event, sort_keys=True).encode() + b"\n" for event in case.events)
    digest = hashlib.sha256(payload).hexdigest()
    key = (
        case.prefix
        + f"{case.events[0]['event_seq']:04}-{case.events[-1]['event_seq']:04}-{digest}.jsonl"
    )
    case.authority.control.put_bytes(key, payload)
    keys = sorted(x["key"] for x in case.authority.control.iter_objects(case.prefix))
    case.status["attempt_terminal"]["drain"]["journal_archive"] = {
        "retention": "durable",
        "prefix": f"runs/{case.manifest.run_id}/attempts/",
        "segment_count": len(keys),
        "inventory_sha256": hashlib.sha256("\n".join(keys).encode()).hexdigest(),
    }
    return key


def audit(case):
    return _audit_training_run(case.authority, case.status, mlflow_client=case.client)


def test_completed_window_model_and_service_are_independently_verified_without_mutation(audit_case):
    seal(audit_case)
    before = sorted(audit_case.authority.control.iter_keys(""))
    result = audit(audit_case)
    assert result["qualified"] and result["remote_model_sha256_verified"]
    assert result["observation"] == {
        "metric_step": 2000,
        "window_episodes": 100,
        "window_clears": 95,
        "cumulative_target_episodes": 400,
        "cumulative_all_episodes": 400,
        "cumulative_clears": 280,
    }
    assert result["journal_events_verified"] == result["mlflow_remote_high_water"] == 2
    document = audit_case.authority.recipe_document(audit_case.manifest.recipe_sha256)
    assert result["declared_environment_sha256"] != result["artifact_environment_sha256"]
    assert result["declared_environment_sha256"] == audit_case.manifest.environment_sha256
    assert result["artifact_environment_sha256"] == document["recipe"][
        "environment_hash"
    ].removeprefix("sha256:")
    assert result["environment_materialization_verified"] and result["rom_asset_identity_verified"]
    assert not result["run_mutations_performed"] and not result["acceptance_claimed"]
    assert sorted(audit_case.authority.control.iter_keys("")) == before


@pytest.mark.parametrize(
    "damage",
    [
        "hash",
        "gap",
        "watermark",
        "service_identity",
        "inventory",
        "metric_history",
        "model",
        "checkpoint_identity",
        "counts",
        "warmup",
        "resumed",
        "source",
        "active",
        "declared_environment",
        "artifact_hash_as_declared",
        "rom_asset",
    ],
)
def test_incomplete_or_conflicting_evidence_cannot_qualify(audit_case, damage):
    c = audit_case
    if damage == "gap":
        c.events[1]["event_seq"] = 3
    elif damage == "watermark":
        c.remote.data.metrics[ORCHESTRATION_EVENT_SEQUENCE] = 1
    elif damage == "service_identity":
        c.remote.data.tags[RUN_ID_TAG] = "another-run"
    elif damage == "metric_history":
        c.client.get_metric_history.side_effect = lambda *args: []
    elif damage == "checkpoint_identity":
        c.status["attempt_terminal"]["checkpoint_inventory"][1]["recipe_sha256"] = "f" * 64
    elif damage == "counts":
        del c.events[1]["payload"]["train/episodes/count"]
    elif damage == "warmup":
        c.events[1]["payload"]["train/episodes/count"] = 99
    elif damage == "resumed":
        c.status["semantic"]["manifest"]["compute"]["resume_checkpoint"] = {"step": 100}
    elif damage == "source":
        c.status["semantic"]["manifest"]["source_sha"] = "f" * 40
    elif damage == "active":
        c.status["completed"] = False
    elif damage == "declared_environment":
        c.status["semantic"]["manifest"]["environment_sha256"] = "f" * 64
    elif damage == "artifact_hash_as_declared":
        document = c.authority.recipe_document(c.manifest.recipe_sha256)
        c.status["semantic"]["manifest"]["environment_sha256"] = document["recipe"][
            "environment_hash"
        ].removeprefix("sha256:")
    elif damage == "rom_asset":
        c.status["semantic"]["manifest"]["modal"]["rom_asset_manifest"]["sha256"] = "f" * 64
    key = seal(c)
    if damage == "inventory":
        c.status["attempt_terminal"]["drain"]["journal_archive"]["inventory_sha256"] = "0" * 64
    elif damage == "hash":
        body = c.authority.control.get_bytes(key).replace(b"0.95", b"0.94")
        c.authority.control.put_bytes(key, body, create_only=False)
    elif damage == "model":
        checkpoint = c.status["attempt_terminal"]["checkpoint_inventory"][1]
        model_key = f"runs/{c.manifest.run_id}/checkpoints/2000-{checkpoint['sha256']}/model.zip"
        c.authority.models.put_bytes(model_key, b"x" * checkpoint["size_bytes"], create_only=False)
    with pytest.raises((ValueError, RuntimeError)):
        audit(c)


def test_unretained_history_peak_is_not_assigned_to_another_checkpoint(audit_case):
    audit_case.events[1]["step"] = 1500
    seal(audit_case)
    result = audit(audit_case)
    assert result["best_observed_window_clears"] == 95
    assert result["observation"]["window_clears"] == 80
    assert result["selected_checkpoint"]["step"] == 1000


def add_delivery_recovery(case):
    original = deepcopy(case.status["attempt_terminal"])
    original.update(state="resumable_failure", stop_reason="supervisor_failure")
    original["drain"]["complete"] = False
    checkpoint = {
        **case.status["attempt_terminal"]["checkpoint_inventory"][-1],
        "schema_version": 2,
        "purpose": "final",
        "model_document_url": "https://models.example.test/model.json",
        "model_document_sha256": "a" * 64,
        "recipe_document_url": "https://models.example.test/recipe.json",
        "recipe_document_sha256": case.manifest.recipe_sha256,
        "evaluation_contract_sha256": "b" * 64,
        "recovery_sidecar_key": "runs/fixture/recovery-sidecar.json",
        "created_at": case.manifest.created_at,
    }
    recovered = replace(
        case.manifest,
        attempt_id=new_attempt_id(),
        compute={
            **case.manifest.compute,
            "recovery_mode": "drain-only",
            "resume_checkpoint": checkpoint,
        },
    )
    terminal = deepcopy(case.status["attempt_terminal"])
    terminal.update(
        attempt_id=recovered.attempt_id, state="succeeded", stop_reason="training_cap_complete"
    )
    case.status["semantic"].update(
        attempts=[case.manifest.to_dict(), recovered.to_dict()],
        attempt_terminals=[original, terminal],
    )
    case.status["attempt_terminal"] = terminal


def test_delivery_only_recovery_preserves_fresh_qualification_and_original_stop(audit_case):
    add_delivery_recovery(audit_case)
    audit_case.events.append(
        {
            "event_seq": 3,
            "event_id": "delivery-progress",
            "kind": "history",
            "source": "supervisor",
            "step": 3,
            "created_at": 3.0,
            "payload": {ORCHESTRATION_EVENT_SEQUENCE: 3.0},
        }
    )
    audit_case.status["attempt_terminal"]["drain"].update(
        metric_segment_high_water=3,
        service_remote_high_water_mark=3,
    )
    audit_case.remote.data.metrics[ORCHESTRATION_EVENT_SEQUENCE] = 3.0
    seal(audit_case)
    result = audit(audit_case)
    assert result["qualified"] and result["remote_model_sha256_verified"]
    assert result["observation"]["window_clears"] == 95
    assert result["attempt_id"] == audit_case.manifest.attempt_id
    assert result["delivery_attempt_id"] != result["attempt_id"]
    assert result["drain_recovery_attempts"] == 1
    assert result["stop_reason"] == "supervisor_failure"
    assert result["delivery_stop_reason"] == "training_cap_complete"


@pytest.mark.parametrize(
    "damage",
    [
        "training_resume",
        "seed",
        "source",
        "recipe",
        "missing_receipt",
        "evaluation",
        "boundary",
        "checkpoint_boundary",
    ],
)
def test_delivery_recovery_requires_unchanged_contract_and_complete_history(audit_case, damage):
    add_delivery_recovery(audit_case)
    recovery = audit_case.status["semantic"]["attempts"][1]
    if damage == "training_resume":
        recovery["compute"]["recovery_mode"] = "resume-training"
    elif damage in {"seed", "source", "recipe"}:
        key = {"seed": "seed", "source": "source_sha", "recipe": "recipe_sha256"}[damage]
        recovery[key] = 99 if damage == "seed" else "f" * (40 if damage == "source" else 64)
    elif damage == "missing_receipt":
        audit_case.status["semantic"]["attempt_terminals"].pop(0)
    elif damage == "evaluation":
        recovery["compute"]["checkpoint_eval_backend"] = "training-container"
    elif damage == "boundary":
        audit_case.status["attempt_terminal"]["final_step"] += 1
    elif damage == "checkpoint_boundary":
        audit_case.status["attempt_terminal"]["checkpoint_inventory"][-1]["step"] += 1
    seal(audit_case)
    with pytest.raises(ValueError):
        audit(audit_case)


def test_recovery_cannot_hide_new_training_at_an_old_transition_step(audit_case):
    add_delivery_recovery(audit_case)
    audit_case.events.append(
        {
            **deepcopy(audit_case.events[-1]),
            "event_seq": 3,
            "event_id": "new-learning",
            "step": 1000,
        }
    )
    terminal = audit_case.status["attempt_terminal"]
    terminal["drain"]["metric_segment_high_water"] = 3
    terminal["drain"]["service_remote_high_water_mark"] = 3
    audit_case.remote.data.metrics[ORCHESTRATION_EVENT_SEQUENCE] = 3
    seal(audit_case)
    with pytest.raises(ValueError, match="new training history"):
        audit(audit_case)


def test_service_failure_reports_stage_without_private_exception_text(audit_case):
    seal(audit_case)
    audit_case.client.get_run.side_effect = RuntimeError("private-service-fixture-detail")
    with pytest.raises(ValueError) as failure:
        audit(audit_case)
    assert str(failure.value) == "MLflow Run evidence unavailable or invalid (RuntimeError)"
    assert "private-service-fixture-detail" not in str(failure.value)


def test_audit_completes_with_one_shared_gateway_slot_available(audit_case, monkeypatch):
    import threading
    import time

    seal(audit_case)
    slot = threading.BoundedSemaphore(1)
    original = audit_case.authority.control.get_bytes

    def get_bytes(key):
        if "/metric-segments/" not in key:
            return original(key)
        if not slot.acquire(blocking=False):
            raise RuntimeError("shared gateway capacity exhausted")
        try:
            time.sleep(0.01)
            return original(key)
        finally:
            slot.release()

    monkeypatch.setattr(audit_case.authority.control, "get_bytes", get_bytes)
    assert audit(audit_case)["qualified"]


def test_return_or_incomplete_success_does_not_substitute_for_full_window(audit_case):
    for event in audit_case.events:
        event["payload"] = {"train/return/mean": 1e8}
    seal(audit_case)
    result = audit(audit_case)
    assert not result["qualified"] and "selected_checkpoint" not in result


def test_cli_returns_safe_domain_failure_without_claiming_a_completed_audit(monkeypatch, capsys):
    from gradlab import experiment_cli

    run_id = new_run_id()
    monkeypatch.setattr(experiment_cli, "repository_root", lambda: None)
    monkeypatch.setattr(experiment_cli, "_status", lambda *args: {"completed": False})
    monkeypatch.setattr(experiment_cli, "_storage", lambda *args: (None, None))
    args = build_parser().parse_args(["status", "--run", run_id, "--audit-training", "--json"])
    assert cmd_status(args) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["audit_complete"] is False and result["qualified"] is False
    assert "completed, released" in result["reason"]


def test_cli_unexpected_failure_withholds_private_provider_details(monkeypatch, capsys):
    from gradlab import experiment_cli

    run_id = new_run_id()
    monkeypatch.setattr(experiment_cli, "repository_root", lambda: None)
    monkeypatch.setattr(experiment_cli, "_status", lambda *args: {"completed": True})
    monkeypatch.setattr(experiment_cli, "_storage", lambda *args: (None, None))

    def fail(*args):
        raise KeyError("private-provider-fixture-detail")

    monkeypatch.setattr(experiment_cli, "_audit_training_run", fail)
    args = build_parser().parse_args(["status", "--run", run_id, "--audit-training", "--json"])
    assert cmd_status(args) == 1
    text = capsys.readouterr().out
    result = json.loads(text)
    assert not result["audit_complete"] and not result["qualified"]
    assert "KeyError" in result["reason"] and "private-provider-fixture-detail" not in text
