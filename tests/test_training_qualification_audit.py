import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gradlab.experiment_cli import _audit_training_run, build_parser, cmd_status
from gradlab.metric_names import ORCHESTRATION_EVENT_SEQUENCE
from gradlab.mlflow_delivery import RUN_ID_TAG
from gradlab.run_contracts import new_attempt_id, new_run_id
from tests import test_run_authority as authority_fixtures
from tests.test_policy_bundle import level1_1_recipe_document


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
        environment_sha256=document["recipe"]["environment_hash"].removeprefix("sha256:"),
        goal_variant=document["recipe"]["goal_variant"],
        goal_sha256=config["effective_goal_contract_sha256"],
        tracking={**config["tracking"], "auth_mode": "network"},
        wandb={},
        compute={**manifest.compute, "checkpoint_eval_backend": "none"},
        modal={**manifest.modal, "enabled": False},
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
    key = case.prefix + f"0001-0002-{digest}.jsonl"
    case.authority.control.put_bytes(key, payload)
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
    assert not result["run_mutations_performed"] and not result["acceptance_claimed"]
    assert sorted(audit_case.authority.control.iter_keys("")) == before


@pytest.mark.parametrize(
    "damage",
    [
        "hash",
        "gap",
        "watermark",
        "service_identity",
        "metric_history",
        "model",
        "checkpoint_identity",
        "counts",
        "warmup",
        "resumed",
        "source",
        "active",
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
    key = seal(c)
    if damage == "hash":
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
