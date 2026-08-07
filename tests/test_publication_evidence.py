from __future__ import annotations

from types import SimpleNamespace

import pytest

from gradlab.publication_evidence import (
    EVALUATION_EQUIVALENCE_FIELDS,
    validate_publication_evidence,
)


def test_publication_evidence_uses_one_structured_record_per_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    episodes = [
        {
            "start_state": "Start-A",
            "return": 1.0,
            "outcome": "success",
            "events": ["goal"],
            "terminated": True,
            "truncated": False,
        },
        {
            "start_state": "Start-A",
            "return": -1.0,
            "outcome": "failure",
            "events": ["life_loss", "wall"],
            "terminated": True,
            "truncated": False,
        },
    ]
    expected_contract = {field: f"value-{field}" for field in EVALUATION_EQUIVALENCE_FIELDS}
    expected_contract["episodes"] = 2
    actual_contract = {
        **expected_contract,
        "checkpoint_sha256": "a" * 64,
        "recipe_sha256": "b" * 64,
    }
    normalized = {
        "status": "accepted",
        "episode_results": episodes,
        "aggregates": {
            "eval/full/episode/return/shaped/mean": 0.0,
            "eval/full/outcome/success/starts/rate/min": 0.5,
        },
        "duration_seconds": 1.0,
        "evidence_sha256": ["evidence"],
        "error": None,
    }
    intent = SimpleNamespace(
        run_id="gradlab-" + "1" * 32,
        checkpoint_id="checkpoint-10-" + "2" * 16,
        idempotency_key="idempotency-key",
        checkpoint_sha256="a" * 64,
        execution_contract=actual_contract,
        evaluation_contract_sha256="c" * 64,
        validate=lambda: None,
    )
    verified = SimpleNamespace(
        run_id=intent.run_id,
        checkpoint_id=intent.checkpoint_id,
        idempotency_key=intent.idempotency_key,
        status="accepted",
        episode_results=episodes,
        aggregates=normalized["aggregates"],
        evidence_sha256=normalized["evidence_sha256"],
        error=None,
        validate=lambda: None,
    )
    checkpoint = SimpleNamespace(
        run_id=intent.run_id,
        checkpoint_id=intent.checkpoint_id,
        sha256="a" * 64,
        recipe_document_sha256="b" * 64,
        evaluation_contract_sha256="d" * 64,
        step=10,
        public_url="https://models.example/model.zip",
        validate=lambda: None,
    )
    monkeypatch.setattr(
        "gradlab.publication_evidence.EvalIntent.from_dict", lambda _value: intent
    )
    monkeypatch.setattr(
        "gradlab.publication_evidence.EvalResult.from_dict", lambda _value: verified
    )
    monkeypatch.setattr(
        "gradlab.publication_evidence.CheckpointManifest.from_dict",
        lambda _value: checkpoint,
    )
    monkeypatch.setattr(
        "gradlab.publication_evidence.evaluation_contract",
        lambda _recipe: expected_contract,
    )
    monkeypatch.setattr(
        "gradlab.publication_evidence.evaluation_contract_sha256",
        lambda _recipe: "c" * 64,
    )
    monkeypatch.setattr(
        "gradlab.publication_evidence.checkpoint_manifest_contract_sha256",
        lambda _recipe: "d" * 64,
    )
    monkeypatch.setattr(
        "gradlab.publication_evidence.normalize_attempt_result",
        lambda *_args, **_kwargs: normalized,
    )

    result = validate_publication_evidence(
        {
            "recipe": {"format_version": 1},
            "intent": {"document": "intent"},
            "raw_result": {"attempt_id": "attempt-1"},
            "verified_result": {"document": "verified"},
            "checkpoint_manifest": {"document": "checkpoint"},
        }
    )

    assert result["evaluation"]["by_start"] == [
        {
            "start_id": "Start-A",
            "episode_count": 2,
            "success_count": 1,
            "success_rate": 0.5,
            "shaped_return_mean": 0.0,
            "failure_reasons": {"life_loss": 1, "wall": 1},
        }
    ]
