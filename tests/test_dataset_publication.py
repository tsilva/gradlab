"""Shared publication failure modes through both concrete queue handlers."""

from importlib import import_module

import pytest

from gradlab.job_queue import JobSubject


@pytest.fixture(params=["split", "migration"])
def publication(request, tmp_path, monkeypatch):
    kind = request.param
    fixtures = import_module(f"tests.test_dataset_{kind}_publication")
    module = import_module(f"gradlab.dataset_{kind}_publication")
    handler = getattr(module, f"{kind.title()}PublicationHandler")()
    queue, payload, hub = fixtures.prepare(tmp_path, monkeypatch)
    return handler, queue, payload, hub


def test_repeated_publication_reuses_receipt_and_rejects_conflicts(publication):
    handler, queue, payload, hub = publication
    job = queue.enqueue(
        job_type=handler.job_type,
        handler_version=handler.version,
        payload=payload,
        idempotency_key="receipt",
        subjects=[JobSubject("dataset", payload["repo"])],
    ).job
    revision = handler.publish(payload, job["job_id"])
    hub.uploaded.clear()
    assert handler.publish(payload, job["job_id"]) == revision
    assert hub.uploaded == {}
    hub.files[payload["receipt"]] = b"conflicting immutable receipt"
    with pytest.raises(ValueError, match="Conflicting immutable"):
        handler.publish(payload, job["job_id"])
    assert hub.head == revision
    assert hub.uploaded == {}


def test_cancellation_prevents_upload_and_commit(publication):
    handler, queue, payload, hub = publication
    head = hub.head
    job = queue.enqueue(
        job_type=handler.job_type,
        handler_version=handler.version,
        payload=payload,
        idempotency_key="cancel",
        subjects=[JobSubject("dataset", payload["repo"])],
    ).job
    queue.request_cancel(job["job_id"])
    with pytest.raises(ValueError, match="publication canceled"):
        handler.publish(payload, job["job_id"])
    assert hub.uploaded == {}
    assert hub.head == head


@pytest.mark.parametrize("attempts, state", [(1, "retry_wait"), (8, "blocked")])
def test_remote_failure_has_bounded_retries(publication, monkeypatch, attempts, state):
    handler, _, payload, hub = publication

    def unavailable(**kwargs):
        raise RuntimeError("remote unavailable")

    monkeypatch.setattr(hub, "repo_info", unavailable)
    result = handler.advance({"payload": payload, "job_id": "unused", "attempts": attempts})
    assert result.state == state
    assert result.message == "RuntimeError: remote unavailable"
    assert hub.uploaded == {}
