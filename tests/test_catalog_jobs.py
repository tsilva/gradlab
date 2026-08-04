from __future__ import annotations

from pathlib import Path

import pytest

from gradlab.catalog_jobs import CatalogProjectJobHandler, enqueue_catalog_projection
from gradlab.goal_catalog_projection import CatalogReconcileResult
from gradlab.job_queue import JobStore, WorkerStart


def _job(tmp_path: Path, *, attempts: int = 1) -> dict:
    return {
        "attempts": attempts,
        "payload": {
            "repo_root": str(tmp_path),
            "queue_root": str((tmp_path / "jobs").resolve()),
            "goal_slug": "Mario/Level1-1",
            "request_id": "event-1",
        },
    }


def test_catalog_projection_job_retries_transient_storage_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingProjector:
        @staticmethod
        def reconcile(_goal_slug: str):
            raise TimeoutError("R2 is temporarily unavailable")

    class FakeAuthority:
        def __init__(self, _storage) -> None:
            pass

        @staticmethod
        def _goal_catalog_projector():
            return FailingProjector()

    monkeypatch.setattr(
        "gradlab.operator_environment.load_repository_operator_environment",
        lambda _root: None,
    )
    monkeypatch.setattr(
        "gradlab.r2_store.RunStorageConfig.from_env",
        classmethod(lambda _cls: object()),
    )
    monkeypatch.setattr("gradlab.run_authority.RunAuthority", FakeAuthority)
    monkeypatch.setattr("gradlab.catalog_jobs.time.time", lambda: 100.0)

    result = CatalogProjectJobHandler().advance(_job(tmp_path, attempts=3))

    assert result.state == "retry_wait"
    assert result.available_at == 104.0
    assert result.subjects[0].state == "retry_wait"
    assert result.subjects[0].detail == {"retry_in_seconds": 4}


def test_catalog_projection_job_reports_a_verified_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = CatalogReconcileResult(
        goal_slug="Mario/Level1-1",
        published=True,
        generation_sha256="a" * 64,
        applied_event_count=2,
        acknowledged_event_count=2,
        orphan_events=(),
        pointer={"generation_sha256": "a" * 64},
    )

    class SuccessfulProjector:
        @staticmethod
        def reconcile(_goal_slug: str):
            return result

    class FakeAuthority:
        def __init__(self, _storage) -> None:
            pass

        @staticmethod
        def _goal_catalog_projector():
            return SuccessfulProjector()

    monkeypatch.setattr(
        "gradlab.operator_environment.load_repository_operator_environment",
        lambda _root: None,
    )
    monkeypatch.setattr(
        "gradlab.r2_store.RunStorageConfig.from_env",
        classmethod(lambda _cls: object()),
    )
    monkeypatch.setattr("gradlab.run_authority.RunAuthority", FakeAuthority)

    handled = CatalogProjectJobHandler().advance(_job(tmp_path))

    assert handled.state == "succeeded"
    assert handled.message == "projected 2 events"
    assert handled.subjects[0].detail == result.to_dict()


def test_catalog_projection_enqueue_is_generation_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JobStore(tmp_path / "jobs")
    monkeypatch.setattr(
        "gradlab.catalog_jobs.ensure_flusher",
        lambda *_args, **_kwargs: WorkerStart("already_running"),
    )

    first = enqueue_catalog_projection(
        repo_root=tmp_path,
        goal_slug="Mario/Level1-1",
        request_id="browse-generation-a",
        store=store,
    )
    repeated = enqueue_catalog_projection(
        repo_root=tmp_path,
        goal_slug="Mario/Level1-1",
        request_id="browse-generation-a",
        store=store,
    )

    assert first["created"] is True
    assert repeated["created"] is False
    assert first["job"]["job_id"] == repeated["job"]["job_id"]
    subject = store.subjects(str(first["job"]["job_id"]))[0]
    assert subject["subject_type"] == "goal-catalog"
    assert subject["subject_id"] == "Mario/Level1-1"
