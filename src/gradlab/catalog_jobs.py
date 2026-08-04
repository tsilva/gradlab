from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gradlab.job_queue import (
    HandlerResult,
    JobStore,
    JobSubject,
    SubjectUpdate,
    ensure_flusher,
    register_handler,
)
from gradlab.json_utils import canonical_json_text


CATALOG_PROJECT_JOB_TYPE = "catalog-project"
CATALOG_PROJECT_JOB_VERSION = 1


class CatalogProjectJobHandler:
    job_type = CATALOG_PROJECT_JOB_TYPE
    version = CATALOG_PROJECT_JOB_VERSION

    @classmethod
    def validate_payload(cls, payload: Mapping[str, Any]) -> dict[str, Any]:
        repo_root = Path(str(payload.get("repo_root") or "")).expanduser().resolve()
        queue_root = Path(str(payload.get("queue_root") or "")).expanduser().resolve()
        goal_slug = str(payload.get("goal_slug") or "").strip()
        request_id = str(payload.get("request_id") or "").strip()
        if not repo_root.is_dir() or not queue_root.is_absolute() or not goal_slug or not request_id:
            raise ValueError("catalog projection job payload is malformed")
        return {
            "repo_root": str(repo_root),
            "queue_root": str(queue_root),
            "goal_slug": goal_slug,
            "request_id": request_id,
        }

    def advance(self, job: Mapping[str, Any]) -> HandlerResult:
        from gradlab.operator_environment import load_repository_operator_environment
        from gradlab.r2_store import RunStorageConfig
        from gradlab.run_authority import RunAuthority

        payload = self.validate_payload(job.get("payload") or {})
        load_repository_operator_environment(Path(payload["repo_root"]))
        try:
            result = RunAuthority(
                RunStorageConfig.from_env()
            )._goal_catalog_projector().reconcile(payload["goal_slug"])
        except (KeyError, TypeError, ValueError):
            raise
        except Exception as exc:
            delay = min(2 ** max(0, int(job.get("attempts") or 1) - 1), 60)
            return HandlerResult(
                state="retry_wait",
                available_at=time.time() + delay,
                message=f"{type(exc).__name__}: {exc}",
                subjects=(
                    SubjectUpdate(
                        subject_type="goal-catalog",
                        subject_id=payload["goal_slug"],
                        state="retry_wait",
                        detail={"retry_in_seconds": delay},
                    ),
                ),
            )
        return HandlerResult(
            state="succeeded",
            message=(
                f"projected {result.applied_event_count} events"
                + (
                    f"; {len(result.orphan_events)} orphan events require inspection"
                    if result.orphan_events
                    else ""
                )
            ),
            subjects=(
                SubjectUpdate(
                    subject_type="goal-catalog",
                    subject_id=payload["goal_slug"],
                    state="succeeded",
                    detail=result.to_dict(),
                ),
            ),
        )


def register_job_handler() -> None:
    register_handler(
        CATALOG_PROJECT_JOB_TYPE,
        CATALOG_PROJECT_JOB_VERSION,
        CatalogProjectJobHandler,
        replace=True,
    )


def enqueue_catalog_projection(
    *,
    repo_root: Path,
    goal_slug: str,
    request_id: str,
    store: JobStore | None = None,
) -> dict[str, Any]:
    queue = store or JobStore()
    queue.init()
    payload = {
        "repo_root": str(Path(repo_root).resolve()),
        "queue_root": str(queue.root),
        "goal_slug": str(goal_slug),
        "request_id": str(request_id),
    }
    identity = hashlib.sha256(
        canonical_json_text(
            {
                "job_type": CATALOG_PROJECT_JOB_TYPE,
                "goal_slug": goal_slug,
                "request_id": request_id,
            },
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    admitted = queue.enqueue(
        job_type=CATALOG_PROJECT_JOB_TYPE,
        handler_version=CATALOG_PROJECT_JOB_VERSION,
        payload=payload,
        idempotency_key=identity,
        subjects=[
            JobSubject(
                subject_type="goal-catalog",
                subject_id=str(goal_slug),
                detail={"request_id": str(request_id)},
            )
        ],
    )
    worker = ensure_flusher(queue, start_timeout_seconds=1.0)
    return {
        "job": admitted.job,
        "created": admitted.created,
        "worker": worker.to_dict(),
    }


__all__ = [
    "CATALOG_PROJECT_JOB_TYPE",
    "CATALOG_PROJECT_JOB_VERSION",
    "CatalogProjectJobHandler",
    "enqueue_catalog_projection",
    "register_job_handler",
]
