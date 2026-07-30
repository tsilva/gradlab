from __future__ import annotations

import os
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from gradlab.job_queue import (
    HandlerResult,
    JobStore,
    JobSubject,
    SubjectUpdate,
    WorkerStart,
    ensure_flusher,
    lock_is_held,
    register_handler,
    run_flusher,
)
from gradlab.manual_evaluation import ManualEvaluationQueue


class _SuccessfulHandler:
    job_type = "test-success"
    version = 1

    @classmethod
    def validate_payload(cls, payload):
        value = str(payload.get("value") or "")
        if not value:
            raise ValueError("value is required")
        return {"value": value}

    def advance(self, job):
        return HandlerResult(
            state="succeeded",
            subjects=(
                SubjectUpdate(
                    subject_type="test",
                    subject_id=str(job["payload"]["value"]),
                    state="succeeded",
                    detail={"result": "done"},
                ),
            ),
        )


class _RetryOnceHandler:
    job_type = "test-retry-once"
    version = 1
    order: list[str] = []

    @classmethod
    def validate_payload(cls, payload):
        return {"value": str(payload["value"])}

    def advance(self, job):
        value = str(job["payload"]["value"])
        self.order.append(value)
        if value == "first" and int(job["attempts"]) == 1:
            return HandlerResult(
                state="retry_wait",
                available_at=job["available_at"] + 0.5,
                subjects=(
                    SubjectUpdate(
                        subject_type="retry",
                        subject_id=value,
                        state="retry_wait",
                    ),
                ),
            )
        return HandlerResult(
            state="succeeded",
            subjects=(
                SubjectUpdate(
                    subject_type="retry",
                    subject_id=value,
                    state="succeeded",
                ),
            ),
        )


class JobQueueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "jobs"
        self.store = JobStore(self.root)
        register_handler(
            _SuccessfulHandler.job_type,
            _SuccessfulHandler.version,
            _SuccessfulHandler,
            replace=True,
        )
        register_handler(
            _RetryOnceHandler.job_type,
            _RetryOnceHandler.version,
            _RetryOnceHandler,
            replace=True,
        )
        _RetryOnceHandler.order = []

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def enqueue(self):
        return self.store.enqueue(
            job_type=_SuccessfulHandler.job_type,
            handler_version=_SuccessfulHandler.version,
            payload={"value": "subject-a"},
            idempotency_key="same-request",
            subjects=[
                JobSubject(
                    subject_type="test",
                    subject_id="subject-a",
                    exclusive_key="test:subject-a",
                )
            ],
        )

    def test_enqueue_is_durable_idempotent_and_private(self) -> None:
        first = self.enqueue()
        repeated = self.enqueue()

        self.assertTrue(first.created)
        self.assertFalse(repeated.created)
        self.assertEqual(first.job["job_id"], repeated.job["job_id"])
        self.assertEqual(self.store.job(first.job["job_id"])["state"], "queued")
        self.assertEqual(self.store.events(first.job["job_id"])[0]["kind"], "enqueued")
        self.assertEqual(os.stat(self.root).st_mode & 0o777, 0o700)
        self.assertEqual(os.stat(self.store.path).st_mode & 0o777, 0o600)
        self.assertEqual(os.stat(self.store.lock_path).st_mode & 0o777, 0o600)

    def test_existing_noncurrent_schema_is_rejected(self) -> None:
        self.root.mkdir(parents=True)
        with sqlite3.connect(self.store.path) as connection:
            connection.execute("PRAGMA user_version=0")

        with self.assertRaisesRegex(RuntimeError, "requires 1"):
            self.store.init()

    def test_flusher_claims_updates_subject_and_releases_leadership(self) -> None:
        enqueued = self.enqueue()

        self.assertEqual(run_flusher(self.store, idle_seconds=0), 0)

        job = self.store.job(enqueued.job["job_id"])
        self.assertEqual(job["state"], "succeeded")
        subject = self.store.subjects(enqueued.job["job_id"])[0]
        self.assertEqual(subject["state"], "succeeded")
        self.assertEqual(subject["detail"], {"result": "done"})
        self.assertFalse(lock_is_held(self.store))

    def test_stale_running_job_is_recovered_before_advancement(self) -> None:
        enqueued = self.enqueue()
        claimed = self.store.claim_next()
        self.assertEqual(claimed["state"], "running")

        self.assertEqual(run_flusher(self.store, idle_seconds=0), 0)

        job = self.store.job(enqueued.job["job_id"])
        self.assertEqual(job["state"], "succeeded")
        kinds = [event["kind"] for event in self.store.events(job["job_id"])]
        self.assertIn("recovered_after_worker_stop", kinds)

    def test_cancel_and_retry_preserve_append_only_history(self) -> None:
        enqueued = self.enqueue()
        job_id = str(enqueued.job["job_id"])

        canceled = self.store.request_cancel(job_id)
        self.assertEqual(canceled["state"], "canceled")
        retried_canceled = self.store.retry(job_id)
        self.assertEqual(retried_canceled["state"], "queued")
        self.assertEqual(self.store.subjects(job_id)[0]["state"], "queued")
        self.store.request_cancel(job_id)

        other = self.store.enqueue(
            job_type=_SuccessfulHandler.job_type,
            handler_version=_SuccessfulHandler.version,
            payload={"value": "subject-b"},
            idempotency_key="other-request",
            subjects=[
                JobSubject(subject_type="test", subject_id="subject-b")
            ],
        )
        claimed = self.store.claim_next()
        self.store.finish(
            str(claimed["job_id"]),
            HandlerResult(state="failed", message="transient"),
        )
        retried = self.store.retry(str(other.job["job_id"]))
        self.assertEqual(retried["state"], "queued")
        kinds = [
            event["kind"]
            for event in self.store.events(str(other.job["job_id"]))
        ]
        self.assertEqual(kinds[-2:], ["state:failed", "retried"])
        canceled_kinds = [
            event["kind"]
            for event in self.store.events(job_id)
        ]
        self.assertEqual(
            canceled_kinds,
            ["enqueued", "canceled", "retried", "canceled"],
        )

    def test_retry_wait_does_not_head_of_line_block_runnable_job(self) -> None:
        for value in ("first", "second"):
            self.store.enqueue(
                job_type=_RetryOnceHandler.job_type,
                handler_version=_RetryOnceHandler.version,
                payload={"value": value},
                idempotency_key=f"request-{value}",
                subjects=[
                    JobSubject(subject_type="retry", subject_id=value)
                ],
            )

        run_flusher(self.store, idle_seconds=0)

        self.assertEqual(_RetryOnceHandler.order, ["first", "second", "first"])
        states = {
            subject["subject_id"]: subject["state"]
            for job in self.store.jobs()
            for subject in self.store.subjects(str(job["job_id"]))
        }
        self.assertEqual(states, {"first": "succeeded", "second": "succeeded"})

    def test_detached_worker_acknowledges_and_observes_enqueue_during_idle(self) -> None:
        first = self.enqueue()

        started = ensure_flusher(
            self.store,
            start_timeout_seconds=3.0,
            idle_seconds=0.5,
        )

        self.assertEqual(started.state, "started")
        second = self.store.enqueue(
            job_type=_SuccessfulHandler.job_type,
            handler_version=_SuccessfulHandler.version,
            payload={"value": "subject-b"},
            idempotency_key="idle-window-request",
            subjects=[
                JobSubject(subject_type="test", subject_id="subject-b")
            ],
        )
        self.assertEqual(ensure_flusher(self.store).state, "already_running")
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            first_state = self.store.job(str(first.job["job_id"]))["state"]
            second_state = self.store.job(str(second.job["job_id"]))["state"]
            if (
                first_state in {"blocked", "failed", "succeeded"}
                and second_state in {"blocked", "failed", "succeeded"}
                and not lock_is_held(self.store)
            ):
                break
            time.sleep(0.05)
        self.assertEqual(self.store.job(str(first.job["job_id"]))["attempts"], 1)
        self.assertEqual(self.store.job(str(second.job["job_id"]))["attempts"], 1)
        self.assertFalse(lock_is_held(self.store))

    def test_checkpoint_facade_deduplicates_and_reports_start_failure(self) -> None:
        run_id = "gradlab-" + "a" * 32
        checkpoint_id = "checkpoint-100-" + "b" * 16
        context = SimpleNamespace(
            checkpoint=SimpleNamespace(checkpoint_id=checkpoint_id),
            intent=SimpleNamespace(idempotency_key="c" * 64),
        )
        planner = SimpleNamespace(
            _contexts=lambda *_args, **_kwargs: [context],
            authority=SimpleNamespace(
                evaluation=SimpleNamespace(
                    get_json_optional=lambda _key: None,
                )
            ),
        )
        facade = ManualEvaluationQueue(
            repo_root=Path.cwd(),
            store=self.store,
        )

        with (
            patch.object(facade, "_planner", return_value=planner),
            patch(
                "gradlab.manual_evaluation.ensure_flusher",
                return_value=WorkerStart(
                    "start_failed",
                    "simulated startup failure",
                ),
            ),
        ):
            first = facade.enqueue(
                run_id=run_id,
                checkpoint_ids=[checkpoint_id],
            )
            repeated = facade.enqueue(
                run_id=run_id,
                checkpoint_ids=[checkpoint_id],
            )

        self.assertEqual(first["items"][0]["state"], "flusher_unavailable")
        self.assertEqual(
            first["items"][0]["message"],
            "simulated startup failure",
        )
        self.assertEqual(
            first["items"][0]["job_id"],
            repeated["items"][0]["job_id"],
        )
        self.assertEqual(len(self.store.jobs()), 1)

    def test_checkpoint_planner_loads_operator_environment_without_launcher_import(
        self,
    ) -> None:
        facade = ManualEvaluationQueue(
            repo_root=Path.cwd(),
            store=self.store,
        )
        storage = object()
        authority = object()
        planner = object()

        with (
            patch(
                "gradlab.manual_evaluation.load_repository_operator_environment"
            ) as load_environment,
            patch(
                "gradlab.manual_evaluation.RunStorageConfig.from_env",
                return_value=storage,
            ),
            patch(
                "gradlab.manual_evaluation.RunAuthority",
                return_value=authority,
            ),
            patch(
                "gradlab.manual_evaluation.ManualEvaluationSupervisor",
                return_value=planner,
            ) as supervisor,
        ):
            result = facade._planner()

        self.assertIs(result, planner)
        load_environment.assert_called_once_with(facade.repo_root)
        supervisor.assert_called_once_with(
            authority=authority,
            repo_root=facade.repo_root,
            project_results=False,
            work_root=self.store.work_root,
        )


if __name__ == "__main__":
    unittest.main()
