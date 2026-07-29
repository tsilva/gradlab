from __future__ import annotations

import fcntl
import json
import os
import secrets
import sqlite3
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from gradlab.clock import Clock, SystemClock
from gradlab.metric_store import SqliteStore


JOB_SCHEMA_VERSION = 1
DEFAULT_IDLE_SECONDS = 30.0
DEFAULT_POLL_SECONDS = 0.5
DEFAULT_START_TIMEOUT_SECONDS = 5.0

ACTIVE_JOB_STATES = frozenset({"queued", "running", "retry_wait"})
TERMINAL_JOB_STATES = frozenset({"blocked", "succeeded", "failed", "canceled"})
JOB_STATES = ACTIVE_JOB_STATES | TERMINAL_JOB_STATES
ACTIVE_SUBJECT_STATES = frozenset(
    {
        "queued",
        "running",
        "retry_wait",
        "waiting_for_training_terminal",
        "waiting_for_run_lease",
        "submitted",
        "submission_uncertain",
        "awaiting_projection",
    }
)
TERMINAL_SUBJECT_STATES = frozenset(
    {
        "succeeded",
        "accepted",
        "rejected",
        "blocked",
        "failed",
        "expired",
        "canceled",
    }
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY,
  job_type TEXT NOT NULL,
  handler_version INTEGER NOT NULL,
  payload_json TEXT NOT NULL,
  idempotency_key TEXT NOT NULL,
  state TEXT NOT NULL,
  available_at REAL NOT NULL,
  attempts INTEGER NOT NULL DEFAULT 0,
  cancel_requested INTEGER NOT NULL DEFAULT 0,
  last_error TEXT,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  started_at REAL,
  finished_at REAL,
  UNIQUE (job_type, handler_version, idempotency_key)
);

CREATE INDEX IF NOT EXISTS jobs_runnable_idx
  ON jobs (state, available_at, created_at, job_id);

CREATE TABLE IF NOT EXISTS job_subjects (
  job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE RESTRICT,
  subject_type TEXT NOT NULL,
  subject_id TEXT NOT NULL,
  exclusive_key TEXT UNIQUE,
  state TEXT NOT NULL,
  detail_json TEXT,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  PRIMARY KEY (job_id, subject_type, subject_id)
);

CREATE INDEX IF NOT EXISTS job_subjects_lookup_idx
  ON job_subjects (subject_type, subject_id, updated_at);

CREATE TABLE IF NOT EXISTS job_events (
  event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE RESTRICT,
  kind TEXT NOT NULL,
  detail_json TEXT,
  created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS job_events_job_idx
  ON job_events (job_id, event_seq);

CREATE TABLE IF NOT EXISTS worker_state (
  singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
  generation TEXT,
  pid INTEGER,
  heartbeat_at REAL,
  executable TEXT,
  worker_version INTEGER,
  last_start_error TEXT,
  updated_at REAL NOT NULL
);
"""


def default_queue_root() -> Path:
    override = os.environ.get("GRADLAB_JOB_QUEUE_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".gradlab" / "jobs" / "v1"


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _job_id() -> str:
    return f"job-{secrets.token_hex(16)}"


@dataclass(frozen=True)
class JobSubject:
    subject_type: str
    subject_id: str
    exclusive_key: str | None = None
    detail: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class SubjectUpdate:
    subject_type: str
    subject_id: str
    state: str
    detail: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class HandlerResult:
    state: Literal["retry_wait", "blocked", "succeeded", "failed", "canceled"]
    message: str | None = None
    available_at: float | None = None
    subjects: tuple[SubjectUpdate, ...] = ()


@dataclass(frozen=True)
class EnqueueResult:
    job: Mapping[str, Any] | None
    subjects: tuple[Mapping[str, Any], ...]
    created: bool


@dataclass(frozen=True)
class WorkerStart:
    state: Literal["started", "already_running", "start_failed"]
    message: str | None = None
    pid: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "message": self.message,
            "pid": self.pid,
        }


class JobHandler(Protocol):
    job_type: str
    version: int

    @classmethod
    def validate_payload(cls, payload: Mapping[str, Any]) -> dict[str, Any]: ...

    def advance(self, job: Mapping[str, Any]) -> HandlerResult: ...


HandlerFactory = Callable[[], JobHandler]
_HANDLERS: dict[tuple[str, int], HandlerFactory] = {}


def register_handler(
    job_type: str,
    version: int,
    factory: HandlerFactory,
    *,
    replace: bool = False,
) -> None:
    identity = (str(job_type), int(version))
    if identity in _HANDLERS and not replace:
        raise ValueError(f"job handler is already registered: {identity[0]} v{identity[1]}")
    _HANDLERS[identity] = factory


def load_builtin_handlers() -> None:
    # Closed, explicit imports prevent queue rows from selecting executable code.
    from gradlab import manual_evaluation

    manual_evaluation.register_job_handler()


def handler_for(job_type: str, version: int) -> JobHandler:
    factory = _HANDLERS.get((str(job_type), int(version)))
    if factory is None:
        load_builtin_handlers()
        factory = _HANDLERS.get((str(job_type), int(version)))
    if factory is None:
        raise ValueError(f"unsupported job handler: {job_type} v{version}")
    return factory()


class JobStore(SqliteStore):
    def __init__(
        self,
        root: Path | str | None = None,
        *,
        timeout: float = 5.0,
        clock: Clock | None = None,
    ) -> None:
        self.root = Path(root or default_queue_root()).expanduser().resolve()
        super().__init__(self.root / "jobs.sqlite3", timeout=timeout, clock=clock)

    @property
    def lock_path(self) -> Path:
        return self.root / "flusher.lock"

    @property
    def log_path(self) -> Path:
        return self.root / "flusher.log"

    @property
    def work_root(self) -> Path:
        return self.root / "work"

    def init(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        self.work_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.work_root, 0o700)
        descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        os.close(descriptor)
        os.chmod(self.lock_path, 0o600)
        with self.connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.executescript(SCHEMA_SQL)
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version not in {0, JOB_SCHEMA_VERSION}:
                raise RuntimeError(
                    f"unsupported local job queue schema {version}; "
                    f"this executable supports {JOB_SCHEMA_VERSION}"
                )
            connection.execute(f"PRAGMA user_version={JOB_SCHEMA_VERSION}")
        os.chmod(self.path, 0o600)

    def _event(
        self,
        connection: sqlite3.Connection,
        *,
        job_id: str,
        kind: str,
        detail: Mapping[str, Any] | None = None,
        now: float | None = None,
    ) -> None:
        connection.execute(
            """
            INSERT INTO job_events (job_id, kind, detail_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                job_id,
                str(kind),
                None if detail is None else _canonical_json(detail),
                self.clock.time() if now is None else float(now),
            ),
        )

    @staticmethod
    def _decode_job(row: sqlite3.Row | Mapping[str, Any]) -> dict[str, Any]:
        item = dict(row)
        item["payload"] = json.loads(str(item.pop("payload_json")))
        item["cancel_requested"] = bool(item["cancel_requested"])
        return item

    @staticmethod
    def _decode_subject(row: sqlite3.Row | Mapping[str, Any]) -> dict[str, Any]:
        item = dict(row)
        raw = item.pop("detail_json")
        item["detail"] = None if raw is None else json.loads(str(raw))
        return item

    def job(self, job_id: str) -> dict[str, Any] | None:
        self.init()
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
        return None if row is None else self._decode_job(row)

    def jobs(
        self,
        *,
        states: Sequence[str] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        self.init()
        query = "SELECT * FROM jobs"
        parameters: list[Any] = []
        if states:
            invalid = set(states) - JOB_STATES
            if invalid:
                raise ValueError(f"invalid job state: {sorted(invalid)[0]}")
            placeholders = ",".join("?" for _ in states)
            query += f" WHERE state IN ({placeholders})"
            parameters.extend(states)
        query += " ORDER BY created_at DESC, job_id DESC LIMIT ?"
        parameters.append(max(1, min(int(limit), 1000)))
        with self.connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._decode_job(row) for row in rows]

    def subjects(self, job_id: str) -> list[dict[str, Any]]:
        self.init()
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM job_subjects
                WHERE job_id = ?
                ORDER BY subject_type, subject_id
                """,
                (str(job_id),),
            ).fetchall()
        return [self._decode_subject(row) for row in rows]

    def events(self, job_id: str) -> list[dict[str, Any]]:
        self.init()
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT event_seq, job_id, kind, detail_json, created_at
                FROM job_events WHERE job_id = ? ORDER BY event_seq
                """,
                (str(job_id),),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            raw = item.pop("detail_json")
            item["detail"] = None if raw is None else json.loads(str(raw))
            result.append(item)
        return result

    def enqueue(
        self,
        *,
        job_type: str,
        handler_version: int,
        payload: Mapping[str, Any],
        idempotency_key: str,
        subjects: Sequence[JobSubject],
    ) -> EnqueueResult:
        self.init()
        handler = handler_for(job_type, handler_version)
        normalized = handler.validate_payload(payload)
        encoded = _canonical_json(normalized)
        identities = [
            (str(subject.subject_type), str(subject.subject_id))
            for subject in subjects
        ]
        if not identities or any(not kind or not identifier for kind, identifier in identities):
            raise ValueError("a job must have at least one valid subject")
        if len(set(identities)) != len(identities):
            raise ValueError("job subjects must be unique")
        now = self.clock.time()
        new_job_id = _job_id()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT * FROM jobs
                WHERE job_type = ? AND handler_version = ? AND idempotency_key = ?
                """,
                (str(job_type), int(handler_version), str(idempotency_key)),
            ).fetchone()
            if existing is not None:
                existing_subjects = connection.execute(
                    "SELECT * FROM job_subjects WHERE job_id = ?",
                    (str(existing["job_id"]),),
                ).fetchall()
                existing_identities = {
                    (str(row["subject_type"]), str(row["subject_id"]))
                    for row in existing_subjects
                }
                if (
                    str(existing["payload_json"]) != encoded
                    or existing_identities != set(identities)
                ):
                    raise ValueError(
                        "job idempotency key conflicts with a different request"
                    )
                return EnqueueResult(
                    job=self._decode_job(existing),
                    subjects=tuple(self._decode_subject(row) for row in existing_subjects),
                    created=False,
                )
            connection.execute(
                """
                INSERT INTO jobs (
                  job_id, job_type, handler_version, payload_json, idempotency_key,
                  state, available_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'queued', ?, ?, ?)
                """,
                (
                    new_job_id,
                    str(job_type),
                    int(handler_version),
                    encoded,
                    str(idempotency_key),
                    now,
                    now,
                    now,
                ),
            )
            for subject in subjects:
                connection.execute(
                    """
                    INSERT INTO job_subjects (
                      job_id, subject_type, subject_id, exclusive_key, state,
                      detail_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 'queued', ?, ?, ?)
                    """,
                    (
                        new_job_id,
                        subject.subject_type,
                        subject.subject_id,
                        subject.exclusive_key,
                        (
                            None
                            if subject.detail is None
                            else _canonical_json(subject.detail)
                        ),
                        now,
                        now,
                    ),
                )
            self._event(
                connection,
                job_id=new_job_id,
                kind="enqueued",
                detail={"subject_count": len(subjects)},
                now=now,
            )
        job = self.job(new_job_id)
        if job is None:
            raise RuntimeError("enqueued job disappeared")
        return EnqueueResult(
            job=job,
            subjects=tuple(self.subjects(new_job_id)),
            created=True,
        )

    def subject_statuses(
        self,
        *,
        subject_type: str,
        subject_ids: Sequence[str],
    ) -> dict[str, dict[str, Any]]:
        self.init()
        identifiers = tuple(dict.fromkeys(str(value) for value in subject_ids if str(value)))
        if not identifiers:
            return {}
        placeholders = ",".join("?" for _ in identifiers)
        with self.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT s.*, j.state AS job_state, j.last_error AS job_error,
                       j.cancel_requested, j.job_type, j.handler_version,
                       j.available_at, j.created_at AS job_created_at
                FROM job_subjects AS s
                JOIN jobs AS j ON j.job_id = s.job_id
                WHERE s.subject_type = ? AND s.subject_id IN ({placeholders})
                ORDER BY s.updated_at DESC, s.job_id DESC
                """,
                (str(subject_type), *identifiers),
            ).fetchall()
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            identifier = str(row["subject_id"])
            if identifier in result:
                continue
            item = dict(row)
            raw = item.pop("detail_json")
            item["detail"] = None if raw is None else json.loads(str(raw))
            item["cancel_requested"] = bool(item["cancel_requested"])
            result[identifier] = item
        return result

    def has_unfinished(self) -> bool:
        self.init()
        placeholders = ",".join("?" for _ in ACTIVE_JOB_STATES)
        with self.connection() as connection:
            row = connection.execute(
                f"SELECT 1 FROM jobs WHERE state IN ({placeholders}) LIMIT 1",
                tuple(ACTIVE_JOB_STATES),
            ).fetchone()
        return row is not None

    def claim_next(self) -> dict[str, Any] | None:
        self.init()
        now = self.clock.time()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM jobs
                WHERE state IN ('queued', 'retry_wait') AND available_at <= ?
                ORDER BY created_at, job_id
                LIMIT 1
                """,
                (now,),
            ).fetchone()
            if row is None:
                return None
            job_id = str(row["job_id"])
            connection.execute(
                """
                UPDATE jobs SET
                  state = 'running',
                  attempts = attempts + 1,
                  started_at = COALESCE(started_at, ?),
                  updated_at = ?,
                  last_error = NULL
                WHERE job_id = ? AND state IN ('queued', 'retry_wait')
                """,
                (now, now, job_id),
            )
            connection.execute(
                """
                UPDATE job_subjects SET
                  state = CASE WHEN state IN ('queued', 'retry_wait') THEN 'running' ELSE state END,
                  updated_at = ?
                WHERE job_id = ?
                """,
                (now, job_id),
            )
            self._event(connection, job_id=job_id, kind="claimed", now=now)
            claimed = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return None if claimed is None else self._decode_job(claimed)

    def finish(self, job_id: str, result: HandlerResult) -> dict[str, Any]:
        self.init()
        if result.state not in {"retry_wait", *TERMINAL_JOB_STATES}:
            raise ValueError(f"invalid handler result state: {result.state}")
        if result.state == "retry_wait" and result.available_at is None:
            raise ValueError("retry_wait requires available_at")
        now = self.clock.time()
        available_at = now if result.available_at is None else float(result.available_at)
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT state FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
            if current is None:
                raise ValueError(f"unknown job: {job_id}")
            if str(current["state"]) != "running":
                raise RuntimeError(
                    f"job {job_id} cannot finish from state {current['state']}"
                )
            finished_at = now if result.state in TERMINAL_JOB_STATES else None
            connection.execute(
                """
                UPDATE jobs SET state = ?, available_at = ?, last_error = ?,
                  updated_at = ?, finished_at = ?
                WHERE job_id = ?
                """,
                (
                    result.state,
                    available_at,
                    result.message,
                    now,
                    finished_at,
                    str(job_id),
                ),
            )
            for subject in result.subjects:
                if subject.state not in ACTIVE_SUBJECT_STATES | TERMINAL_SUBJECT_STATES:
                    raise ValueError(f"invalid subject state: {subject.state}")
                cursor = connection.execute(
                    """
                    UPDATE job_subjects SET state = ?, detail_json = ?, updated_at = ?
                    WHERE job_id = ? AND subject_type = ? AND subject_id = ?
                    """,
                    (
                        subject.state,
                        (
                            None
                            if subject.detail is None
                            else _canonical_json(subject.detail)
                        ),
                        now,
                        str(job_id),
                        subject.subject_type,
                        subject.subject_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise ValueError(
                        f"job {job_id} has no subject "
                        f"{subject.subject_type}/{subject.subject_id}"
                    )
            if result.state in TERMINAL_JOB_STATES:
                fallback = {
                    "succeeded": "succeeded",
                    "failed": "failed",
                    "blocked": "blocked",
                    "canceled": "canceled",
                }[result.state]
                connection.execute(
                    """
                    UPDATE job_subjects SET state = ?, updated_at = ?
                    WHERE job_id = ? AND state IN (
                      'queued', 'running', 'retry_wait',
                      'waiting_for_training_terminal', 'waiting_for_run_lease',
                      'submitted', 'submission_uncertain', 'awaiting_projection'
                    )
                    """,
                    (fallback, now, str(job_id)),
                )
            self._event(
                connection,
                job_id=str(job_id),
                kind=f"state:{result.state}",
                detail=None if result.message is None else {"message": result.message},
                now=now,
            )
        updated = self.job(job_id)
        if updated is None:
            raise RuntimeError("finished job disappeared")
        return updated

    def recover_running(self) -> int:
        self.init()
        now = self.clock.time()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT job_id FROM jobs WHERE state = 'running'"
            ).fetchall()
            for row in rows:
                job_id = str(row["job_id"])
                connection.execute(
                    """
                    UPDATE jobs SET state = 'queued', available_at = ?, updated_at = ?,
                      last_error = 'worker stopped before completing this advancement'
                    WHERE job_id = ?
                    """,
                    (now, now, job_id),
                )
                connection.execute(
                    """
                    UPDATE job_subjects SET state = 'queued', updated_at = ?
                    WHERE job_id = ? AND state = 'running'
                    """,
                    (now, job_id),
                )
                self._event(
                    connection,
                    job_id=job_id,
                    kind="recovered_after_worker_stop",
                    now=now,
                )
        return len(rows)

    def request_cancel(self, job_id: str) -> dict[str, Any]:
        self.init()
        now = self.clock.time()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT state FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
            if row is None:
                raise ValueError(f"unknown job: {job_id}")
            state = str(row["state"])
            if state in TERMINAL_JOB_STATES:
                return self._decode_job(
                    connection.execute(
                        "SELECT * FROM jobs WHERE job_id = ?",
                        (str(job_id),),
                    ).fetchone()
                )
            if state in {"queued", "retry_wait"}:
                connection.execute(
                    """
                    UPDATE jobs SET state = 'canceled', cancel_requested = 1,
                      updated_at = ?, finished_at = ?
                    WHERE job_id = ?
                    """,
                    (now, now, str(job_id)),
                )
                connection.execute(
                    """
                    UPDATE job_subjects SET state = 'canceled', updated_at = ?
                    WHERE job_id = ? AND state NOT IN ('accepted', 'rejected', 'failed')
                    """,
                    (now, str(job_id)),
                )
                kind = "canceled"
            else:
                connection.execute(
                    "UPDATE jobs SET cancel_requested = 1, updated_at = ? WHERE job_id = ?",
                    (now, str(job_id)),
                )
                kind = "cancel_requested"
            self._event(connection, job_id=str(job_id), kind=kind, now=now)
            updated = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
        return self._decode_job(updated)

    def retry(self, job_id: str) -> dict[str, Any]:
        self.init()
        now = self.clock.time()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT state FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
            if row is None:
                raise ValueError(f"unknown job: {job_id}")
            state = str(row["state"])
            if state not in {"blocked", "failed", "canceled"}:
                raise ValueError(f"job {job_id} cannot be retried from {state}")
            connection.execute(
                """
                UPDATE jobs SET state = 'queued', available_at = ?,
                  cancel_requested = 0, last_error = NULL, updated_at = ?,
                  finished_at = NULL
                WHERE job_id = ?
                """,
                (now, now, str(job_id)),
            )
            connection.execute(
                """
                UPDATE job_subjects SET state = 'queued', updated_at = ?
                WHERE job_id = ? AND state IN (
                  'blocked', 'failed', 'expired', 'canceled'
                )
                """,
                (now, str(job_id)),
            )
            self._event(connection, job_id=str(job_id), kind="retried", now=now)
            updated = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
        return self._decode_job(updated)

    def worker_state(self) -> dict[str, Any]:
        self.init()
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM worker_state WHERE singleton = 1"
            ).fetchone()
        return {} if row is None else dict(row)

    def record_worker(
        self,
        *,
        generation: str,
        pid: int,
        heartbeat_at: float | None = None,
    ) -> None:
        self.init()
        now = self.clock.time() if heartbeat_at is None else float(heartbeat_at)
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO worker_state (
                  singleton, generation, pid, heartbeat_at, executable,
                  worker_version, last_start_error, updated_at
                ) VALUES (1, ?, ?, ?, ?, ?, NULL, ?)
                ON CONFLICT(singleton) DO UPDATE SET
                  generation = excluded.generation,
                  pid = excluded.pid,
                  heartbeat_at = excluded.heartbeat_at,
                  executable = excluded.executable,
                  worker_version = excluded.worker_version,
                  last_start_error = NULL,
                  updated_at = excluded.updated_at
                """,
                (
                    str(generation),
                    int(pid),
                    now,
                    sys.executable,
                    JOB_SCHEMA_VERSION,
                    now,
                ),
            )

    def record_start_error(self, message: str) -> None:
        self.init()
        now = self.clock.time()
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO worker_state (
                  singleton, last_start_error, updated_at
                ) VALUES (1, ?, ?)
                ON CONFLICT(singleton) DO UPDATE SET
                  last_start_error = excluded.last_start_error,
                  updated_at = excluded.updated_at
                """,
                (str(message)[:4000], now),
            )

    def next_available_at(self) -> float | None:
        self.init()
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT MIN(available_at) FROM jobs
                WHERE state IN ('queued', 'retry_wait')
                """
            ).fetchone()
        value = None if row is None else row[0]
        return None if value is None else float(value)


def _open_lock(store: JobStore) -> int:
    store.init()
    descriptor = os.open(store.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    os.chmod(store.lock_path, 0o600)
    return descriptor


def lock_is_held(store: JobStore) -> bool:
    descriptor = _open_lock(store)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return False
    finally:
        os.close(descriptor)


def _rotate_log(path: Path, *, maximum_bytes: int = 2 * 1024 * 1024) -> None:
    if not path.exists() or path.stat().st_size < maximum_bytes:
        return
    rotated = path.with_suffix(path.suffix + ".1")
    if rotated.exists():
        rotated.unlink()
    path.replace(rotated)


def ensure_flusher(
    store: JobStore,
    *,
    start_timeout_seconds: float = DEFAULT_START_TIMEOUT_SECONDS,
    idle_seconds: float = DEFAULT_IDLE_SECONDS,
) -> WorkerStart:
    store.init()
    if not store.has_unfinished():
        return WorkerStart("already_running", "no unfinished work")
    if lock_is_held(store):
        state = store.worker_state()
        return WorkerStart(
            "already_running",
            pid=int(state["pid"]) if state.get("pid") is not None else None,
        )
    generation = secrets.token_hex(16)
    _rotate_log(store.log_path)
    try:
        log = store.log_path.open("ab", buffering=0)
        os.chmod(store.log_path, 0o600)
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "gradlab.jobs_cli",
                    "--queue-dir",
                    str(store.root),
                    "_worker",
                    "--generation",
                    generation,
                    "--idle-seconds",
                    str(max(0.0, float(idle_seconds))),
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                close_fds=True,
                start_new_session=True,
            )
        finally:
            log.close()
    except Exception as exc:
        message = f"could not launch the job flusher: {exc}"
        store.record_start_error(message)
        return WorkerStart("start_failed", message)

    deadline = time.monotonic() + max(0.1, float(start_timeout_seconds))
    while time.monotonic() < deadline:
        state = store.worker_state()
        if state.get("generation") == generation:
            if lock_is_held(store):
                return WorkerStart("started", pid=int(state.get("pid") or process.pid))
            return_code = process.poll()
            if return_code == 0 and not store.has_unfinished():
                return WorkerStart(
                    "started",
                    "flusher started and drained the queue",
                    int(state.get("pid") or process.pid),
                )
        if process.poll() is not None:
            if lock_is_held(store):
                state = store.worker_state()
                return WorkerStart(
                    "already_running",
                    pid=int(state["pid"]) if state.get("pid") is not None else None,
                )
            message = f"job flusher exited during startup with status {process.returncode}"
            store.record_start_error(message)
            return WorkerStart("start_failed", message)
        time.sleep(0.05)
    if lock_is_held(store):
        state = store.worker_state()
        return WorkerStart(
            "already_running",
            "another flusher won the startup race",
            int(state["pid"]) if state.get("pid") is not None else None,
        )
    message = "job flusher did not acknowledge startup before the timeout"
    store.record_start_error(message)
    return WorkerStart("start_failed", message)


def run_flusher(
    store: JobStore,
    *,
    generation: str | None = None,
    idle_seconds: float = DEFAULT_IDLE_SECONDS,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    clock: Clock | None = None,
) -> int:
    store.init()
    worker_clock = clock or store.clock or SystemClock()
    descriptor = _open_lock(store)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        identity = generation or secrets.token_hex(16)
        store.recover_running()
        store.record_worker(generation=identity, pid=os.getpid())
        idle_since: float | None = None
        while True:
            store.record_worker(generation=identity, pid=os.getpid())
            job = store.claim_next()
            if job is not None:
                idle_since = None
                try:
                    try:
                        handler = handler_for(
                            str(job["job_type"]),
                            int(job["handler_version"]),
                        )
                    except ValueError as exc:
                        result = HandlerResult(
                            state="blocked",
                            message=str(exc),
                        )
                    else:
                        result = handler.advance(job)
                except Exception as exc:
                    result = HandlerResult(
                        state="failed",
                        message=f"{type(exc).__name__}: {exc}",
                    )
                store.finish(str(job["job_id"]), result)
                continue

            next_at = store.next_available_at()
            if next_at is not None:
                idle_since = None
                delay = max(0.0, next_at - worker_clock.time())
                worker_clock.sleep(max(0.01, min(float(poll_seconds), delay or poll_seconds)))
                continue

            now = worker_clock.monotonic()
            if idle_since is None:
                idle_since = now
            if now - idle_since < max(0.0, float(idle_seconds)):
                worker_clock.sleep(max(0.01, float(poll_seconds)))
                continue

            # The RESERVED lock makes enqueue commit mutually exclusive with
            # the final emptiness check. Release leadership before committing:
            # a later enqueue can then start a successor, which waits for this
            # transaction to finish before claiming.
            with store.connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    """
                    SELECT 1 FROM jobs
                    WHERE state IN ('queued', 'running', 'retry_wait')
                    LIMIT 1
                    """
                ).fetchone()
                if row is not None:
                    continue
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            return 0
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(descriptor)


__all__ = [
    "ACTIVE_JOB_STATES",
    "HandlerResult",
    "JobHandler",
    "JobStore",
    "JobSubject",
    "SubjectUpdate",
    "TERMINAL_JOB_STATES",
    "WorkerStart",
    "default_queue_root",
    "ensure_flusher",
    "handler_for",
    "lock_is_held",
    "register_handler",
    "run_flusher",
]
