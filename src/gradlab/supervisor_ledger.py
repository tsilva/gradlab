from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from gradlab.json_utils import canonical_json_text
from gradlab.metric_store import MetricStore
from gradlab.run_contracts import (
    EVAL_INVENTORY_SETTLED_STATUSES,
    EVAL_RESULT_TERMINAL_STATUSES,
)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS supervisor_state (
  key TEXT PRIMARY KEY,
  value_json TEXT NOT NULL,
  updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS metric_segments (
  first_event_seq INTEGER NOT NULL,
  last_event_seq INTEGER NOT NULL UNIQUE,
  object_key TEXT NOT NULL UNIQUE,
  sha256 TEXT NOT NULL,
  event_count INTEGER NOT NULL,
  created_at REAL NOT NULL,
  PRIMARY KEY (first_event_seq, last_event_seq)
);

CREATE TABLE IF NOT EXISTS checkpoint_publications (
  checkpoint_ledger_id INTEGER PRIMARY KEY,
  checkpoint_id TEXT NOT NULL UNIQUE,
  step INTEGER NOT NULL,
  purpose TEXT NOT NULL,
  manifest_json TEXT NOT NULL,
  published_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_dispatches (
  idempotency_key TEXT PRIMARY KEY,
  checkpoint_ledger_id INTEGER NOT NULL UNIQUE,
  checkpoint_id TEXT NOT NULL,
  checkpoint_step INTEGER NOT NULL,
  intent_json TEXT NOT NULL,
  attempt INTEGER NOT NULL DEFAULT 0,
  modal_call_id TEXT,
  attempt_expires_at REAL,
  status TEXT NOT NULL DEFAULT 'pending',
  result_json TEXT,
  result_observed_at REAL,
  stop_requested_at REAL,
  last_error TEXT,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS eval_dispatches_status_idx
  ON eval_dispatches (status, checkpoint_step);
"""


class SupervisorLedger(MetricStore):
    def init(self) -> None:
        super().init()
        with self.connection() as connection:
            connection.executescript(SCHEMA_SQL)

    def state(self, key: str, default: Any = None) -> Any:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT value_json FROM supervisor_state WHERE key = ?",
                (key,),
            ).fetchone()
        return default if row is None else json.loads(str(row["value_json"]))

    def set_state(self, key: str, value: Any) -> None:
        now = self.clock.time()
        payload = canonical_json_text(value, default=str, ensure_ascii=True)
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO supervisor_state (key, value_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                  value_json = excluded.value_json,
                  updated_at = excluded.updated_at
                """,
                (key, payload, now),
            )

    def next_metric_events(self, *, limit: int = 1000) -> list[dict[str, Any]]:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(last_event_seq), 0) FROM metric_segments"
            ).fetchone()
            high_water = int(row[0] if row else 0)
            rows = connection.execute(
                """
                SELECT id, event_id, step, source, kind, payload_json, created_at
                FROM metric_frames
                WHERE id > ?
                ORDER BY id
                LIMIT ?
                """,
                (high_water, max(1, int(limit))),
            ).fetchall()
        return [
            {
                "event_seq": int(row["id"]),
                "event_id": str(row["event_id"]),
                "step": row["step"],
                "source": str(row["source"]),
                "kind": str(row["kind"]),
                "payload": json.loads(str(row["payload_json"])),
                "created_at": float(row["created_at"]),
            }
            for row in rows
        ]

    def record_metric_segment(
        self,
        *,
        events: Sequence[Mapping[str, Any]],
        object_key: str,
        sha256: str,
    ) -> None:
        if not events:
            raise ValueError("metric segment cannot be empty")
        first = int(events[0]["event_seq"])
        last = int(events[-1]["event_seq"])
        now = self.clock.time()
        with self.connection() as connection:
            previous = connection.execute(
                "SELECT COALESCE(MAX(last_event_seq), 0) FROM metric_segments"
            ).fetchone()
            if int(previous[0] if previous else 0) >= first:
                raise ValueError("metric segment overlaps the durable high-water mark")
            connection.execute(
                """
                INSERT INTO metric_segments (
                  first_event_seq, last_event_seq, object_key, sha256, event_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (first, last, object_key, sha256, len(events), now),
            )

    def metric_segment_high_water(self) -> int:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(last_event_seq), 0) FROM metric_segments"
            ).fetchone()
        return int(row[0] if row else 0)

    def record_checkpoint_publication(
        self,
        *,
        checkpoint_ledger_id: int,
        manifest: Mapping[str, Any],
    ) -> None:
        now = self.clock.time()
        payload = canonical_json_text(dict(manifest), ensure_ascii=True)
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO checkpoint_publications (
                  checkpoint_ledger_id, checkpoint_id, step, purpose, manifest_json, published_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(checkpoint_ledger_id) DO NOTHING
                """,
                (
                    int(checkpoint_ledger_id),
                    str(manifest["checkpoint_id"]),
                    int(manifest["step"]),
                    str(manifest["purpose"]),
                    payload,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT manifest_json FROM checkpoint_publications "
                "WHERE checkpoint_ledger_id = ?",
                (int(checkpoint_ledger_id),),
            ).fetchone()
            if row is None or json.loads(str(row["manifest_json"])) != dict(manifest):
                raise RuntimeError("checkpoint publication conflicts with durable local state")

    def checkpoint_publication(self, checkpoint_ledger_id: int) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT manifest_json FROM checkpoint_publications "
                "WHERE checkpoint_ledger_id = ?",
                (int(checkpoint_ledger_id),),
            ).fetchone()
        return None if row is None else json.loads(str(row["manifest_json"]))

    def checkpoint_publication_by_id(
        self,
        checkpoint_id: str,
    ) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT checkpoint_ledger_id, manifest_json "
                "FROM checkpoint_publications WHERE checkpoint_id = ?",
                (str(checkpoint_id),),
            ).fetchone()
        if row is None:
            return None
        return {
            "checkpoint_ledger_id": int(row["checkpoint_ledger_id"]),
            **json.loads(str(row["manifest_json"])),
        }

    def checkpoint_publications(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT checkpoint_ledger_id, manifest_json FROM checkpoint_publications "
                "ORDER BY step, checkpoint_ledger_id"
            ).fetchall()
        return [
            {
                "checkpoint_ledger_id": int(row["checkpoint_ledger_id"]),
                **json.loads(str(row["manifest_json"])),
            }
            for row in rows
        ]

    def ensure_eval(
        self,
        *,
        checkpoint_ledger_id: int,
        intent: Mapping[str, Any],
    ) -> None:
        now = self.clock.time()
        payload = canonical_json_text(dict(intent), ensure_ascii=True)
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO eval_dispatches (
                  idempotency_key, checkpoint_ledger_id, checkpoint_id,
                  checkpoint_step, intent_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(idempotency_key) DO NOTHING
                """,
                (
                    str(intent["idempotency_key"]),
                    int(checkpoint_ledger_id),
                    str(intent["checkpoint_id"]),
                    int(intent["checkpoint_step"]),
                    payload,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT intent_json FROM eval_dispatches WHERE idempotency_key = ?",
                (str(intent["idempotency_key"]),),
            ).fetchone()
            if row is None or json.loads(str(row["intent_json"])) != dict(intent):
                raise RuntimeError("eval intent conflicts with durable local state")

    def evals(self, *, statuses: Sequence[str] | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM eval_dispatches"
        parameters: list[Any] = []
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            query += f" WHERE status IN ({placeholders})"
            parameters.extend(str(status) for status in statuses)
        query += " ORDER BY checkpoint_step, checkpoint_ledger_id"
        with self.connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["intent"] = json.loads(str(item.pop("intent_json")))
            if item.get("result_json") is not None:
                item["result"] = json.loads(str(item.pop("result_json")))
            result.append(item)
        return result

    def eval(self, idempotency_key: str) -> dict[str, Any] | None:
        rows = [
            row
            for row in self.evals()
            if str(row["idempotency_key"]) == str(idempotency_key)
        ]
        return rows[0] if rows else None

    def _transition_eval(
        self,
        idempotency_key: str,
        *,
        from_statuses: Sequence[str],
        conflict: str,
        predicate: str = "",
        **fields: Any,
    ) -> None:
        assignments = ", ".join(f"{name} = ?" for name in fields)
        statuses = ",".join("?" for _ in from_statuses)
        with self.connection() as connection:
            cursor = connection.execute(
                f"""
                UPDATE eval_dispatches
                SET {assignments}, updated_at = ?
                WHERE idempotency_key = ? AND status IN ({statuses}) {predicate}
                """,
                (
                    *fields.values(),
                    self.clock.time(),
                    idempotency_key,
                    *from_statuses,
                ),
            )
        if cursor.rowcount != 1:
            raise RuntimeError(f"{conflict}: {idempotency_key}")

    def mark_eval_submitted(
        self,
        *,
        idempotency_key: str,
        attempt: int,
        modal_call_id: str,
        attempt_expires_at: float,
    ) -> None:
        self._transition_eval(
            idempotency_key,
            from_statuses=("pending",),
            conflict="eval intent is not pending",
            status="submitted",
            attempt=int(attempt),
            modal_call_id=modal_call_id,
            attempt_expires_at=float(attempt_expires_at),
            last_error=None,
        )

    def reset_expired_eval(self, *, idempotency_key: str, error: str) -> None:
        self._transition_eval(
            idempotency_key,
            from_statuses=("submitted",),
            predicate="AND attempt < 2",
            conflict="eval cannot be retried",
            status="pending",
            modal_call_id=None,
            attempt_expires_at=None,
            last_error=error[:4000],
        )

    def record_eval_error(self, *, idempotency_key: str, error: str) -> None:
        self._transition_eval(
            idempotency_key,
            from_statuses=("pending", "submitted"),
            conflict="active eval not found",
            last_error=error[:4000],
        )

    def mark_eval_terminal(
        self,
        *,
        idempotency_key: str,
        status: str,
        result: Mapping[str, Any],
    ) -> None:
        if status not in EVAL_RESULT_TERMINAL_STATUSES:
            raise ValueError(f"invalid terminal eval status: {status}")
        now = self.clock.time()
        payload = canonical_json_text(dict(result), default=str, ensure_ascii=True)
        with self.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE eval_dispatches
                SET status = ?, result_json = ?, result_observed_at = ?,
                    last_error = NULL, updated_at = ?
                WHERE idempotency_key = ?
                  AND status IN ('pending', 'submitted')
                """,
                (status, payload, now, now, idempotency_key),
            )
            if cursor.rowcount == 0:
                row = connection.execute(
                    "SELECT status, result_json FROM eval_dispatches WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
                if (
                    row is None
                    or str(row["status"]) != status
                    or json.loads(str(row["result_json"])) != dict(result)
                ):
                    raise RuntimeError(f"eval terminal result conflicts: {idempotency_key}")

    def mark_eval_deferred(self, *, idempotency_key: str, reason: str) -> None:
        now = self.clock.time()
        with self.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE eval_dispatches
                SET status = 'deferred', last_error = ?, updated_at = ?
                WHERE idempotency_key = ?
                  AND status IN ('pending', 'submitted')
                """,
                (reason[:4000], now, idempotency_key),
            )
            if cursor.rowcount == 0:
                row = connection.execute(
                    "SELECT status FROM eval_dispatches WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
                if row is None or str(row["status"]) != "deferred":
                    raise RuntimeError(f"eval cannot be deferred: {idempotency_key}")

    def mark_stop_requested(self, *, idempotency_key: str) -> float:
        now = self.clock.time()
        with self.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE eval_dispatches
                SET stop_requested_at = COALESCE(stop_requested_at, ?), updated_at = ?
                WHERE idempotency_key = ? AND status = 'accepted'
                """,
                (now, now, idempotency_key),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"accepted eval not found: {idempotency_key}")
            row = connection.execute(
                "SELECT stop_requested_at FROM eval_dispatches WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
        return float(row["stop_requested_at"])

    def all_evals_terminal(self) -> bool:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) FROM eval_dispatches
                WHERE status NOT IN ('accepted', 'rejected', 'failed', 'expired', 'canceled')
                """
            ).fetchone()
        return int(row[0] if row else 0) == 0

    def all_evals_settled(self) -> bool:
        with self.connection() as connection:
            placeholders = ", ".join("?" for _ in EVAL_INVENTORY_SETTLED_STATUSES)
            row = connection.execute(
                f"SELECT COUNT(*) FROM eval_dispatches WHERE status NOT IN ({placeholders})",
                tuple(sorted(EVAL_INVENTORY_SETTLED_STATUSES)),
            ).fetchone()
        return int(row[0] if row else 0) == 0

    def terminal_eval_count(self) -> int:
        with self.connection() as connection:
            placeholders = ", ".join("?" for _ in EVAL_RESULT_TERMINAL_STATUSES)
            row = connection.execute(
                f"SELECT COUNT(*) FROM eval_dispatches WHERE status IN ({placeholders})",
                tuple(sorted(EVAL_RESULT_TERMINAL_STATUSES)),
            ).fetchone()
        return int(row[0] if row else 0)

    def deferred_eval_count(self) -> int:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM eval_dispatches WHERE status = 'deferred'"
            ).fetchone()
        return int(row[0] if row else 0)
