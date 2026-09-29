"""Verified, service-independent reads of a Run's retained metric journal."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from gradlab.metric_names import (
    EVAL_CHECKPOINT_STEP,
    METRICS_SCHEMA_VERSION,
    MONITORING_SCALAR_METRICS,
    TRAIN_GLOBAL_STEP,
    validate_metric_payload,
)
from gradlab.evaluation_projection import validate_evaluation_metric_payload
from gradlab.run_contracts import RUN_ID_PATTERN


def read_control_journal(control: Any, run_id: str) -> list[dict[str, Any]]:
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("invalid GradLab Run ID for metric journal")
    prefix = f"runs/{run_id}/attempts/"
    events: dict[int, dict[str, Any]] = {}
    for key in sorted(control.iter_keys(prefix)):
        if "/metric-segments/" not in key or not key.endswith(".jsonl"):
            continue
        content = control.get_bytes(key)
        expected = key.removesuffix(".jsonl").rsplit("-", 1)[-1]
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"metric journal segment hash mismatch: {key}")
        for line in content.splitlines():
            raw = json.loads(line)
            if not isinstance(raw, Mapping):
                raise ValueError("metric journal event is not an object")
            event = dict(raw)
            sequence = int(event["event_seq"])
            if sequence < 1 or not str(event.get("event_id") or ""):
                raise ValueError("metric journal event has invalid identity")
            prior = events.setdefault(sequence, event)
            if prior != event:
                raise ValueError("conflicting metric journal event sequence")
    ordered = [events[key] for key in sorted(events)]
    if ordered and [int(row["event_seq"]) for row in ordered] != list(
        range(1, int(ordered[-1]["event_seq"]) + 1)
    ):
        raise ValueError("metric journal sequence has a gap")
    return ordered


class JournalHistory:
    """Small scan_history-compatible view used by Playback's metric reducers."""

    def __init__(self, events: list[Mapping[str, Any]]):
        self.rows: list[dict[str, Any]] = []
        for event in events:
            kind = str(event.get("kind") or "")
            step = int(event.get("step") or 0)
            if kind == "history":
                payload = dict(event["payload"])
                is_eval = str(event.get("source") or "").startswith("eval")
                if is_eval:
                    validate_evaluation_metric_payload(payload, schema_version=METRICS_SCHEMA_VERSION)
                else:
                    validate_metric_payload(payload)
                axis = EVAL_CHECKPOINT_STEP if is_eval else TRAIN_GLOBAL_STEP
                self.rows.append({**payload, axis: float(step)})
            elif kind == "monitoring":
                payload = dict(event["payload"])
                metrics = dict(payload["metrics"])
                if not set(metrics).issubset(MONITORING_SCALAR_METRICS):
                    raise ValueError("journal monitoring contains an unregistered measure")
                self.rows.append({**metrics, EVAL_CHECKPOINT_STEP: float(step)})

    def scan_history(self, *, keys: list[str], page_size: int):
        del page_size
        return ({key: row[key] for key in keys if key in row} for row in self.rows)


class PublicJournalHistory:
    """A verified public telemetry document through the same history reader seam."""

    def __init__(self, document: Mapping[str, Any], *, run_id: str, digest: str):
        from gradlab.json_utils import canonical_json_sha256

        if canonical_json_sha256(document) != digest:
            raise ValueError("public telemetry content hash mismatch")
        if document.get("schema_version") != 1 or document.get("run_id") != run_id:
            raise ValueError("public telemetry Run identity mismatch")
        raw_rows = document.get("histories")
        if not isinstance(raw_rows, list) or not raw_rows:
            raise ValueError("public telemetry has no scientific histories")
        self.rows = []
        for row in raw_rows:
            if not isinstance(row, Mapping):
                raise ValueError("public telemetry history row is invalid")
            self.rows.append(dict(row))

    def scan_history(self, *, keys: list[str], page_size: int):
        del page_size
        return ({key: row[key] for key in keys if key in row} for row in self.rows)
