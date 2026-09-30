"""Verified, service-independent reads of a Run's retained metric journal."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from gradlab.evaluation_projection import validate_evaluation_metric_payload
from gradlab.metric_names import (
    EVAL_CHECKPOINT_STEP,
    METRICS_SCHEMA_VERSION,
    MONITORING_SCALAR_METRICS,
    TRAIN_GLOBAL_STEP,
    validate_metric_payload,
)
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


class MetricHistory:
    rows: list[dict[str, Any]]

    def read(self, keys: list[str]) -> Iterable[Mapping[str, Any]]:
        return ({key: row[key] for key in keys if key in row} for row in self.rows)


class JournalHistory(MetricHistory):
    """Validated scientific histories from a private Run journal."""

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



class PublicJournalHistory(MetricHistory):
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
        self.media = []
        for item in document.get("media", []):
            if (not isinstance(item, Mapping)
                    or item.get("kind") not in {"evaluation_video", "monitoring"}
                    or type(item.get("step")) is not int or item["step"] < 0
                    or type(item.get("bytes")) is not int or item["bytes"] <= 0):
                raise ValueError("public telemetry media reference is invalid")
            self.media.append(dict(item))
        for row in raw_rows:
            if not isinstance(row, Mapping):
                raise ValueError("public telemetry history row is invalid")
            self.rows.append(dict(row))



@dataclass(frozen=True)
class ScientificEvidence:
    """Backend-neutral history, frozen configuration, and representative media."""
    config: Mapping[str, Any]
    _read: Callable[[list[str]], Iterable[Mapping[str, Any]]]
    media: tuple[Mapping[str, Any], ...] = ()

    def read(self, keys: list[str]) -> Iterable[Mapping[str, Any]]:
        return (dict(row) for row in self._read(keys) if isinstance(row, Mapping))

    @classmethod
    def journal(cls, history: MetricHistory, config: Mapping[str, Any]) -> ScientificEvidence:
        return cls(dict(config), history.read, tuple(getattr(history, "media", ())))

    @classmethod
    def historical_wandb(cls, run: Any) -> ScientificEvidence:
        # Historical releases keep their original reader. Sparse SDK requests
        # stay independent so missing optional measures cannot hide verdicts.
        return cls(dict(getattr(run, "config", {}) or {}),
                   lambda keys: run.scan_history(keys=keys, page_size=10_000))
