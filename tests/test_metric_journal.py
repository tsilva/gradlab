from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gradlab.metric_journal import JournalHistory, read_control_journal
from gradlab.supervisor_ledger import SupervisorLedger


class _Control:
    def __init__(self, key: str, payload: bytes):
        self.key = key
        self.payload = payload

    def iter_keys(self, prefix: str):
        return [self.key] if self.key.startswith(prefix) else []

    def get_bytes(self, key: str):
        assert key == self.key
        return self.payload


def test_playback_reads_verified_journal_without_a_service(tmp_path: Path) -> None:
    run_id = "gradlab-" + "d" * 32
    store = SupervisorLedger(tmp_path / "run.sqlite")
    store.init()
    store.append_metrics({"train/return/mean": 0.25}, step=64, source="train")
    store.append_metrics({"train/return/mean": 0.5}, step=128, source="train")
    events = store.next_metric_events()
    payload = b"".join(
        json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        + b"\n"
        for event in events
    )
    digest = hashlib.sha256(payload).hexdigest()
    key = (
        f"runs/{run_id}/attempts/attempt-{'e' * 16}/metric-segments/"
        f"{1:020d}-{2:020d}-{digest}.jsonl"
    )
    control = _Control(key, payload)
    view = JournalHistory(read_control_journal(control, run_id))
    rows = list(view.scan_history(keys=["train/step", "train/return/mean"], page_size=100))
    assert rows == [
        {"train/step": 64.0, "train/return/mean": 0.25},
        {"train/step": 128.0, "train/return/mean": 0.5},
    ]
    control.payload = payload.replace(b"0.25", b"0.75")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_control_journal(control, run_id)
