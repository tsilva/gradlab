from __future__ import annotations

from pathlib import Path
import json
import sys

import pytest

from gradlab import autoresearch
from gradlab.json_utils import canonical_json_sha256
from gradlab.r2_store import BucketConfig, RunStorageConfig
from gradlab.run_authority import RunAuthority


RUN_ID = "gradlab-" + "a" * 32
ATTEMPT_ID = "attempt-" + "b" * 16


def history_event(seq: int, step: int, metrics: dict) -> dict:
    return {
        "event_seq": seq,
        "event_id": f"event-{seq}",
        "source": "train",
        "kind": "history",
        "step": step,
        "payload": metrics,
    }


def retained_run(tmp_path: Path, events: list[dict], *, high_water: int | None = None):
    authority = RunAuthority(
        RunStorageConfig(
            control=BucketConfig((tmp_path / "control").as_uri()),
            evaluation=BucketConfig((tmp_path / "eval").as_uri()),
            models=BucketConfig(
                (tmp_path / "models").as_uri(), public_base_url="https://models.example.test"
            ),
        )
    )
    manifest = {"run_id": RUN_ID, "attempt_id": ATTEMPT_ID}
    terminal = {
        **manifest,
        "state": "complete_local",
        "drain": {"metric_segment_high_water": high_water or events[-1]["event_seq"]},
    }
    authority.control.put_json(f"runs/{RUN_ID}/manifest.json", manifest)
    authority.control.put_json(f"runs/{RUN_ID}/attempts/{ATTEMPT_ID}/terminal.json", terminal)
    segment_key, _digest = authority.seal_metric_segment(
        run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        events=events,
    )
    args = {
        "root": tmp_path,
        "authority": authority,
        "run_id": RUN_ID,
        "attempt_id": ATTEMPT_ID,
        "journal_high_water": terminal["drain"]["metric_segment_high_water"],
        "terminal_sha256": canonical_json_sha256(terminal),
        "manifest_sha256": canonical_json_sha256(manifest),
        "starts": ["A", "B"],
        "strong_threshold": 0.9,
    }
    return args, segment_key


def test_success_evidence_reads_all_starts_and_ignores_later_events(tmp_path: Path):
    first = history_event(
        1,
        50_176,
        {
            "train/success/A/count": 2,
            "train/success/B/count": 0,
            "train/success/min": 0.91,
        },
    )
    args, _ = retained_run(tmp_path, [first])
    args["authority"].seal_metric_segment(
        run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        events=[
            history_event(
                2,
                100_000,
                {
                    "train/success/B/count": 99,
                    "train/success/min": 1.0,
                },
            )
        ],
    )
    evidence = autoresearch.fetch_training_evidence(**args)
    assert evidence["authority"] == "gradlab_metric_journal"
    assert evidence["run_id"] == RUN_ID
    assert not evidence["all_starts_succeeded"]
    assert evidence["success_counts_by_start"] == {"A": 2, "B": 0}
    assert evidence["peak_rolling_success_rate_min"] == 0.91
    assert evidence["first_strong_step"] == 50_176
    assert evidence["observed_max_step"] == 50_176


def test_return_evidence_needs_no_success_series_or_metrics_service(tmp_path: Path, monkeypatch):
    events = [history_event(i, i * 4096, {"train/return/mean": float(i)}) for i in range(1, 101)]
    args, _ = retained_run(tmp_path, events)
    monkeypatch.setitem(sys.modules, "wandb", None)
    monkeypatch.setitem(sys.modules, "mlflow", None)
    evidence = autoresearch.fetch_training_evidence(**args, mode=autoresearch.EVIDENCE_RETURN)
    assert evidence["return_evidence_valid"]
    assert evidence["return_points"] == 100
    assert evidence["return_tail_mean"] == 95.5
    assert evidence["return_tail_p05"] == 91.45
    assert evidence["success_counts_by_start"] == {"A": 0, "B": 0}
    assert not evidence["strong"]


def test_return_tail_uses_training_steps_with_irregular_logging(tmp_path: Path):
    events = [history_event(i, i, {"train/return/mean": 1.0}) for i in range(1, 101)]
    events += [
        history_event(101 + i, 91_000 + i * 1000, {"train/return/mean": 100.0}) for i in range(10)
    ]
    args, _ = retained_run(tmp_path, events)
    evidence = autoresearch.fetch_training_evidence(**args, mode=autoresearch.EVIDENCE_RETURN)
    assert evidence["return_tail_points"] == 10
    assert evidence["return_tail_mean"] == 100.0
    assert evidence["return_evidence_valid"]


@pytest.mark.parametrize("failure", ["tamper", "gap", "short", "terminal", "manifest"])
def test_invalid_or_unfenced_journal_is_rejected(tmp_path: Path, failure: str):
    events = [history_event(1, 64, {"train/return/mean": 1.0})]
    if failure == "gap":
        events.append(history_event(3, 128, {"train/return/mean": 2.0}))
    args, segment_key = retained_run(tmp_path, events, high_water=2 if failure == "short" else None)
    control = args["authority"].control
    if failure == "tamper":
        content = control.get_bytes(segment_key).replace(b"1.0", b"9.0")
        (tmp_path / "control" / segment_key).write_bytes(content)
    elif failure == "terminal":
        key = f"runs/{RUN_ID}/attempts/{ATTEMPT_ID}/terminal.json"
        terminal = control.get_json(key)
        (tmp_path / "control" / key).write_text(json.dumps({**terminal, "state": "failed"}))
    elif failure == "manifest":
        (tmp_path / "control" / f"runs/{RUN_ID}/manifest.json").write_text(
            json.dumps({"run_id": RUN_ID, "seed": 999})
        )
    with pytest.raises((ValueError, RuntimeError)):
        autoresearch.fetch_training_evidence(**args)
