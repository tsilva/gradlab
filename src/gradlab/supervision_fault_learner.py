from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from gradlab.file_utils import atomic_write_json
from gradlab.training_lifecycle import (
    LEARNER_READY_FILENAME,
    LEARNER_STATE_FORMAT_VERSION,
    TRAINING_RESULT_FILENAME,
)


FIXTURE_MODES = (
    "failed-result-live-process",
    "completed-result-hung-process",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _child_command(*, ignore_term: bool) -> list[str]:
    handlers = (
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "signal.signal(signal.SIGUSR1, signal.SIG_IGN);"
        if ignore_term
        else ""
    )
    return [
        sys.executable,
        "-c",
        (
            "import signal,time;"
            f"{handlers}"
            "time.sleep(3600)"
        ),
    ]


def run_fixture(*, train_config_path: Path, mode: str) -> int:
    if mode not in FIXTURE_MODES:
        raise ValueError(f"unsupported supervision fault fixture: {mode}")
    config = json.loads(train_config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("fault fixture train config must be an object")
    run_id = str(config["wandb_run_id"])
    attempt_id = str(config["attempt_id"])
    backend_id = str(dict(config["training_backend"])["id"])
    run_dir = Path(str(config["runs_dir"])) / str(config["run_name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    atomic_write_json(
        run_dir / LEARNER_READY_FILENAME,
        {
            "document_type": "gradlab.learner-ready",
            "format_version": LEARNER_STATE_FORMAT_VERSION,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "learner_pid": pid,
            "status": "ready",
            "execution_mode": "supervised",
            "training_backend_id": backend_id,
            "ready_at": _utc_now(),
        },
    )
    completed_hung = mode == "completed-result-hung-process"
    child = subprocess.Popen(
        _child_command(ignore_term=completed_hung),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if completed_hung:
        signal.signal(signal.SIGUSR1, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    status = "completed" if completed_hung else "failed"
    terminal_reason = "resource_exhaustion" if completed_hung else "failed"
    document = {
        "document_type": "gradlab.training-result",
        "format_version": LEARNER_STATE_FORMAT_VERSION,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "learner_pid": pid,
        "training_backend_id": backend_id,
        "status": status,
        "terminal_reason": terminal_reason,
        "execution_mode": "supervised",
        "execution_policy": {
            "mode": "supervised",
            "console_mode": "plain",
            "persist_intermediate_checkpoints": True,
            "stop_on_first_completion": False,
            "handle_sigint": False,
        },
        "first_completion_step": None,
        "final_step": 0,
        "requested_limit": 1,
        "execution_limit": 1,
        "model_kind": "final" if completed_hung else None,
        "model": "final_model.zip" if completed_hung else None,
        "terminal_at": _utc_now(),
    }
    if not completed_hung:
        document.update(
            {
                "error_type": "SupervisionFaultFixture",
                "error_message": "intentional failed-result/live-process fixture",
            }
        )
    atomic_write_json(run_dir / TRAINING_RESULT_FILENAME, document)
    print(
        f"supervision fault fixture active mode={mode} learner_pid={pid} child_pid={child.pid}",
        flush=True,
    )
    while True:
        time.sleep(60)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Non-production learner fixture for supervisor fault certification."
    )
    parser.add_argument("--train-config-json", type=Path, required=True)
    parser.add_argument("--mode", choices=FIXTURE_MODES, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_fixture(train_config_path=args.train_config_json, mode=args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
