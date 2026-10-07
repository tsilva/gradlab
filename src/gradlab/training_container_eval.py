"""Bounded, recoverable CPU evaluation inside the training container."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from gradlab.eval_backend import EvalHandle, EvalPoll
from gradlab.file_utils import atomic_write_json
from gradlab.json_utils import canonical_json_sha256
from gradlab.monitor_bootstrap import process_start


_CALL_ID = re.compile(r"[0-9a-f]{64}\Z")


class TrainingContainerEvalBackend:
    """Queue attempts on disk and run at most ``max_workers`` CPU children."""

    def __init__(self, root: Path, *, max_workers: int = 1) -> None:
        if max_workers < 1:
            raise ValueError("training-container evaluation requires a positive worker limit")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_workers = int(max_workers)
        self.processes: dict[str, subprocess.Popen[bytes]] = {}

    def _directory(self, handle: EvalHandle) -> Path:
        if handle.provider != "training-container" or _CALL_ID.fullmatch(handle.call_id) is None:
            raise ValueError("invalid training-container evaluation handle")
        return self.root / handle.call_id

    @staticmethod
    def _held(directory: Path) -> bool:
        lock_path = directory / "ownership.lock"
        if not lock_path.exists():
            return False
        with lock_path.open("a+b") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
        return False

    def _pump(self) -> None:
        active = sum(self._held(path) for path in self.root.iterdir() if path.is_dir())
        if active >= self.max_workers:
            return
        for directory in sorted(path for path in self.root.iterdir() if path.is_dir()):
            if active >= self.max_workers:
                break
            if any((directory / name).exists() for name in ("started.json", "result.json", "canceled.json")):
                continue
            with (directory / "ownership.lock").open("a+b") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                environment = {
                    key: os.environ[key]
                    for key in ("PATH", "HOME", "TMPDIR", "SSL_CERT_FILE", "LD_LIBRARY_PATH")
                    if key in os.environ
                }
                environment.update(
                    PYTHONPATH=str(Path(__file__).resolve().parents[1]),
                    CUDA_VISIBLE_DEVICES="",
                    OMP_NUM_THREADS="1",
                    MKL_NUM_THREADS="1",
                    OPENBLAS_NUM_THREADS="1",
                    NUMEXPR_NUM_THREADS="1",
                )
                with (directory / "worker.log").open("ab") as output:
                    process = subprocess.Popen(
                        [sys.executable, "-m", "gradlab.training_container_eval_worker", str(directory / "request.json")],
                        env=environment,
                        stdin=subprocess.DEVNULL,
                        stdout=output,
                        stderr=output,
                        start_new_session=True,
                        pass_fds=(lock.fileno(),),
                    )
                atomic_write_json(
                    directory / "started.json",
                    {"pid": process.pid, "created": process_start(process.pid)},
                )
                self.processes[directory.name] = process
                active += 1

    def submit(self, payload: dict[str, object]) -> EvalHandle:
        call_id = canonical_json_sha256(payload)
        handle = EvalHandle(provider="training-container", call_id=call_id)
        directory = self._directory(handle)
        directory.mkdir(parents=True, exist_ok=True)
        request = directory / "request.json"
        if request.exists():
            if json.loads(request.read_text(encoding="utf-8")) != payload:
                raise ValueError("training-container evaluation request identity conflict")
        else:
            atomic_write_json(request, payload)
        self._pump()
        return handle

    def poll(self, handle: EvalHandle) -> EvalPoll:
        directory = self._directory(handle)
        if not directory.is_dir():
            return EvalPoll(status="failed", error="training-container worker state unavailable")
        self._pump()
        process = self.processes.get(handle.call_id)
        if process is not None:
            process.poll()
        if self._held(directory):
            return EvalPoll(status="running")
        if (directory / "canceled.json").exists():
            return EvalPoll(status="canceled")
        result_path = directory / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") == "succeeded":
                return EvalPoll(status="succeeded", provider_result=result.get("result"))
            return EvalPoll(status="failed", error=str(result.get("error") or "worker failed"))
        if (directory / "started.json").exists():
            return EvalPoll(status="failed", error="training-container worker exited without a result")
        return EvalPoll(status="running")

    def cancel(self, handle: EvalHandle) -> None:
        directory = self._directory(handle)
        if not directory.is_dir():
            return
        atomic_write_json(directory / "canceled.json", {"canceled": True})
        process = self.processes.get(handle.call_id)
        if process is not None and process.poll() is None:
            pid = process.pid
        elif self._held(directory):
            started = json.loads((directory / "started.json").read_text(encoding="utf-8"))
            pid = int(started["pid"])
            created = started.get("created")
            if created is None or process_start(pid) != created:
                raise RuntimeError("training-container worker process identity changed")
        else:
            return
        for signal_value in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pid, signal_value)
            except ProcessLookupError:
                return
            deadline = time.monotonic() + 5
            while self._held(directory) and time.monotonic() < deadline:
                time.sleep(0.05)
            if not self._held(directory):
                return
        raise RuntimeError("training-container worker did not quiesce")
