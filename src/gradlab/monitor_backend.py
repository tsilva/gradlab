"""Same-host CPU implementation of the existing evaluation backend boundary."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

from gradlab.monitor_bootstrap import process_start

from gradlab.eval_backend import EvalHandle, EvalPoll
from gradlab.file_utils import atomic_write_json
from gradlab.json_utils import canonical_json_sha256


class SameHostEvalBackend:
    def __init__(self, root, bucket_config):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.bucket_config = bucket_config
        self.processes = {}

    @staticmethod
    def handle_for(intent):
        return EvalHandle("cpu", canonical_json_sha256(intent))

    def _locked(self, identity):
        directory = self.root / identity
        if not directory.exists():
            return False
        with (directory / "ownership.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
        return False

    def quiescent(self):
        return not any(self._locked(p.name) for p in self.root.iterdir() if p.is_dir())

    def submit(self, intent):
        handle = self.handle_for(intent)
        identity = handle.call_id
        directory = self.root / identity
        directory.mkdir(exist_ok=True)
        request = directory / "request.json"
        if request.exists() and json.loads(request.read_text()) != intent:
            raise ValueError("CPU monitoring intent identity conflict")
        if self._locked(identity) or (directory / "result.json").exists():
            return handle
        used = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
        settings = intent["settings"]
        active = sum(self._locked(p.name) for p in self.root.iterdir() if p.is_dir())
        if used >= settings.get("spool_bytes", 2**63) or (active + 1) * settings.get(
            "worker_spool_bytes", 0
        ) > settings.get("spool_bytes", 2**63):
            raise OSError("shared monitoring spool budget exhausted")
        atomic_write_json(request, intent)
        environment = {
            k: os.environ[k] for k in ("PATH", "HOME", "TMPDIR", "SSL_CERT_FILE") if k in os.environ
        }
        environment.update(
            PYTHONPATH=str(Path(__file__).resolve().parents[1]),
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            NUMEXPR_NUM_THREADS="1",
            VECLIB_MAXIMUM_THREADS="1",
            CUDA_VISIBLE_DEVICES="",
            GRADLAB_MONITOR_BUDGET_ROOT=str(self.root),
            GRADLAB_MONITOR_PARENT_PID=str(os.getpid()),
        )
        for name in (
            "uri",
            "endpoint_url",
            "region",
            "access_key_id",
            "secret_access_key",
            "public_base_url",
        ):
            environment["GRADLAB_MONITOR_R2_" + name.upper()] = getattr(self.bucket_config, name)
        # No W&B, control-private, HF or inherited cloud credentials enter workers.
        with (directory / "ownership.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return handle
            environment["GRADLAB_MONITOR_LOCK_FD"] = str(lock.fileno())
            with (directory / "worker.log").open("ab") as log:
                self.processes[identity] = subprocess.Popen(
                    [sys.executable, "-m", "gradlab.monitor_bootstrap", str(request)],
                    env=environment,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                    pass_fds=(lock.fileno(),),
                )
        return handle

    def poll(self, handle):
        if (
            handle.provider != "cpu"
            or len(handle.call_id) != 64
            or any(c not in "0123456789abcdef" for c in handle.call_id)
        ):
            raise ValueError("invalid CPU evaluation handle")
        directory = self.root / handle.call_id
        process = self.processes.get(handle.call_id)
        request = json.loads((directory / "request.json").read_text())
        if process is not None:
            process.poll()  # Reap local exits before inspecting the ownership lock.
        if self._locked(handle.call_id):
            if time.time() >= request["deadline"]:
                self.cancel(handle)
                return EvalPoll("failed", error="whole-run monitoring deadline exhausted")
            return EvalPoll("running")
        result = directory / "result.json"
        if result.exists():
            document = json.loads(result.read_text())
            if document["status"] == "succeeded":
                return EvalPoll("succeeded", provider_result=document["result"])
            return EvalPoll("failed", error=document["error"])
        return EvalPoll("failed", error="CPU worker interrupted before durable completion")

    def cancel(self, handle):
        process = self.processes.get(handle.call_id)
        if process is not None:
            process.poll()
        if not self._locked(handle.call_id):
            return
        if process is not None and process.poll() is None:
            pid = process.pid
        else:
            identity = self.root / handle.call_id / "process.json"
            deadline = time.monotonic() + 5
            while not identity.exists() and time.monotonic() < deadline:
                if not self._locked(handle.call_id):
                    return
                time.sleep(0.05)
            if not identity.exists():
                raise RuntimeError("monitoring worker ownership has not become recoverable")
            recorded = json.loads(identity.read_text())
            current = process_start(recorded["pid"])
            if current != recorded["created"] or current is None:
                if not self._locked(handle.call_id):
                    return
                raise RuntimeError("monitoring process identity changed")
            pid = recorded["pid"]
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pid, sig)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 5
            while self._locked(handle.call_id) and time.monotonic() < deadline:
                time.sleep(0.05)
            if not self._locked(handle.call_id):
                if process is not None:
                    process.wait(timeout=5)
                return
        raise RuntimeError("monitoring process group did not quiesce")

    def close(self):
        failures = []
        for directory in self.root.iterdir():
            if directory.is_dir() and (directory / "request.json").exists():
                try:
                    self.cancel(EvalHandle("cpu", directory.name))
                except Exception as exc:
                    failures.append(exc)
        if failures:
            raise RuntimeError("monitoring workers did not all quiesce") from failures[0]

    def forget(self, handle):
        if self._locked(handle.call_id):
            raise RuntimeError("cannot reclaim a running monitoring worker")
        process = self.processes.pop(handle.call_id, None)
        if process is not None:
            process.wait(timeout=5)
        shutil.rmtree(self.root / handle.call_id, ignore_errors=True)
