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
from gradlab.monitor_work import EpisodeWork

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
        return not any(
            self._locked(p.parent.relative_to(self.root)) for p in self.root.rglob("ownership.lock")
        )

    def available_slots(self, limit):
        return max(
            0,
            limit
            - sum(
                self._locked(p.parent.relative_to(self.root))
                for p in self.root.rglob("ownership.lock")
            ),
        )

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
        active = settings.get("task_cpus", 1) - self.available_slots(settings.get("task_cpus", 1))
        if active >= settings.get("task_cpus", 1):
            raise OSError("shared monitoring process budget exhausted")
        if used >= settings.get("spool_bytes", 2**63) or (active + 1) * settings.get(
            "worker_spool_bytes", 0
        ) > settings.get("spool_bytes", 2**63):
            raise OSError("shared monitoring spool budget exhausted")
        self._spawn(directory, intent)
        return handle

    def _spawn(self, directory, intent, *, episode_root=None):
        identity = str(directory.relative_to(self.root))
        directory.mkdir(parents=True, exist_ok=True)
        request = directory / "request.json"
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
        if episode_root is not None:
            environment["GRADLAB_MONITOR_EPISODE_ROOT"] = str(episode_root)
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
                return
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

    def expand(self, handles, limit):
        """Fill final-drain slots fairly; each process owns one memory/spool share."""
        while True:
            if not self.available_slots(limit):
                return
            candidates = []
            for handle in handles:
                directory = self.root / handle.call_id
                if not self._locked(handle.call_id) or (directory / "result.json").exists():
                    continue
                intent = json.loads((directory / "request.json").read_text())
                work = EpisodeWork(directory, len(intent["manifest"]))
                if work.remaining() > 0:
                    children = list((directory / "helpers").glob("*/request.json"))
                    candidates.append((len(children), directory, intent, work))
            if not candidates:
                return
            _, directory, intent, work = min(candidates, key=lambda row: (row[0], str(row[1])))
            with work.locked():
                if work.remaining() <= 0:
                    continue
                helper = directory / "helpers" / str(len(list((directory / "helpers").glob("*"))))
                try:
                    self._spawn(helper, intent, episode_root=directory)
                except Exception as exc:
                    # Preserve the failed helper so poll routes it through the
                    # checkpoint's one execution retry, rather than failing drain.
                    atomic_write_json(
                        helper / "result.json",
                        dict(status="failed", error=f"{type(exc).__name__}: {exc}"[:1000]),
                    )
                    return

    def _helpers(self, handle):
        return [
            EvalHandle("cpu", str(p.parent.relative_to(self.root)))
            for p in (self.root / handle.call_id / "helpers").glob("*/request.json")
        ]

    def poll(self, handle):
        if (
            handle.provider != "cpu"
            or len(handle.call_id) != 64
            or any(c not in "0123456789abcdef" for c in handle.call_id)
        ):
            raise ValueError("invalid CPU evaluation handle")
        directory = self.root / handle.call_id
        for helper in self._helpers(handle):
            process = self.processes.get(helper.call_id)
            if process is not None:
                process.poll()
            if not self._locked(helper.call_id):
                result = self.root / helper.call_id / "result.json"
                document = json.loads(result.read_text()) if result.exists() else {}
                if document.get("status") != "succeeded":
                    return EvalPoll(
                        "failed", error=document.get("error", "episode worker interrupted")
                    )
        process = self.processes.get(handle.call_id)
        request = json.loads((directory / "request.json").read_text())
        if process is not None:
            process.poll()  # Reap local exits before inspecting the ownership lock.
        if time.time() >= request["deadline"]:
            self.cancel(handle)
            return EvalPoll("failed", error="whole-run monitoring deadline exhausted")
        if self._locked(handle.call_id):
            return EvalPoll("running")
        result = directory / "result.json"
        if result.exists():
            document = json.loads(result.read_text())
            if document["status"] == "succeeded":
                if any(self._locked(h.call_id) for h in self._helpers(handle)):
                    return EvalPoll("running")
                return EvalPoll("succeeded", provider_result=document["result"])
            return EvalPoll("failed", error=document["error"])
        return EvalPoll("failed", error="CPU worker interrupted before durable completion")

    def cancel(self, handle):
        for helper in self._helpers(handle):
            self.cancel(helper)
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
        helpers = self._helpers(handle)
        if self._locked(handle.call_id) or any(self._locked(h.call_id) for h in helpers):
            raise RuntimeError("cannot reclaim a running monitoring worker")
        for helper in helpers:
            self.forget(helper)
        process = self.processes.pop(handle.call_id, None)
        if process is not None:
            process.wait(timeout=5)
        shutil.rmtree(self.root / handle.call_id, ignore_errors=True)
