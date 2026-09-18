"""Same-host CPU implementation of the existing evaluation backend boundary."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

from gradlab.eval_backend import EvalHandle, EvalPoll
from gradlab.file_utils import atomic_write_json
from gradlab.json_utils import canonical_json_sha256


class SameHostEvalBackend:
    def __init__(self, root, bucket_config):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.bucket_config = bucket_config
        self.processes = {}

    def submit(self, intent):
        used = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
        settings = intent["settings"]
        active = sum(process.poll() is None for process in self.processes.values())
        if used >= settings.get("spool_bytes", 2**63) or (active + 1) * settings.get("worker_spool_bytes", 0) > settings.get("spool_bytes", 2**63):
            raise OSError("shared monitoring spool budget exhausted")
        identity = canonical_json_sha256(intent)
        directory = self.root / identity
        directory.mkdir(exist_ok=True)
        request = directory / "request.json"
        if request.exists() and json.loads(request.read_text()) != intent:
            raise ValueError("CPU monitoring intent identity conflict")
        atomic_write_json(request, intent)
        handle = EvalHandle("cpu", identity)
        if identity in self.processes or (directory / "result.json").exists():
            return handle
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
        with (directory / "worker.log").open("ab") as log:
            self.processes[identity] = subprocess.Popen(
                [sys.executable, "-m", "gradlab.monitor_worker", str(request)],
                env=environment,
                stdout=log,
                stderr=log,
                start_new_session=True,
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
        if process is not None and process.poll() is None:
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
        if process is None or process.poll() is not None:
            return
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)

    def forget(self, handle):
        process = self.processes.pop(handle.call_id, None)
        if process is not None and process.poll() is None:
            raise RuntimeError("cannot reclaim a running monitoring worker")
        shutil.rmtree(self.root / handle.call_id, ignore_errors=True)
