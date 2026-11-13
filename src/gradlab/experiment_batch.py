"""Deferred, sequential experiment admission in the existing durable job queue.

The optional reviewed Runbox transport mints credentials only at dispatch. Queue
payloads and observer snapshots contain metadata, never session capabilities.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from gradlab.job_queue import (
    HandlerResult,
    JobStore,
    JobSubject,
    SubjectUpdate,
    ensure_flusher,
    register_handler,
)

TYPE = "experiment-batch"
VERSION = 1
RUN = re.compile(r"gradlab-[0-9a-f]{32}\Z")


def _write(path: Path, value: dict) -> None:
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(value, sort_keys=True) + "\n")
    pending.replace(path)


def _rows(text: str) -> list[dict]:
    result = []
    decoder = json.JSONDecoder()
    while text.strip():
        text = text.lstrip()
        try:
            item, end = decoder.raw_decode(text)
        except ValueError:
            _, _, text = text.partition("\n")
            continue
        if isinstance(item, dict):
            result.append(item)
        text = text[end:]
    return result


def _command(payload: dict, arguments: list[str], *, follow: bool = False) -> list[str]:
    runner = shutil.which("runbox")
    if runner is None:
        raise RuntimeError("Reviewed Runbox transport is unavailable; no fallback")
    profile, command = (
        ("training-follow", "gradlab-follow") if follow else ("training", "gradlab-training")
    )
    return [
        runner,
        "--repo",
        payload["repo_root"],
        "--profile",
        profile,
        "run",
        "--",
        command,
        *arguments,
    ]


def _call(payload: dict, arguments: list[str], output: Path) -> dict:
    try:
        result = subprocess.run(
            _command(payload, arguments),
            cwd=payload["repo_root"],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.SubprocessError:
        raise RuntimeError(
            "Experiment command interrupted; inspect durable admission before recovery"
        ) from None
    rows = _rows(result.stdout)
    # Do not retain provider diagnostics or credential-bearing URLs.
    _write(output, {"returncode": result.returncode, "records": rows})
    if result.returncode:
        raise RuntimeError("Experiment command failed; private provider output withheld")
    return next((r for r in reversed(rows) if "event" not in r), {})


def _observer(payload: dict, run: str, work: Path) -> dict | None:
    snapshot, lock = work / (run + ".json"), work / (run + ".lock")
    if (work / (run + ".observer-error.json")).exists():
        raise RuntimeError("Run observer exited unsuccessfully; access/recovery review required")
    start = False
    with lock.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            terminal = json.loads(snapshot.read_text()) if snapshot.exists() else {}
            if terminal.get("completed"):
                return terminal
            start = True
    if start:
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "gradlab.experiment_batch",
                "_follow",
                "--repo",
                payload["repo_root"],
                "--run",
                run,
                "--work",
                str(work),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            cwd=payload["repo_root"],
        )
    return json.loads(snapshot.read_text()) if snapshot.exists() else None


def _follow(root: str, run: str, work: Path) -> int:
    if not RUN.fullmatch(run):
        raise ValueError("Invalid Run identity")
    with (work / (run + ".lock")).open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        try:
            child = subprocess.Popen(
                _command({"repo_root": root}, ["--run", run, "--pretty"], follow=True),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                cwd=root,
            )
        except OSError, RuntimeError:
            _write(work / (run + ".observer-error.json"), {"startup_failed": True})
            return 1
        try:
            pending = []
            latest = None
            for line in child.stdout:
                if pending:
                    pending.append(line)
                    if line.rstrip("\r\n") != "}":
                        continue
                    text = "".join(pending)
                    pending = []
                elif line.strip() == "{":
                    pending = [line]
                    continue
                else:
                    text = line
                for row in _rows(text):
                    if row.get("run_id") == run and "dstack" in row:
                        _write(work / (run + ".json"), row)
                        latest = row
            code = child.wait()
            if code or not latest or not latest.get("completed"):
                _write(
                    work / (run + ".observer-error.json"),
                    {
                        "returncode": code,
                        "terminal_snapshot_received": bool(latest and latest.get("completed")),
                    },
                )
                return code or 1
            return 0
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=30)


class ExperimentBatch:
    job_type, version = TYPE, VERSION

    @classmethod
    def validate_payload(cls, value):
        p = dict(value)
        root = Path(str(p.get("repo_root", ""))).resolve()
        if not root.is_dir() or not (root / ".runbox.yaml").is_file():
            raise ValueError("Batch requires an existing reviewed Runbox checkout")
        if not re.fullmatch("[0-9a-f]{40}", str(p.get("source_sha", ""))):
            raise ValueError("Batch requires an immutable source SHA")
        if not re.fullmatch(r"[a-z0-9-]{1,120}", str(p.get("batch_id", ""))):
            raise ValueError("Batch requires a stable identity")
        if not Path(str(p.get("queue_root", ""))).is_absolute():
            raise ValueError("Batch requires the durable queue root")
        intents = p.get("intents")
        if not isinstance(intents, list) or not 1 <= len(intents) <= 31:
            raise ValueError("Batch requires 1..31 finite intents")
        keys = set()
        for intent in intents:
            if not isinstance(intent, dict) or not re.fullmatch(
                r"[a-z0-9-]{1,120}", str(intent.get("key", ""))
            ):
                raise ValueError("Invalid batch intent")
            if intent["key"] in keys:
                raise ValueError("Duplicate batch intent")
            keys.add(intent["key"])
            recipe = Path(str(intent.get("recipe", "")))
            if (
                recipe.is_absolute()
                or ".." in recipe.parts
                or not str(recipe).startswith("experiments/goals/")
            ):
                raise ValueError("Invalid batch recipe")
            if not (root / recipe).is_file() or recipe.suffix != ".yaml":
                raise ValueError("Batch recipe is missing")
            duration = intent.get("max_duration_seconds")
            if type(duration) is not int or not 1 <= duration <= 21600:
                raise ValueError("Batch duration must be finite and at most six hours")
            if type(intent.get("seed")) is not int:
                raise ValueError("Batch requires an explicit seed")
            if not isinstance(intent.get("description"), str) or not intent["description"].strip():
                raise ValueError("Batch requires each Run's purpose")
            if not isinstance(intent.get("overrides", []), list) or any(
                not isinstance(v, str) for v in intent.get("overrides", [])
            ):
                raise ValueError("Invalid recipe overrides")
        if p.get("after_run") and not RUN.fullmatch(str(p["after_run"])):
            raise ValueError("Invalid predecessor Run")
        if (
            type(p.get("deadline")) not in (int, float)
            or not math.isfinite(p["deadline"])
            or p["deadline"] <= 0
        ):
            raise ValueError("Batch requires a finite admission deadline")
        if not re.fullmatch(r"[a-z0-9-]+", str(p.get("target", ""))):
            raise ValueError("Batch requires an explicit local target")
        p["repo_root"] = str(root)
        return p

    def advance(self, job):
        p = self.validate_payload(job["payload"])
        store = JobStore(p["queue_root"])
        work = store.work_root / job["job_id"]
        work.mkdir(mode=0o700, parents=True, exist_ok=True)
        subjects = {
            s["subject_id"]: s
            for s in store.subjects(job["job_id"])
            if s["subject_type"] == "experiment-intent"
        }
        wait = HandlerResult(state="retry_wait", available_at=time.time() + 30)
        if job.get("cancel_requested"):
            return HandlerResult(
                state="canceled",
                message="Future admissions canceled; admitted Runs retain their finite bounds",
            )
        if time.time() >= p["deadline"]:
            return HandlerResult(state="blocked", message="Batch admission deadline expired")
        if p.get("after_run"):
            previous = _observer(p, p["after_run"], work)
            if not previous or not previous.get("completed"):
                return wait
        for index, intent in enumerate(p["intents"]):
            subject = subjects[intent["key"]]
            if subject["state"] in {"succeeded", "failed"}:
                continue
            detail = subject.get("detail") or {}
            if detail.get("run_id"):
                status = _observer(p, detail["run_id"], work)
                if not status or not status.get("completed"):
                    return wait
                terminal = status.get("attempt_terminal") or {}
                if index == 0:
                    ack = work / "startup-verified.json"
                    proof = json.loads(ack.read_text()) if ack.exists() else {}
                    if (
                        proof.get("run_id") != detail["run_id"]
                        or proof.get("image_digest") != detail.get("image_digest")
                        or any(
                            proof.get(k) is not True
                            for k in (
                                "gpu_assigned",
                                "cuda_tensor_verified",
                                "learner_cuda_verified",
                            )
                        )
                    ):
                        return HandlerResult(
                            state="blocked",
                            message="First worker needs reviewed GPU/CUDA/learner startup verification before further admission",
                        )
                state = (
                    "succeeded" if terminal.get("state") in {"succeeded", "stopped"} else "failed"
                )
                if terminal.get("state") == "resumable_failure":
                    return HandlerResult(
                        state="blocked",
                        message="Operational Run failure requires review before more admissions",
                        subjects=(
                            SubjectUpdate(
                                "experiment-intent",
                                intent["key"],
                                "failed",
                                {**detail, "attempt_terminal": terminal},
                            ),
                        ),
                    )
                if state == "succeeded" and not (terminal.get("drain") or {}).get("complete"):
                    return HandlerResult(
                        state="blocked",
                        message="Successful Run is missing complete delivery/drain evidence",
                    )
                return HandlerResult(
                    state="retry_wait",
                    available_at=time.time() + 30,
                    subjects=(
                        SubjectUpdate(
                            "experiment-intent",
                            intent["key"],
                            state,
                            {**detail, "attempt_terminal": terminal},
                        ),
                    ),
                )
            if subject["state"] == "submission_uncertain" or detail.get("admission_started"):
                return HandlerResult(
                    state="blocked",
                    message="Admission acknowledgement is uncertain; reconcile immutable manifests before recovery",
                )
            if time.time() >= p["deadline"]:
                return HandlerResult(state="blocked", message="Batch admission deadline expired")
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=p["repo_root"], text=True
            ).strip()
            dirty = subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=p["repo_root"], text=True
            ).strip()
            if sha != p["source_sha"] or dirty:
                return HandlerResult(
                    state="blocked",
                    message="Batch source changed or is dirty; no automatic approval or admission",
                )
            common = [
                "--recipe-file",
                intent["recipe"],
                "--target",
                p["target"],
                "--checkpoint-eval-backend",
                "none",
            ]
            for override in intent.get("overrides", []):
                common += ["--set", override]
            ready = _call(
                p,
                ["experiment", "operator-preflight", *common, "--json"],
                work / (intent["key"] + "-preflight.json"),
            )
            if ready.get("status") != "ready":
                return HandlerResult(state="blocked", message="Experiment preflight is not ready")
            # Persist before the child can perform admission. A crash in this
            # interval always requires reconciliation, never duplicate launch.
            canceled = store.checkpoint(
                job["job_id"],
                kind="admission-starting",
                subjects=(
                    SubjectUpdate(
                        "experiment-intent",
                        intent["key"],
                        "submission_uncertain",
                        {"submission_key": intent["key"], "admission_started": True},
                    ),
                ),
            )
            if canceled:
                return HandlerResult(state="canceled", message="Canceled before admission")
            receipt = _call(
                p,
                [
                    "experiment",
                    "launch",
                    *common,
                    "--seed",
                    str(intent["seed"]),
                    "--compute",
                    "local",
                    "--max-duration",
                    str(intent["max_duration_seconds"]),
                    "--existing-runtime-only",
                    "--submission-key",
                    intent["key"],
                    "--run-description",
                    intent["description"],
                    "--json",
                ],
                work / (intent["key"] + "-admission.json"),
            )
            run = receipt.get("run_id", "")
            if not RUN.fullmatch(run):
                return HandlerResult(
                    state="blocked",
                    message="Admission receipt missing; reconcile immutable manifests",
                )
            return HandlerResult(
                state="retry_wait",
                available_at=time.time() + 1,
                subjects=(SubjectUpdate("experiment-intent", intent["key"], "submitted", receipt),),
            )
        # Intent identity reservations remain permanent. The fleet reservation
        # is released only after every admitted Run has terminal evidence.
        with store.connection() as connection:
            connection.execute(
                "UPDATE job_subjects SET exclusive_key = NULL WHERE job_id = ? AND exclusive_key = ?",
                (job["job_id"], "experiment-batch:" + p["target"]),
            )
        return HandlerResult(
            state="succeeded",
            message="All bounded batch intents reached terminal evidence; scientific qualification remains separate",
        )


def register_job_handler():
    register_handler(TYPE, VERSION, ExperimentBatch, replace=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["enqueue", "_follow"])
    parser.add_argument("--file", type=Path)
    parser.add_argument("--repo")
    parser.add_argument("--run")
    parser.add_argument("--work", type=Path)
    args = parser.parse_args()
    if args.action == "_follow":
        return _follow(args.repo, args.run, args.work)
    payload = ExperimentBatch.validate_payload(json.loads(args.file.read_text()))
    store = JobStore(payload["queue_root"])
    queued = store.enqueue(
        job_type=TYPE,
        handler_version=VERSION,
        payload=payload,
        idempotency_key=payload["batch_id"],
        subjects=[
            JobSubject(
                "compute-batch",
                payload["target"],
                exclusive_key="experiment-batch:" + payload["target"],
            )
        ]
        + [
            JobSubject(
                "experiment-intent",
                item["key"],
                exclusive_key="experiment-intent:" + item["key"],
            )
            for item in payload["intents"]
        ],
    )
    worker = ensure_flusher(store, job_types=[TYPE])
    print(
        json.dumps(
            {"job_id": queued.job["job_id"], "created": queued.created, "worker": worker.to_dict()}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
