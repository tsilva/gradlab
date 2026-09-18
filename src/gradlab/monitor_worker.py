"""Credential-isolated, finite checkpoint-monitoring worker."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import time

from gradlab.checkpoint_monitoring import (
    finalize_monitoring,
    monitor_episode,
    verify_monitoring_inventory,
    verified_get,
    verified_put,
)
from gradlab.file_utils import atomic_write_json
from gradlab.json_utils import canonical_json_bytes
from gradlab.r2_store import BucketConfig, R2Bucket
from gradlab.run_contracts import CheckpointManifest


class ContributionBudget:
    """A shared local lock serializes durable, idempotent per-object reservations.

    Canonical reservations survive Attempts. Actual object bytes are charged along
    with reservation bytes; local deletion never refunds durable contributions.
    """

    def __init__(self, root, bucket, run_id, limit):
        self.root, self.bucket, self.limit = Path(root), bucket, limit
        self.root.mkdir(parents=True, exist_ok=True)
        self.prefix = f"monitoring/{run_id}/budget"
        self.cache = self.root / f"{run_id}-reservations.json"

    def reserve(self, key, size):
        # Small metadata reservations permit retry diagnostics/timings to vary
        # without spending the same identity twice; slack remains charged forever.
        if key.endswith(".json"):
            size = max(4096, 1 << max(0, size - 1).bit_length())
        name = f"{self.prefix}/{hashlib.sha256(key.encode()).hexdigest()}.json"
        with (self.root / "contribution.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.cache.exists():
                cached = json.loads(self.cache.read_text())
            else:
                entries = {
                    item: self.bucket.get_json(item) for item in self.bucket.iter_keys(self.prefix)
                }
                cached = dict(
                    entries=entries, total=sum(row["charged_bytes"] for row in entries.values())
                )
            old = cached["entries"].get(name)
            if old is not None:
                if old["key"] != key or old["object_bytes"] < size:
                    raise ValueError("monitoring contribution identity conflict")
                # The cache is persisted before upload; a lost upload still has to
                # be reconciled remotely before its reservation is considered safe.
                verified_put(self.bucket, name, canonical_json_bytes(old))
                return
            document = dict(key=key, object_bytes=size, charged_bytes=size)
            while True:
                encoded = canonical_json_bytes(document)
                charged = size + len(encoded)
                if charged == document["charged_bytes"]:
                    break
                document["charged_bytes"] = charged
            if cached["total"] + charged > self.limit:
                raise ValueError("cumulative monitoring contribution budget exhausted")
            cached["entries"][name] = document
            cached["total"] += charged
            atomic_write_json(self.cache, cached)
            verified_put(self.bucket, name, encoded)


def run_monitoring(intent, root, bucket):
    import torch
    from gradlab.model_sources import download_remote_model_source
    from gradlab.policy_models import load_internal_policy_model
    from gradlab.policy_registry import resolve_policy_algorithm
    from gradlab.env_config import env_config_from_mapping
    from gradlab.env import resolve_env_config

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = Path(root)
    settings = intent["settings"]
    started = time.perf_counter()
    measured = (settings.get("calibration") or {}).get("status") == "measuring"
    phases = {"r2_delivery": 0.0}

    class MeasuredBucket:
        def __getattr__(self, name):
            target = getattr(original_bucket, name)
            if not callable(target):
                return target

            def call(*args, **kwargs):
                before = time.perf_counter()
                try:
                    return target(*args, **kwargs)
                finally:
                    phases["r2_delivery"] += time.perf_counter() - before

            return call

    original_bucket = bucket
    if measured:
        bucket = MeasuredBucket()
    from gradlab.policy_runtime import PolicyRuntime

    checkpoint = CheckpointManifest.from_dict(
        {k: v for k, v in intent["checkpoint"].items() if k != "checkpoint_ledger_id"}
    )
    if checkpoint.run_id != intent["run_id"] or checkpoint.recipe_sha256 != intent["recipe_sha256"]:
        raise ValueError("monitoring checkpoint differs from the frozen Run")
    bucket_prefix = intent["prefix"]
    completed = bucket.get_json_optional(bucket_prefix + "/result.json")
    if completed is not None:
        verify_monitoring_inventory(bucket, completed, intent["manifest"], prefix=bucket_prefix)
        return completed
    model_ref = checkpoint.public_url.rsplit("/", 1)[0] + "/manifest.json"
    source = download_remote_model_source(model_ref, root=root / "policy", require_pinned=True)
    if source.bundle.model["checkpoint"]["sha256"] != checkpoint.sha256:
        raise ValueError("monitoring loaded a different checkpoint")
    recipe = source.bundle.recipe["recipe"]
    train = recipe["train_config"]
    backend_id = train["training_backend"]["id"]
    if backend_id not in {"gradlab.ppo", "sb3.ppo", "sb3.a2c"}:
        raise ValueError("monitoring requires a verified PPO/A2C checkpoint")
    if (
        train["env_provider"] != "env-breakoutatari2600-turbo-native"
        or train["game"] != "Breakout-Atari2600-v0"
    ):
        raise ValueError("monitoring requires native Breakout")
    config = resolve_env_config(env_config_from_mapping(train))
    config.env_args["num_threads"] = 1
    model = load_internal_policy_model(
        source.model_path,
        device="cpu",
        algorithm_id=resolve_policy_algorithm(source.bundle.model["policy"]),
        execution_id=f"monitoring:{intent['evaluation_id']}",
    )
    budget = ContributionBudget(
        os.environ["GRADLAB_MONITOR_BUDGET_ROOT"],
        bucket,
        intent["run_id"],
        settings["contribution_bytes"],
    )
    provenance = {
        k: intent[k]
        for k in (
            "run_id",
            "attempt_id",
            "evaluation_id",
            "contract_sha256",
            "source_sha",
            "runtime",
            "recipe_sha256",
            "goal_sha256",
            "environment_sha256",
            "goal_variant",
            "training_seed",
        )
    }
    provenance.update(
        checkpoint_id=checkpoint.checkpoint_id,
        checkpoint_step=checkpoint.step,
        checkpoint_sha256=checkpoint.sha256,
        provider=config.env_provider,
        planned_training_steps=train["timesteps"],
    )
    episodes = []
    peaks = {"peak_memory_bytes": 0, "peak_spool_bytes": 0}

    def guard():
        if time.time() >= intent["deadline"]:
            raise TimeoutError("whole-run monitoring deadline exhausted")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if sys.platform == "darwin" else 1024
        )
        if rss > settings["worker_memory_bytes"]:
            raise MemoryError("monitoring worker memory budget exhausted")
        used = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
        peaks["peak_memory_bytes"] = max(peaks["peak_memory_bytes"], rss)
        peaks["peak_spool_bytes"] = max(peaks["peak_spool_bytes"], used)
        if (
            used > settings["worker_spool_bytes"]
            or shutil.disk_usage(root).free < settings["scratch_headroom_bytes"]
        ):
            raise OSError("monitoring worker spool/headroom budget exhausted")

    guard()
    for episode in intent["manifest"]:
        prefix = f"{bucket_prefix}/{episode['episode_id']}"
        result = bucket.get_json_optional(prefix + "/episode.json")
        if result is not None:
            if (
                any(result.get(k) != v for k, v in {**episode, **provenance}.items())
                or not result["complete"]
            ):
                raise ValueError("recovered monitoring episode identity mismatch")
            for reference in result["chunks"]:
                verified_get(bucket, reference)
        else:
            result = monitor_episode(
                model=model,
                config=config,
                episode=episode,
                bucket=bucket,
                root=root / "spool",
                prefix=prefix,
                provenance=provenance,
                chunk_bytes=settings["chunk_bytes"],
                watchdog_steps=settings["watchdog_steps"],
                reserve=budget.reserve,
                guard=guard,
                measure=measured,
                policy_runtime=PolicyRuntime(
                    model, algorithm_id=resolve_policy_algorithm(source.bundle.model["policy"])
                ),
            )
        episodes.append(result)
        guard()

    def measurements():
        return dict(
            **peaks,
            seconds=time.perf_counter() - started,
            retained_bytes=sum(bucket.head(key)["size"] for key in bucket.iter_keys(bucket_prefix)),
            longest_episode_steps=max(e["steps"] for e in episodes),
            uninterrupted=intent.get("execution_attempt") == 1,
            phase_seconds={
                **phases,
                **{
                    name: sum(e.get("phase_seconds", {}).get(name, 0) for e in episodes)
                    for name in ("inference", "capture", "encoding")
                },
            },
        )

    result = finalize_monitoring(
        episodes,
        intent["manifest"],
        bucket=bucket,
        root=root / "video",
        prefix=bucket_prefix,
        fps=60 / config.frame_skip,
        reserve=budget.reserve,
        guard=guard,
        measurements=measurements if measured else None,
        max_video_bytes=min(
            settings["worker_spool_bytes"] // 4, settings["worker_memory_bytes"] // 8
        ),
    )
    shutil.rmtree(root / "policy")
    return result


def main():
    request = Path(sys.argv[1])
    intent = json.loads(request.read_text())

    def expired(_signal, _frame):
        raise TimeoutError("whole-run monitoring deadline exhausted")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(max(1, int(intent["deadline"] - time.time())))
    try:
        result = run_monitoring(
            intent, request.parent, R2Bucket(BucketConfig.from_env("GRADLAB_MONITOR_R2"))
        )
        document = dict(status="succeeded", result=result)
    except Exception as exc:
        document = dict(status="failed", error=f"{type(exc).__name__}: {exc}"[:1000])
    atomic_write_json(request.parent / "result.json", document)
    return 0 if document["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
