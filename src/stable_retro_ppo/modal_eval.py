from __future__ import annotations

import json
import os
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras
from stable_baselines3 import PPO

from stable_retro_ppo.artifacts import (
    require_env_config_from_model_metadata,
    require_training_metadata,
    stable_json_hash,
)
from stable_retro_ppo.device import resolve_sb3_device
from stable_retro_ppo.env import assert_rom_imported
from stable_retro_ppo.eval_runner import evaluate_model_episodes
from stable_retro_ppo.modal_core import (
    RUNS_DIR,
    VOLUME_ROOT,
    app,
    ensure_remote_roms,
    eval_queue_secret,
    image,
    safe_path_name,
    volume,
)
from stable_retro_ppo.wandb_artifacts import model_zip_from_download, write_downloaded_artifact_metadata


LOCAL_ARTIFACT_CACHE_DIR = Path("/tmp/stable-retro-ppo-eval-artifacts")


@dataclass(frozen=True)
class PrefetchedArtifact:
    path: Path
    download_elapsed_seconds: float


@dataclass(frozen=True)
class PrefetchedJob:
    job: dict[str, Any]
    future: Future[PrefetchedArtifact]


CLAIM_SQL = """
WITH next_job AS (
  SELECT id
  FROM eval_jobs
  WHERE
    status = 'pending'
    OR (
      status = 'running'
      AND lease_expires_at < now()
      AND attempts < max_attempts
    )
  ORDER BY priority DESC, id ASC
  LIMIT 1
  FOR UPDATE SKIP LOCKED
)
UPDATE eval_jobs AS job
SET
  status = 'running',
  lease_owner = %(worker_id)s,
  lease_expires_at = now() + (%(lease_seconds)s || ' seconds')::interval,
  attempts = attempts + 1,
  started_at = COALESCE(started_at, now()),
  error = NULL
FROM next_job
WHERE job.id = next_job.id
RETURNING
  job.id,
  job.candidate_id,
  job.stage,
  job.episodes,
  job.seed_start,
  job.priority,
  job.attempts,
  job.lease_owner,
  job.lease_expires_at,
  (
    SELECT artifact_ref
    FROM checkpoint_candidates
    WHERE checkpoint_candidates.id = job.candidate_id
  ) AS artifact_ref;
"""


def database_url() -> str:
    value = os.environ.get("DATABASE_URL") or os.environ.get("DIRECT_DATABASE_URL")
    if not value:
        raise RuntimeError("DATABASE_URL or DIRECT_DATABASE_URL must be available in Modal")
    return value


def connect():
    return psycopg2.connect(database_url(), cursor_factory=psycopg2.extras.RealDictCursor)


def close_quietly(conn) -> None:
    try:
        conn.close()
    except Exception:
        pass


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {
            "array_shape": list(value.shape),
            "array_dtype": str(value.dtype),
        }
    return value


def claim_job(conn, *, worker_id: str, lease_seconds: int) -> dict[str, Any] | None:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                CLAIM_SQL,
                {"worker_id": worker_id, "lease_seconds": lease_seconds},
            )
            row = cur.fetchone()
            return dict(row) if row else None


def mark_job_failed(conn, *, job: dict[str, Any], worker_id: str, error: str) -> bool:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE eval_jobs
                SET
                  status = CASE WHEN attempts >= max_attempts THEN 'failed' ELSE 'pending' END,
                  lease_owner = NULL,
                  lease_expires_at = NULL,
                  error = %(error)s,
                  finished_at = CASE WHEN attempts >= max_attempts THEN now() ELSE finished_at END
                WHERE id = %(job_id)s
                  AND lease_owner = %(worker_id)s
                  AND status = 'running'
                """,
                {
                    "job_id": job["id"],
                    "worker_id": worker_id,
                    "error": error[:4000],
                },
            )
            return cur.rowcount == 1


def commit_result(
    conn,
    *,
    job: dict[str, Any],
    worker_id: str,
    metrics: dict[str, Any],
) -> bool:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1
                FROM eval_jobs
                WHERE id = %(job_id)s
                  AND lease_owner = %(worker_id)s
                  AND status = 'running'
                FOR UPDATE
                """,
                {"job_id": job["id"], "worker_id": worker_id},
            )
            if cur.fetchone() is None:
                return False

            cur.execute(
                """
                INSERT INTO eval_results (
                  candidate_id,
                  job_id,
                  training_metadata_hash,
                  stage,
                  episodes,
                  seed_start,
                  completion_count,
                  completion_rate,
                  max_x_max,
                  reward_mean,
                  metrics_json
                )
                VALUES (
                  %(candidate_id)s,
                  %(job_id)s,
                  %(training_metadata_hash)s,
                  %(stage)s,
                  %(episodes)s,
                  %(seed_start)s,
                  %(completion_count)s,
                  %(completion_rate)s,
                  %(max_x_max)s,
                  %(reward_mean)s,
                  %(metrics_json)s
                )
                ON CONFLICT (candidate_id, training_metadata_hash, stage, episodes, seed_start)
                DO UPDATE SET
                  job_id = EXCLUDED.job_id,
                  completion_count = EXCLUDED.completion_count,
                  completion_rate = EXCLUDED.completion_rate,
                  max_x_max = EXCLUDED.max_x_max,
                  reward_mean = EXCLUDED.reward_mean,
                  metrics_json = EXCLUDED.metrics_json,
                  created_at = now()
                """,
                {
                    "candidate_id": job["candidate_id"],
                    "job_id": job["id"],
                    "training_metadata_hash": metrics["training_metadata_hash"],
                    "stage": job["stage"],
                    "episodes": job["episodes"],
                    "seed_start": job["seed_start"],
                    "completion_count": int(metrics["completion_count"]),
                    "completion_rate": float(metrics["completion_rate"]),
                    "max_x_max": int(metrics["max_x_max"]),
                    "reward_mean": float(metrics["reward_mean"]),
                    "metrics_json": psycopg2.extras.Json(json_safe(metrics)),
                },
            )
            cur.execute(
                """
                UPDATE eval_jobs
                SET
                  status = 'done',
                  lease_owner = NULL,
                  lease_expires_at = NULL,
                  finished_at = now(),
                  error = NULL
                WHERE id = %(job_id)s
                  AND lease_owner = %(worker_id)s
                  AND status = 'running'
                """,
                {"job_id": job["id"], "worker_id": worker_id},
            )
            return cur.rowcount == 1


def commit_result_with_reconnect(
    conn,
    *,
    job: dict[str, Any],
    worker_id: str,
    metrics: dict[str, Any],
) -> tuple[bool, Any]:
    try:
        return commit_result(conn, job=job, worker_id=worker_id, metrics=metrics), conn
    except (psycopg2.OperationalError, psycopg2.InterfaceError):
        close_quietly(conn)
        conn = connect()
        return commit_result(conn, job=job, worker_id=worker_id, metrics=metrics), conn


def mark_job_failed_with_reconnect(
    conn,
    *,
    job: dict[str, Any],
    worker_id: str,
    error: str,
) -> tuple[bool, Any]:
    try:
        return mark_job_failed(conn, job=job, worker_id=worker_id, error=error), conn
    except (psycopg2.OperationalError, psycopg2.InterfaceError):
        close_quietly(conn)
        conn = connect()
        return mark_job_failed(conn, job=job, worker_id=worker_id, error=error), conn


def download_model_artifact(
    ref: str,
    *,
    base_dir: Path = RUNS_DIR / "wandb_artifacts",
) -> Path:
    import wandb

    download_root = base_dir / safe_path_name(ref)
    download_root.mkdir(parents=True, exist_ok=True)
    artifact = wandb.Api().artifact(ref, type="model")
    model_path = model_zip_from_download(Path(artifact.download(root=str(download_root))))
    write_downloaded_artifact_metadata(model_path, artifact)
    return model_path


def prefetch_model_artifact(job: dict[str, Any]) -> PrefetchedArtifact:
    started_at = time.monotonic()
    model_path = download_model_artifact(
        str(job["artifact_ref"]),
        base_dir=LOCAL_ARTIFACT_CACHE_DIR,
    )
    return PrefetchedArtifact(
        path=model_path,
        download_elapsed_seconds=round(time.monotonic() - started_at, 3),
    )


def evaluate_job(
    job: dict[str, Any],
    *,
    device: str,
    model_path: Path | None = None,
) -> dict[str, Any]:
    if model_path is None:
        model_path = download_model_artifact(str(job["artifact_ref"]))
    training = require_training_metadata(model_path)
    config = require_env_config_from_model_metadata(model_path)
    training_hash = stable_json_hash(training)
    assert_rom_imported(config.game)
    model = PPO.load(model_path, device=resolve_sb3_device(device))
    metrics, _ = evaluate_model_episodes(
        model=model,
        config=config,
        episodes=int(job["episodes"]),
        seed=int(job["seed_start"]),
        max_steps=config.max_episode_steps,
        deterministic=False,
        completion_x_threshold=config.completion_x_threshold,
        n_envs=1,
        capture_best_video=False,
        extra={
            "job_id": int(job["id"]),
            "candidate_id": int(job["candidate_id"]),
            "checkpoint_artifact": str(job["artifact_ref"]),
            "training_metadata_hash": training_hash,
            "eval_seed": int(job["seed_start"]),
        },
    )
    return metrics


@app.function(
    image=image,
    volumes={str(VOLUME_ROOT): volume},
    timeout=6 * 60 * 60,
    cpu=4.0,
    memory=8192,
    secrets=[eval_queue_secret],
)
def eval_artifact_benchmark_remote(
    artifact_ref: str,
    episodes: int = 100,
    seed: int = 10007,
    n_envs: int = 0,
    env_threads: int = 0,
    device: str = "cpu",
    benchmark_cpu: float = 4.0,
    benchmark_memory_mib: int = 8192,
) -> dict[str, Any]:
    total_started_at = time.monotonic()
    ensure_remote_roms("evaluation benchmark")
    effective_n_envs = n_envs if n_envs > 0 else 1
    model_path = download_model_artifact(artifact_ref)
    training = require_training_metadata(model_path)
    training_hash = stable_json_hash(training)
    config = require_env_config_from_model_metadata(model_path)
    if env_threads > 0:
        config = replace(config, env_threads=env_threads)
    assert_rom_imported(config.game)
    model = PPO.load(model_path, device=resolve_sb3_device(device))
    eval_started_at = time.monotonic()
    try:
        metrics, _ = evaluate_model_episodes(
            model=model,
            config=config,
            episodes=episodes,
            seed=seed,
            max_steps=config.max_episode_steps,
            deterministic=False,
            completion_x_threshold=config.completion_x_threshold,
            n_envs=effective_n_envs,
            capture_best_video=False,
            extra={
                "checkpoint_artifact": artifact_ref,
                "training_metadata_hash": training_hash,
                "eval_seed": seed,
                "benchmark": True,
                "benchmark_cpu": benchmark_cpu,
                "benchmark_memory_mib": benchmark_memory_mib,
                "benchmark_env_threads": env_threads,
            },
        )
    finally:
        eval_elapsed_seconds = time.monotonic() - eval_started_at
        volume.commit()

    total_elapsed_seconds = time.monotonic() - total_started_at
    return {
        "artifact_ref": artifact_ref,
        "training_metadata_hash": training_hash,
        "episodes": episodes,
        "n_envs": effective_n_envs,
        "env_threads": env_threads,
        "cpu": benchmark_cpu,
        "memory_mib": benchmark_memory_mib,
        "eval_elapsed_seconds": round(eval_elapsed_seconds, 3),
        "total_elapsed_seconds": round(total_elapsed_seconds, 3),
        "eval_episodes_per_second": round(episodes / eval_elapsed_seconds, 4),
        "total_episodes_per_second": round(episodes / total_elapsed_seconds, 4),
        "completion_rate": float(metrics["completion_rate"]),
        "death_rate": float(metrics["death_rate"]),
        "max_x_max": int(metrics["max_x_max"]),
        "reward_mean": float(metrics["reward_mean"]),
    }


@app.local_entrypoint()
def eval_artifact_benchmark(
    artifact_ref: str,
    episodes: int = 100,
    seed: int = 10007,
    n_envs: int = 0,
    env_threads: int = 0,
    cpu: float = 4.0,
    memory_mib: int = 8192,
    device: str = "cpu",
) -> None:
    remote = eval_artifact_benchmark_remote.with_options(cpu=cpu, memory=memory_mib)
    result = remote.remote(
        artifact_ref=artifact_ref,
        episodes=episodes,
        seed=seed,
        n_envs=n_envs,
        env_threads=env_threads,
        device=device,
        benchmark_cpu=cpu,
        benchmark_memory_mib=memory_mib,
    )
    print(json.dumps(result, sort_keys=True), flush=True)


@app.function(
    image=image,
    volumes={str(VOLUME_ROOT): volume},
    timeout=6 * 60 * 60,
    cpu=4.0,
    memory=8192,
    secrets=[eval_queue_secret],
)
def eval_worker_remote(
    worker_name: str = "",
    max_jobs: int = 0,
    idle_polls: int = 2,
    idle_sleep_seconds: float = 5.0,
    lease_seconds: int = 1800,
    device: str = "cpu",
    prefetch_jobs: int = 2,
) -> dict[str, Any]:
    ensure_remote_roms("evaluation")
    worker_id = worker_name or f"modal-eval-{uuid.uuid4()}"
    started_at = time.monotonic()
    completed = 0
    failed = 0
    idle = 0
    processed_jobs: list[int] = []
    prefetch_target = max(prefetch_jobs, 1)
    pending_jobs: list[PrefetchedJob] = []

    conn = connect()
    prefetcher = ThreadPoolExecutor(max_workers=1, thread_name_prefix="artifact-prefetch")
    try:
        while max_jobs <= 0 or completed + failed < max_jobs:
            while len(pending_jobs) < prefetch_target and (
                max_jobs <= 0 or completed + failed + len(pending_jobs) < max_jobs
            ):
                job = claim_job(conn, worker_id=worker_id, lease_seconds=lease_seconds)
                if job is None:
                    break

                idle = 0
                future = prefetcher.submit(prefetch_model_artifact, job)
                pending_jobs.append(PrefetchedJob(job=job, future=future))
                print(
                    "claimed "
                    f"worker={worker_id} "
                    f"job_id={job['id']} "
                    f"candidate_id={job['candidate_id']} "
                    f"stage={job['stage']} "
                    f"episodes={job['episodes']} "
                    f"prefetch_queued={len(pending_jobs)}",
                    flush=True,
                )

            if not pending_jobs:
                idle += 1
                if idle >= idle_polls:
                    break
                time.sleep(idle_sleep_seconds)
                continue

            prefetched_job = pending_jobs.pop(0)
            job = prefetched_job.job
            try:
                prefetch_wait_started_at = time.monotonic()
                prefetched_artifact = prefetched_job.future.result()
                prefetch_wait_seconds = round(time.monotonic() - prefetch_wait_started_at, 3)
                print(
                    "prefetched "
                    f"worker={worker_id} "
                    f"job_id={job['id']} "
                    f"download_seconds={prefetched_artifact.download_elapsed_seconds:.3f} "
                    f"wait_seconds={prefetch_wait_seconds:.3f}",
                    flush=True,
                )
                metrics = evaluate_job(
                    job,
                    device=device,
                    model_path=prefetched_artifact.path,
                )
                metrics["artifact_download_seconds"] = prefetched_artifact.download_elapsed_seconds
                metrics["artifact_prefetch_wait_seconds"] = prefetch_wait_seconds
                committed, conn = commit_result_with_reconnect(
                    conn, job=job, worker_id=worker_id, metrics=metrics
                )
                if committed:
                    completed += 1
                    processed_jobs.append(int(job["id"]))
                    print(
                        "done "
                        f"worker={worker_id} "
                        f"job_id={job['id']} "
                        f"completion_rate={metrics['completion_rate']:.3f} "
                        f"max_x_max={metrics['max_x_max']} "
                        f"reward_mean={metrics['reward_mean']:.2f}",
                        flush=True,
                    )
                else:
                    failed += 1
                    print(f"lost lease worker={worker_id} job_id={job['id']}", flush=True)
            except Exception as exc:
                failed += 1
                try:
                    marked_failed, conn = mark_job_failed_with_reconnect(
                        conn, job=job, worker_id=worker_id, error=repr(exc)
                    )
                    suffix = "" if marked_failed else " mark_failed=false"
                except Exception as mark_exc:
                    suffix = f" mark_failed_error={mark_exc!r}"
                print(
                    f"failed worker={worker_id} job_id={job['id']} error={exc!r}{suffix}",
                    flush=True,
                )
    finally:
        prefetcher.shutdown(wait=False, cancel_futures=True)
        close_quietly(conn)
        volume.commit()

    return {
        "worker_id": worker_id,
        "completed": completed,
        "failed": failed,
        "processed_jobs": processed_jobs,
        "elapsed_seconds": round(time.monotonic() - started_at, 3),
    }


@app.local_entrypoint()
def eval_queue(
    runners: int = 2,
    max_jobs_per_runner: int = 0,
    idle_polls: int = 2,
    idle_sleep_seconds: float = 5.0,
    lease_seconds: int = 1800,
    cpu: float = 4.0,
    memory_mib: int = 8192,
    device: str = "cpu",
    prefetch_jobs: int = 2,
) -> None:
    calls = []
    remote = eval_worker_remote.with_options(cpu=cpu, memory=memory_mib)
    for index in range(runners):
        worker_name = f"modal-eval-{uuid.uuid4()}-{index + 1}"
        call = remote.spawn(
            worker_name=worker_name,
            max_jobs=max_jobs_per_runner,
            idle_polls=idle_polls,
            idle_sleep_seconds=idle_sleep_seconds,
            lease_seconds=lease_seconds,
            device=device,
            prefetch_jobs=prefetch_jobs,
        )
        calls.append((worker_name, call))
        print(f"started {worker_name}", flush=True)

    results = []
    for worker_name, call in calls:
        result = call.get()
        print(f"result {worker_name}: {json.dumps(result, sort_keys=True)}", flush=True)
        results.append(result)

    print(json.dumps({"runners": runners, "results": results}, sort_keys=True), flush=True)
