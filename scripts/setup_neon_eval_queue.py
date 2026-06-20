from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras


SCHEMA_SQL = """
ALTER TABLE IF EXISTS eval_results
  DROP CONSTRAINT IF EXISTS eval_results_unique_profile_stage_seed;

ALTER TABLE IF EXISTS eval_jobs
  DROP CONSTRAINT IF EXISTS eval_jobs_unique_profile_stage_seed;

CREATE TABLE IF NOT EXISTS checkpoint_candidates (
  id BIGSERIAL PRIMARY KEY,
  artifact_ref TEXT NOT NULL UNIQUE,
  run_name TEXT,
  run_path TEXT,
  checkpoint_step INTEGER,
  priority INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'pending',
  note TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS eval_jobs (
  id BIGSERIAL PRIMARY KEY,
  candidate_id BIGINT NOT NULL REFERENCES checkpoint_candidates(id) ON DELETE CASCADE,
  stage TEXT NOT NULL,
  episodes INTEGER NOT NULL,
  seed_start INTEGER NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  priority INTEGER NOT NULL DEFAULT 0,
  attempts INTEGER NOT NULL DEFAULT 0,
  max_attempts INTEGER NOT NULL DEFAULT 3,
  lease_owner TEXT,
  lease_expires_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  started_at TIMESTAMPTZ,
  finished_at TIMESTAMPTZ,
  error TEXT
);

CREATE TABLE IF NOT EXISTS eval_results (
  id BIGSERIAL PRIMARY KEY,
  candidate_id BIGINT NOT NULL REFERENCES checkpoint_candidates(id) ON DELETE CASCADE,
  job_id BIGINT REFERENCES eval_jobs(id) ON DELETE SET NULL,
  training_metadata_hash TEXT NOT NULL,
  stage TEXT NOT NULL,
  episodes INTEGER NOT NULL,
  seed_start INTEGER NOT NULL,
  completion_count INTEGER NOT NULL,
  completion_rate DOUBLE PRECISION NOT NULL,
  max_x_max INTEGER NOT NULL,
  reward_mean DOUBLE PRECISION NOT NULL,
  metrics_json JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE eval_jobs
  DROP COLUMN IF EXISTS eval_profile;

ALTER TABLE eval_results
  DROP COLUMN IF EXISTS eval_profile;

ALTER TABLE eval_results
  ADD COLUMN IF NOT EXISTS training_metadata_hash TEXT;

ALTER TABLE eval_jobs
  DROP CONSTRAINT IF EXISTS eval_jobs_candidate_id_stage_episodes_seed_start_key;

ALTER TABLE eval_results
  DROP CONSTRAINT IF EXISTS eval_results_candidate_id_stage_episodes_seed_start_key;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'eval_jobs_unique_stage_seed'
  ) THEN
    ALTER TABLE eval_jobs
      ADD CONSTRAINT eval_jobs_unique_stage_seed
      UNIQUE (candidate_id, stage, episodes, seed_start);
  END IF;
END $$;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'eval_results_unique_training_metadata_stage_seed'
  ) THEN
    ALTER TABLE eval_results
      ADD CONSTRAINT eval_results_unique_training_metadata_stage_seed
      UNIQUE (candidate_id, training_metadata_hash, stage, episodes, seed_start);
  END IF;
END $$;

CREATE INDEX IF NOT EXISTS eval_jobs_claim_idx
  ON eval_jobs (status, priority DESC, id)
  WHERE status IN ('pending', 'running');

CREATE INDEX IF NOT EXISTS eval_results_rank_v2_idx
  ON eval_results (training_metadata_hash, stage, completion_rate DESC, reward_mean DESC, max_x_max DESC);
"""


RESET_SCHEMA_SQL = """
DROP TABLE IF EXISTS eval_results;
DROP TABLE IF EXISTS eval_jobs;
DROP TABLE IF EXISTS checkpoint_candidates;
"""


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


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def database_url(use_direct: bool) -> str:
    load_dotenv()
    if use_direct:
        value = os.environ.get("DIRECT_DATABASE_URL") or os.environ.get("DATABASE_URL")
    else:
        value = os.environ.get("DATABASE_URL") or os.environ.get("DIRECT_DATABASE_URL")
    if not value:
        raise SystemExit("DATABASE_URL or DIRECT_DATABASE_URL must be set in .env")
    return value


def connect(url: str):
    return psycopg2.connect(url, cursor_factory=psycopg2.extras.RealDictCursor)


def apply_schema(conn, *, reset_db: bool = False) -> None:
    with conn:
        with conn.cursor() as cur:
            if reset_db:
                cur.execute(RESET_SCHEMA_SQL)
            cur.execute(SCHEMA_SQL)


def seed_eval_jobs(
    conn,
    *,
    stage: str,
    episodes: int,
    seed_start: int,
    max_attempts: int,
) -> int:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO eval_jobs (
                  candidate_id, stage, episodes, seed_start, priority, max_attempts
                )
                SELECT
                  id,
                  %(stage)s,
                  %(episodes)s,
                  %(seed_start)s,
                  priority,
                  %(max_attempts)s
                FROM checkpoint_candidates
                ON CONFLICT (
                  candidate_id, stage, episodes, seed_start
                ) DO NOTHING
                """,
                {
                    "stage": stage,
                    "episodes": episodes,
                    "seed_start": seed_start,
                    "max_attempts": max_attempts,
                },
            )
            return cur.rowcount


def claim_one_job(conn, *, worker_id: str, lease_seconds: int) -> dict[str, Any] | None:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                CLAIM_SQL,
                {"worker_id": worker_id, "lease_seconds": lease_seconds},
            )
            row = cur.fetchone()
            return dict(row) if row else None


def release_job(conn, *, job_id: int, worker_id: str) -> int:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE eval_jobs
                SET
                  status = 'pending',
                  lease_owner = NULL,
                  lease_expires_at = NULL
                WHERE id = %(job_id)s
                  AND lease_owner = %(worker_id)s
                  AND status = 'running'
                """,
                {"job_id": job_id, "worker_id": worker_id},
            )
            return cur.rowcount


def counts(conn) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS count FROM checkpoint_candidates")
        candidates = cur.fetchone()["count"]
        cur.execute("SELECT status, COUNT(*) AS count FROM eval_jobs GROUP BY status ORDER BY status")
        jobs = {row["status"]: row["count"] for row in cur.fetchall()}
        cur.execute("SELECT COUNT(*) AS count FROM eval_results")
        results = cur.fetchone()["count"]
    return {"checkpoint_candidates": candidates, "eval_jobs": jobs, "eval_results": results}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Set up and seed the Neon eval queue.")
    parser.add_argument("--stage", default="quick")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=10007)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--lease-seconds", type=int, default=1800)
    parser.add_argument("--direct", action="store_true", help="Use DIRECT_DATABASE_URL.")
    parser.add_argument("--no-seed-jobs", action="store_true")
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Drop and recreate eval queue tables for the metadata-v2 cutoff migration.",
    )
    parser.add_argument("--smoke-claim", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    conn = connect(database_url(args.direct))
    try:
        apply_schema(conn, reset_db=args.reset_db)
        seeded = 0
        if not args.no_seed_jobs:
            seeded = seed_eval_jobs(
                conn,
                stage=args.stage,
                episodes=args.episodes,
                seed_start=args.seed_start,
                max_attempts=args.max_attempts,
            )

        smoke = None
        if args.smoke_claim:
            worker_id = f"setup-smoke-{uuid.uuid4()}"
            smoke = claim_one_job(conn, worker_id=worker_id, lease_seconds=args.lease_seconds)
            if smoke is not None:
                release_job(conn, job_id=int(smoke["id"]), worker_id=worker_id)

        print(f"seeded_jobs={seeded}")
        if smoke is not None:
            print(
                "smoke_claim="
                f"job_id={smoke['id']} "
                f"candidate_id={smoke['candidate_id']} "
                f"stage={smoke['stage']}"
            )
        print(f"counts={counts(conn)}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
