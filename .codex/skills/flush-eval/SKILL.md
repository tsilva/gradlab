---
name: flush-eval
description: Flush sandbox-sb3 Modal eval jobs for checkpoint candidates and report ranked results. Use when the user asks to evaluate unevaluated checkpoints, flush the eval queue, run pending Modal evals, check what checkpoints are not evaluated yet, or produce a post-eval report from the Neon eval database.
---

# Flush Eval

## Contract

Evaluate checkpoint candidates whose artifacts carry current training metadata, then report queue status and ranked results from the eval database.

Default eval protocol unless the user says otherwise:

- `stage=quick`
- `episodes=100`
- `seed_start=10007`
- Modal resources: `--cpu 1 --memory-mib 4096`
- Modal runners: `--runners 1`
- Per-runner checkpoint lookahead: `--prefetch-jobs 2`

This skill is for eval-only work. Do not launch training, upload model cards, publish checkpoints, or mutate candidate selection beyond seeding missing eval jobs.

Before any Modal run, estimate expected cost from the current pending job count, episodes, runner/resource shape, and observed or benchmarked runtime. Ask the user whether to run and state the approval cap. Do not launch Modal workers until the user approves.

At the end of every Modal run, report actual cost. If actual cost is unavailable from provider data, estimate it from worker elapsed seconds and Modal CPU/memory rates, and say that it is an estimate. If actual cost differs significantly from the approved estimate, investigate the root cause and fix the process before considering the task complete. Treat a discrepancy as significant when it is both more than 25% and more than $0.10, unless the user supplied a stricter threshold.

## Workflow

1. Read `AGENTS.md` and `INSTANCES.md`.
   - Confirm no newer project rule overrides the defaults above.
   - Use `.env` for `DATABASE_URL`, `DIRECT_DATABASE_URL`, W&B, and R2 credentials.

2. Inspect current DB state before launching:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/setup_neon_eval_queue.py \
  --stage quick \
  --episodes 100 \
  --seed-start 10007 \
  --no-seed-jobs
```

3. Seed missing jobs for all current `checkpoint_candidates`:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/setup_neon_eval_queue.py \
  --stage quick \
  --episodes 100 \
  --seed-start 10007
```

If the user requested a different stage, episode count, or seed, substitute it consistently in every command and in the final report.

For the metadata-v2 cutoff migration only, the user may explicitly approve purging existing eval queue data:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/setup_neon_eval_queue.py \
  --reset-db \
  --stage quick \
  --episodes 100 \
  --seed-start 10007
```

4. Estimate Modal cost before launching and get explicit approval.
   - Recompute the estimate after seeding jobs, because seeding can change the pending count.
   - Default `--cpu 1 --memory-mib 4096 --runners 1` is the cost-effective worker shape.
   - If the user asks to flush faster, scale by increasing `--runners`; do not raise CPU per runner by default.
   - Preserve `--prefetch-jobs 2` unless the user asks otherwise. This makes each runner hold up to two leased jobs: one active eval and one sequential background checkpoint download. Account for the extra held lease when choosing `--lease-seconds`; the default `1800` is intended to cover normal download plus eval time.
   - Base the estimate on observed eval runtimes for the same stage/episode count when the database has finished jobs. If none exist, use the closest benchmark from `INSTANCES.md` and call out the assumption.
   - Mention that the eval queue writes results to Neon and downloads W&B/R2 model artifacts, with per-runner local prefetch rather than a shared checkpoint volume.
   - Ask the user whether to start the Modal run and state the maximum approved cost cap.

5. Run the Modal queue until idle:

```bash
.venv/bin/modal run src/stable_retro_ppo/modal_app.py::eval_queue \
  --runners 1 \
  --cpu 1 \
  --memory-mib 4096 \
  --max-jobs-per-runner 0 \
  --idle-polls 2 \
  --idle-sleep-seconds 5 \
  --lease-seconds 1800 \
  --device cpu \
  --prefetch-jobs 2
```

Use `UV_CACHE_DIR=.uv-cache uv run modal ...` only if the local `.venv/bin/modal` is unavailable.

6. Generate the final report:

```bash
python .codex/skills/flush-eval/scripts/eval_report.py \
  --stage quick \
  --episodes 100 \
  --seed-start 10007
```

The report should include:

- total candidates
- job status counts
- result count for the requested stage/episode/seed selection
- remaining pending/running/failed jobs
- top ranked checkpoints by completion rate, reward mean, then max x-position
- eval runtime mean/std when finished job timestamps are available
- artifact download and prefetch wait timing when present in `metrics_json`

7. Compute and report actual Modal cost.
   - Prefer provider-reported usage/cost when available.
   - If provider-reported cost is not available, estimate actual cost from the Modal worker `elapsed_seconds`, `--cpu`, `--memory-mib`, and current Modal listed rates used for the pre-run estimate.
   - Compare actual cost with the approved estimate and cap.
   - If the discrepancy is significant, identify the root cause before ending the task. Check for missing/incomplete checkpoint training metadata, changed pending job count, slower-than-expected episode length, runner/resource shape mismatch, prefetch depth mismatch, retries/failures, lease expiry, local client disconnects, and stale cost-rate assumptions.
   - If the root cause is a repo or skill issue, patch it. If the fix requires changing queue behavior or billing assumptions, explain the change and verify it with a small safe check before launching more paid work.

## Safety

- Do not print `.env` values or database URLs.
- Do not reset failed jobs unless the user asks for retries.
- Do not change `checkpoint_candidates` selection logic unless the user asks to repopulate candidates.
- Do not launch Modal workers without explicit cost approval in the current turn.
- Do not use terminal-on-life eval as the default for this skill; the default is explicitly no terminal on life.
- Keep generated reports and scratch outputs under ignored paths such as `runs/`.

## Final Response

Report the exact stage, episode count, seed, runner/resource shape, prefetch depth, number of jobs evaluated, remaining job counts, top results, training metadata hash when useful, estimated cost, approved cap, actual cost, and any cost discrepancy/root-cause/fix. Include the remote-provider retrospective required by `AGENTS.md`.
