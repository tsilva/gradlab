# 01 - Modal Eval Secretless Deserialization

Severity: High
Confidence: Medium
Subagent: Confucius

## Problem

Modal eval workers claim DB-selected jobs, resolve `eval_config.artifact_ref`, `model_artifact`, or `model_path`, then call `PPO.load()` on the selected file while `eval_queue_secret` is mounted. SB3 model loading can deserialize pickle content, so a malicious or compromised queue-selected artifact could execute inside a process with `DATABASE_URL`, `WANDB_API_KEY`, and R2/AWS credentials.

Affected files:

- `src/stable_retro_ppo/modal_core.py:45-55`
- `src/stable_retro_ppo/modal_eval.py:162-174`
- `src/stable_retro_ppo/modal_eval.py:196`
- `src/stable_retro_ppo/modal_eval.py:319-325`

## Desired state

No Modal code path that calls `PPO.load()` for a queue-selected or user-supplied eval artifact has DB, W&B, AWS, or R2 secrets mounted or cached in its environment. Modal eval treats artifact refs and paths as untrusted until validated, then evaluates only a sanitized cached artifact path in a secretless worker.

## Implementation plan

1. Split the broad `eval_queue_secret` in `modal_core.py` into least-privilege secrets:
   - DB-only secret for claiming, heartbeating, and committing eval jobs.
   - Artifact-fetch-only secret for W&B/R2 downloads.
   - No secret for model deserialization and episode evaluation.
2. Refactor `modal_eval.py` into explicit phases:
   - `eval_worker_remote`: DB secret only; claims jobs and commits results.
   - `fetch_eval_artifact_remote`: W&B/R2 secret only; validates artifact ref, downloads the zip plus metadata to a non-secret cache path, commits the volume, and returns path plus sha256.
   - `evaluate_cached_model_remote`: no secrets; calls `PPO.load()` and runs episodes.
3. Do not rely on `os.environ.pop()` before `PPO.load()`. Use a separate Modal function/container boundary for the secretless evaluator.
4. Add Modal-specific model selector validation before fetch:
   - Require exactly one model selector.
   - Prefer `artifact_ref`; reject `model_path` for remote Modal queue unless a trusted path allowlist exists.
   - Restrict W&B refs to the expected entity/project and `type="model"`.
   - If `train_job_id` is present, verify the selected artifact appears in that training job's recorded artifact refs.
   - Require downloaded artifacts to contain exactly one `.zip` and training metadata; fail closed on missing metadata.
5. Move W&B cache/config/data dirs used during fetch out of the shared eval volume, or purge them before committing. The secretless evaluator should only see ROMs plus copied model zip, metadata, and hash.
6. Preserve local trusted eval behavior unless intentionally changing local CLI semantics. Keep the stricter policy scoped to Modal queue/benchmark paths.

## Tests/verification

- Add unit tests for Modal eval config validation:
  - Reject `model_path` in remote queue.
  - Reject both `artifact_ref` and `model_path`.
  - Reject external or unexpected W&B projects.
  - Reject artifact refs not tied to `train_job_id` when linkage exists.
  - Reject missing metadata and multi-zip artifacts.
- Add a regression test that monkeypatches `PPO.load()` in the secretless evaluator and asserts `DATABASE_URL`, `WANDB_API_KEY`, `AWS_ACCESS_KEY_ID`, and `AWS_SECRET_ACCESS_KEY` are absent.
- Add a static/AST test: any Modal function that calls `PPO.load()` must have no `secrets=[...]` decorator.

Run:

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_campaign_runner.py tests/test_modal_eval_security.py
```

Smoke one safe Modal eval job after deployment and confirm it still commits metrics.

## Rollout notes

Pause Modal eval workers before rollout. Inspect pending `eval_jobs` for `model_path` or unexpected artifact refs and cancel/requeue anything that fails the new policy.

Purge old Modal/shared artifact caches that may contain unvalidated artifacts or W&B cache state. After deployment, run one known-good artifact through the new fetch/evaluate split before restoring normal runner count.

Rotate W&B, R2/AWS, and eval DB credentials if any untrusted or cross-project artifact may already have been evaluated under the old broad-secret worker.

## Open questions

- Who can insert or update `eval_jobs.eval_config` in the campaign DB?
- Should Modal queue support `model_path` at all, or should it be local-only?
- What exact W&B entity/project allowlist should Modal eval accept?
- Can the DB credential be reduced to claim/heartbeat/finish permissions only?
- Should benchmark evals use the same secretless evaluator path as queue workers?
