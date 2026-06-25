# 02 - Training Resume Trust Gate

Severity: High
Confidence: Medium
Subagent: Ptolemy

## Problem

Training resume inputs can flow into unsafe SB3/PyTorch deserialization:

- `src/rlab/train_runner.py:54-65` converts `resume_artifact` into a downloaded local `resume` path.
- `src/rlab/modal_train.py:104` accepts `resume`, `resume_artifact`, and `auto_resume_latest`, then forwards the resolved path.
- `src/rlab/train.py:84-86` passes `args.resume` directly to `PPO.load()`.

`PPO.load()` deserializes SB3 checkpoint archives. Treat checkpoint files and downloaded W&B artifacts as code-bearing inputs unless provenance and integrity are explicitly trusted.

## Desired state

No training code path calls `PPO.load()` for a resume unless the checkpoint passes one centralized trust gate.

Accepted resume sources are limited to project-produced model artifacts/checkpoints with current metadata, expected W&B entity/project provenance, a `.zip` model file, and integrity checks. Arbitrary local paths or cross-project W&B artifact refs fail before launching local subprocesses or Modal workers, unless the project intentionally adds a clearly named unsafe operator override.

## Implementation plan

1. Add `src/rlab/resume_trust.py`:
   - `validate_resume_path(path, allowed_roots, require_metadata=True) -> Path`.
   - Resolve paths; reject missing files, non-`.zip` files, symlink escapes, directories, and paths outside allowed roots.
   - Use existing metadata helpers from `artifacts.py` to verify metadata version, training metadata, and training metadata hash.
   - Add SHA-256 helpers and compare against a new metadata field such as `model_sha256` when present.
2. Extend model artifact metadata:
   - Add `model_sha256` to model metadata construction and W&B artifact metadata.
   - Ensure downloaded artifact sidecars keep that field.
   - Keep old metadata parsing tolerant, but make training resume validation strict.
3. Add a trusted W&B resume download helper:
   - Fetch the W&B artifact object.
   - Require `type == "model"`.
   - Require an allowed entity/project from config/env.
   - Require expected metadata fields.
   - Download, write sidecar, validate local path and hash.
   - Reject malformed refs and refs outside the configured project before download where possible.
4. Wire the trust gate into local campaign workers:
   - In `normalize_train_config()`, validate existing `resume` values before command construction.
   - Replace the `resume_artifact` branch with the trusted W&B helper.
   - Preserve the existing `resume` versus `resume_artifact` conflict error.
5. Wire the trust gate into Modal training:
   - In `train_remote()`, validate raw `resume`, trusted downloaded `resume_artifact`, and `_latest_checkpoint(run_name)` before adding `train_options["resume"]`.
   - Use Modal allowed roots such as `RUNS_DIR` and the W&B artifact cache root.
   - Keep validation inside the remote worker too, not only in the local entrypoint.
6. Reduce direct sink usage:
   - In `train.py`, replace direct `PPO.load(args.resume, ...)` with `load_trusted_resume_model(...)`.
   - The helper should call validation immediately before `PPO.load()`, covering direct CLI use.

## Tests/verification

- Add unit tests for the trust module: valid checkpoint plus metadata, missing metadata, bad metadata hash, SHA mismatch, non-zip file, symlink escape, outside-root path, and missing file.
- Update campaign runner resume tests so `resume_artifact` succeeds only when the mocked download returns a checkpoint accepted by the trust gate.
- Add tests that `normalize_train_config()` rejects unsafe raw `resume` paths and disallowed W&B refs.
- Extract Modal resume resolution into a helper if needed, then test `resume`, `resume_artifact`, `auto_resume_latest`, and conflict behavior without launching Modal.
- Add a light `train.py` test around the new loader helper, monkeypatching `PPO.load()` to prove validation runs first.

Run:

```bash
uv run pytest tests/test_campaign_runner.py tests/test_core_helpers.py tests/test_resume_trust.py
rg -n "PPO\\.load\\(args\\.resume|download_model_artifact\\(.*resume" src tests
```

## Rollout notes

This will likely break resuming from older checkpoints that lack sidecar metadata or hashes. Provide a migration path: re-upload/re-log known-good checkpoints with current metadata, or add a temporary explicit `--allow-legacy-resume` only if the owner accepts the residual pickle risk.

Do not delete existing artifacts during remediation. First validate with a known current checkpoint locally, then with one Modal smoke resume, then enable stricter campaign-worker behavior.

Adjacent `PPO.load()` use in eval/play scripts is visible but outside this issue. Track it separately so the training fix stays small.

## Open questions

- Which W&B entity/project names are allowed for trusted training resumes?
- Should arbitrary local `--resume` remain supported for trusted operators, or require an explicit unsafe flag?
- Do current promoted checkpoints already have sidecar metadata everywhere needed for resume workflows?
- Should the same trust gate be applied next to eval, playback, distillation, and hash scripts?
