# 03 - Modal Packaging Excludes Local Secrets

Severity: High
Confidence: High
Subagent: Euler

## Problem

`src/stable_retro_ppo/modal_core.py:77-94` packages the repo root into the Modal image with a manual ignore list. It excludes `.env` but not `.secret`, while `scripts/upload_youtube_video.py:375-376` defaults OAuth inputs to `.secret/youtube_client_secret.json` and `.secret/youtube_token.json`. Local proof showed both files exist and would be included by Modal packaging.

## Desired state

Modal images never include repo-local OAuth tokens, client secrets, or the `.secret` directory. The YouTube upload script can remain a local tool, but its credential defaults must not create deployable artifacts in Modal images.

## Implementation plan

1. In `src/stable_retro_ppo/modal_core.py`, replace the inline `ignore=[...]` with a named constant or helper such as `MODAL_REPO_IGNORE_PATTERNS`.
2. Add explicit deny patterns:
   - `.secret`
   - `.secret/**`
   - optionally `**/youtube_token.json` and `**/youtube_client_secret.json` as defense in depth.
3. Keep existing excludes unchanged: `.git`, `.env`, `.env.*`, `.venv`, `.uv-cache`, run/log/model/video output dirs, and `wandb`.
4. Prefer a small pure helper module, such as `src/stable_retro_ppo/modal_packaging.py`, if tests would otherwise need to import Modal.
5. Do not move or rename the YouTube script defaults unless product direction changes. The bug is Modal package scope, not local credential path choice.
6. Add a comment near the Modal ignore list explaining that it is a deployment boundary and must include local secret directories even when Git already ignores them.

## Tests/verification

- Add a unit test asserting the Modal ignore pattern set contains `.secret` and `.secret/**`, while `.env` and `.env.*` remain present.
- Add a focused path-matching test if a helper is introduced:
  - Excluded: `.secret/youtube_token.json`.
  - Excluded: `.secret/youtube_client_secret.json`.
  - Excluded: `.env`.
  - Included: `src/stable_retro_ppo/modal_app.py`.

Run:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m unittest tests.test_core_helpers
UV_CACHE_DIR=.uv-cache uv run python -c "import stable_retro_ppo.modal_core as m; print(m.APP_NAME)"
```

Run a Modal dry/smoke packaging command only after confirming it will not launch expensive training.

## Rollout notes

This is a source-only packaging hardening change. Existing local YouTube uploads should keep working because `.secret` remains local and is already ignored by Git.

Treat any already-built Modal image or cached package from before the fix as potentially exposed and rotate the YouTube OAuth token/client secret if that image may have been uploaded to Modal.

## Open questions

- Should YouTube OAuth credentials be rotated immediately, or only after confirming a vulnerable Modal image was built/uploaded?
- Should the repo standardize all deploy-package excludes in one helper for Modal, SkyPilot, and future providers?
- Should `.secret` be documented in README as local-only credential storage?
