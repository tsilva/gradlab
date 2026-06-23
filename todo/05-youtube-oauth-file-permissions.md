# 05 - YouTube OAuth File Permissions

Severity: Medium
Confidence: High
Subagent: Gibbs

## Problem

`scripts/upload_youtube_video.py` persists OAuth material with default filesystem permissions. `save_json()` uses `Path.write_text()`, so new or refreshed `.secret/youtube_token.json` can end up `0644` under a `0755` `.secret/` directory. The token stores sensitive OAuth material, and the default client secret path is read without checking permissions.

Affected files:

- `scripts/upload_youtube_video.py:97-99`
- `scripts/upload_youtube_video.py:142-145`
- `scripts/upload_youtube_video.py:165-170`
- `scripts/upload_youtube_video.py:375-376`

## Desired state

`.secret/` is `0700`; OAuth client secret and token files are regular, owner-only files with mode `0600`. The uploader creates or repairs owned secret paths securely, refuses unsafe symlink paths, and fails closed if it cannot make permissions private. Non-secret output JSON keeps existing behavior.

## Implementation plan

1. Add helpers near `save_json()`:
   - `private_mode(path) -> int`, using `stat.S_IMODE(path.stat().st_mode)`.
   - `ensure_private_dir(path: Path)`, creating missing parent dirs with `0o700` and repairing owned group/other bits.
   - `ensure_private_file(path: Path, label: str)`, rejecting symlinks, requiring a regular file, and applying `0o600` when owned by the current user.
   - `save_private_json(path, payload)`, using `os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)` plus `os.fdopen(...)`, then enforcing `chmod(0o600)`.
2. Keep `save_json()` for non-secret outputs such as `runs/youtube_upload_result.json`.
3. Change token writes to call `save_private_json(token_path, token)`.
4. In `access_token()`:
   - Before reading an existing token file, call `ensure_private_file(token_path, "OAuth token")`.
   - Before authorizing a new token, call `ensure_private_dir(token_path.parent)`.
5. In `main()`, before `load_client_config(args.client_secret)`:
   - Call `ensure_private_file(args.client_secret, "OAuth client secret")`.
   - Optionally call `ensure_private_dir(args.token.parent)` early.
6. Leave parser defaults at `.secret/youtube_client_secret.json` and `.secret/youtube_token.json`.

## Tests/verification

Add `tests/test_upload_youtube_video_permissions.py`:

- `save_private_json()` creates a new token file as `0600` even under a permissive umask.
- `save_private_json()` repairs an existing `0644` token file to `0600`.
- `ensure_private_file()` repairs an owned `0644` client secret file to `0600`.
- `ensure_private_dir()` repairs an owned `0755` `.secret` dir to `0700`.
- Symlink token/client-secret paths raise `SystemExit`.
- Existing non-secret `save_json()` behavior remains usable.

Run:

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_upload_youtube_video_permissions.py
UV_CACHE_DIR=.uv-cache uv run python -m py_compile scripts/upload_youtube_video.py
stat -f '%Lp %N' .secret .secret/youtube_client_secret.json .secret/youtube_token.json
```

Expected final local modes:

```text
700 .secret
600 .secret/youtube_client_secret.json
600 .secret/youtube_token.json
```

## Rollout notes

Before or after merging, repair the currently observed local files:

```bash
chmod 700 .secret
chmod 600 .secret/youtube_client_secret.json .secret/youtube_token.json
```

If these files were exposed through shared storage, backups, logs, or another user account, revoke the YouTube OAuth grant and regenerate the token.

## Open questions

- Should the script auto-repair owned client secret files, or fail and require a manual `chmod`?
- Should unsafe parent directories above `.secret/` be checked too?
- Should token writes be atomic with a temp file plus `os.replace()`, or is direct truncate/write acceptable for this single-user CLI?
