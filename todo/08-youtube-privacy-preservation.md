# 08 - YouTube Privacy Preservation On Updates

Severity: Low
Confidence: High
Subagent: Plato

## Problem

In `scripts/upload_youtube_video.py:402-405`, `--privacy-status` defaults to `unlisted`. On the existing-video update path, `scripts/upload_youtube_video.py:429-445` tries to preserve the existing privacy status, but `args.privacy_status` is always populated by argparse. Updating metadata for an existing private video without explicitly passing `--privacy-status private` changes it to `unlisted`.

## Desired state

Existing-video updates preserve the current YouTube privacy status unless the caller explicitly passes `--privacy-status`.

New uploads continue to default to `unlisted`.

Explicit `--privacy-status private|public|unlisted` keeps working for both upload and update paths.

If an existing video's current privacy status cannot be read, the script fails with a clear error instead of silently defaulting to `unlisted`.

## Implementation plan

1. In `build_parser()`, remove the argparse default from `--privacy-status` so omitted means `None`.
2. Add a constant near the YouTube constants:

```python
DEFAULT_UPLOAD_PRIVACY_STATUS = "unlisted"
```

3. In the `--video-id` update path, compute privacy as:

```python
privacy_status = args.privacy_status or existing_status.get("privacyStatus")
```

If `privacy_status` is missing, raise `SystemExit` telling the user the existing privacy status could not be read and they should pass `--privacy-status`.

4. Pass the computed value to `update_video_metadata()`.
5. In the new-upload path, compute:

```python
upload_privacy_status = args.privacy_status or DEFAULT_UPLOAD_PRIVACY_STATUS
```

Use it for `start_resumable_upload()` and `find_or_create_playlist()`.

6. Update CLI help to explain that omitted privacy preserves existing privacy for `--video-id` and defaults to `unlisted` for new uploads.

## Tests/verification

Add `tests/test_upload_youtube_video.py` with network-free monkeypatched tests:

- Parser behavior:
  - `build_parser().parse_args(["--video-id", "abc"]).privacy_status is None`.
  - Explicit `--privacy-status private` remains `private`.
- Existing-video update preserves private:
  - Mock `load_client_config`, `access_token`, `get_video_metadata`, `update_video_metadata`, and `save_json`.
  - Fake metadata has `status.privacyStatus = "private"`.
  - Assert `update_video_metadata()` receives `privacy_status="private"`.
- Existing-video update honors explicit override:
  - Pass `--privacy-status unlisted`.
  - Assert `privacy_status="unlisted"`.
- New upload default remains unlisted:
  - Mock file checks and upload functions.
  - Run upload path without `--privacy-status`.
  - Assert `start_resumable_upload()` receives `privacy_status="unlisted"`.
- Missing existing privacy fails closed:
  - Fake existing metadata without `status.privacyStatus`.
  - Assert `SystemExit` and no `update_video_metadata()` call.

Run:

```bash
uv run pytest tests/test_upload_youtube_video.py
uv run ruff check scripts/upload_youtube_video.py tests/test_upload_youtube_video.py
```

## Rollout notes

This is a backward-compatible safety fix for explicit callers. Any automation already passing `--privacy-status` behaves the same.

The only behavior change is for existing-video updates that omit `--privacy-status`: they preserve current privacy instead of defaulting to `unlisted`.

No migration is needed. After merge, retry one metadata-only update against a known private test video and confirm it remains private in the returned API payload or YouTube Studio.

## Open questions

- Should metadata-only updates also preserve the existing description when no description fields are provided?
- Should `private` become the default for newly uploaded model-preview videos, or is the current new-upload default intentionally shareable by link?
