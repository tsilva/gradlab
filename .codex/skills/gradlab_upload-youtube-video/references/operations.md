# Upload YouTube Video

Distinguish an accepted Checkpoint Release from standalone video maintenance.
Apply the [entry skill](../SKILL.md)'s preparation;
[release-metadata.md](release-metadata.md) owns model-preview policy.

## Checkpoint-release previews

Use `$gradlab_upload-checkpoint` and the existing player Publication worker. Do not use
`scripts/upload_youtube_video.py` to upload a second release preview. The worker
owns generated metadata, admitted capture identity, resumable sessions,
processing verification, HF cross-links, environment containers, thumbnails,
and final privacy. The script is a foreground maintenance utility and does not
provide that durable release protocol.

For an existing publication, retain its job ID and request fingerprint. Inspect
`gradlab jobs show <job-id>` or the player's publication status. If upload
completion is uncertain, reconcile the persisted session and marker through the
worker; use its verified video-ID resolution when appropriate. Never resolve
uncertainty by automatically uploading a duplicate.

## Existing-video maintenance

Use `scripts/upload_youtube_video.py --video-id <id>` for the explicitly
requested fields. Omitted title, tags, description, and privacy preserve existing
values. Do not pass an example's `public` privacy or generic title inadvertently.

```bash
uv run python scripts/upload_youtube_video.py \
  --video-id <youtube-video-id> \
  --model-page "Model: https://huggingface.co/<owner>/<repo>/tree/<vN>" \
  --output ~/.config/gradlab/runs/<artifact-dir>/youtube_update_result.json
```

A link-only update preserves the existing descriptive content. For a generated
release preview, first determine whether the durable request still owns its
metadata. Requested editorial maintenance must not alter verified claims or
break admitted identities; fix generated evidence-bearing metadata through its
owning project code/workflow.

`--video-id` updates do not add playlist membership. For a requested playlist
operation, use the supported `YouTubeClient` methods in
`src/gradlab/youtube_publication.py` and verify membership. Environment previews
belong to `GradLab — <canonical environment ID>`; do not invent a generic
`gradlab` playlist or revive the retired Featured Research option.

## Standalone upload

For an explicitly requested standalone local video, the foreground utility may
upload once with the user's title and privacy. Default new uploads to unlisted
when privacy was not specified. Use the canonical environment playlist for
model previews, and include only source-backed claims. Keep the process handle
through completion; the utility does not persist resumable upload state, so a
lost final response requires reconciliation before any repeat upload.

```bash
uv run python scripts/upload_youtube_video.py <video.mp4> \
  --title "<requested title>" \
  --privacy-status unlisted \
  --playlist-title "GradLab — <canonical environment ID>" \
  --output ~/.config/gradlab/runs/<artifact-dir>/youtube_upload_result.json
```

An acknowledged upload is not successful video processing. Read
`YouTubeClient.video_metadata`, including `processingDetails`, to verify
`processingStatus=succeeded`. A standalone upload creates no accepted Release
and authorizes no HF mutation. Operator-requested metadata updates may use an
existing video without re-uploading bytes.

## Credentials, thumbnails, and verification

The utility reads OAuth files from GradLab's user configuration directory;
never print client secrets or tokens. For player uploads, use the player's
OAuth controls. For a foreground utility needing authorization, retain the
waiting process and relay its printed URL; use `--no-browser` when appropriate.
Continue the same flow after authorization instead of starting another one.

Gameplay thumbnails use the 10-second frame, or the last safe frame for shorter
videos. Explicit thumbnail overrides are supported; disclose a rejected custom
thumbnail rather than claiming success.

Verify the saved result, video ID/URL, requested fields, direct model link,
processing state for new uploads, thumbnail response, and playlist membership
when requested. Report those outcomes and the result path. After utility code
changes, run its meaningful uploader tests; syntax compilation alone does not
verify upload or metadata behavior.

Encode future preview-policy changes here or in the linked metadata reference,
then align the generated renderer and its tests. `AGENTS.md` remains a trigger
pointer, not another copy of the policy.
