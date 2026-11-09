---
name: gradlab_upload-youtube-video
description: Maintain GradLab YouTube videos, metadata, thumbnails, playlists, privacy, and OAuth access. Route accepted-release previews through the durable publisher; support requested standalone maintenance.
---

# Upload YouTube Video

## When to use

Use for existing-video maintenance or an explicitly requested standalone upload.
Accepted checkpoint-release previews use `$gradlab_upload-checkpoint` and its durable
worker. Existing release-upload uncertainty stays with its existing job.

## Prepare

Apply the repository specification review once for this task. For model previews,
read `docs/specs/publication.md` and
[release-metadata.md](references/release-metadata.md), the preview-policy authority.
Read the requested maintenance path in [operations.md](references/operations.md).

## Act

Use `scripts/upload_youtube_video.py --video-id <id>` for requested existing-video
fields. Omitted fields preserve existing values; a link-only update preserves prose.
Playlist changes use supported `YouTubeClient` methods and membership verification.
Explicit standalone uploads use the foreground utility once. Default unspecified
new-upload privacy to unlisted; use the user's title and source-backed claims.
The script owns argument semantics; the durable worker owns accepted-release uploads.

## Recover

Retain the utility process during OAuth and upload. Relay its authorization URL
without printing tokens or secrets. If the final response is lost, reconcile the
video before repeating; the utility does not persist resumable state.
Editorial maintenance must preserve verified claims and admitted identity.

## Finish

Verify saved result, video ID/link, requested fields, model link, thumbnail response,
and requested playlist membership. New uploads additionally require successful
processing. Report rejected thumbnails accurately. Use the 10-second gameplay
frame, or the last safe frame for shorter videos, unless overridden explicitly.
After uploader-code changes, run behavior tests. Future preview-policy changes
belong in the metadata reference, then renderer/tests; `AGENTS.md` supplies the trigger.
