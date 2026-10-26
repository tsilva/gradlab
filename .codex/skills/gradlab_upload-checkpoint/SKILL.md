---
name: gradlab_upload-checkpoint
description: Publish GradLab checkpoints or accepted policy releases through durable Publication workflows, verify completion, and preserve immutable result identity.
---

# Upload Checkpoint

## When to use

Choose Checkpoint Publication, accepted Release, or evaluation/Promotion.
Checkpoint Publication can expose unevaluated artifacts. An accepted Release
requires exact verified goal-owned Acceptance evidence; requested evaluation uses
`$gradlab_launch-experiment`. A bare combined-model upload selects accepted Release.

## Prepare

Apply the repository specification review once for this task. Read
`docs/specs/publication.md`, `docs/specs/queued-execution.md`, and `CONTEXT.md`.
Read the requested path in [operations.md](references/operations.md).
For visual Releases, read the
[preview contract](../gradlab_upload-youtube-video/references/release-metadata.md)
and use `$model-card-author` for generated-card quality.

## Act

For completed direct local-only Runs, use `gradlab publish-local <run-directory>`.
For queued Runs, inspect their supervisor-owned public inventory. Tracker sync
requires its own request and preserves the original `complete_local` outcome.

For accepted Releases, resolve the catalog Checkpoint, open `gradlab play --run <id>`,
capture/render the current completed episode, run credential preflight, preview
metadata, and admit once. A clear publication request authorizes admission.
Retain the job ID/fingerprint. The player service/worker owns generated metadata,
resumable uploads, HF commits/tags, and final privacy. Quality fixes belong in the
renderer or verified inputs. There is no standalone headless HF-release CLI.

## Recover

Use `gradlab jobs show <job-id>` and existing-job recovery controls. Reconcile
uncertain uploads with persisted sessions/markers before any repeat submission.
Use the durable publisher rather than ad hoc HF commits or duplicate uploads.
Report unsupported input or missing evidence; cancellation/cleanup needs its own request.

## Finish

Require job success and the reference's artifact/identity checklist: immutable
HF tag/commit, admitted replay bytes, processed YouTube video with requested privacy,
thumbnail, environment containers, and cross-links. Report identities, job/fingerprint,
public links, and evidence paths. Distinguish Checkpoint Publication from accepted Release.
