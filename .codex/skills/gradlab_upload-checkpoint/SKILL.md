---
name: gradlab_upload-checkpoint
description: Publish GradLab checkpoints or accepted policy releases, verify Hugging Face replays and collections, and create or refresh game dashboards.
---

# Upload Checkpoint

## When to use

Choose Checkpoint Publication, accepted Release, or evaluation/Promotion.
Checkpoint Publication can expose unevaluated artifacts. An accepted Release
requires exact verified goal-owned Acceptance evidence; requested evaluation uses
`$gradlab_launch-experiment`. A bare combined-model upload selects accepted Release.
For standalone game dashboard creation or maintenance, use the
[dashboard procedure](references/dashboard.md).

## Prepare

Apply the repository specification review once for this task. Read
`docs/specs/publication.md`, `docs/specs/queued-execution.md`, and `CONTEXT.md`.
Read the requested path in [operations.md](references/operations.md).
For Hugging Face model cards, use `$private-skills_model-card-author` for the
compact Results, Quick start, and Provenance layout. Check generated cards
against that guidance through the renderer and verified inputs. For visual
Releases, also read the
[preview contract](../gradlab_upload-youtube-video/references/release-metadata.md).

## Act

For completed direct local-only Runs, use `gradlab publish-local <run-directory>`.
For queued Runs, inspect their supervisor-owned public inventory. Tracker sync
requires its own request and preserves the original `complete_local` outcome.

For a visual or interactive Checkpoint or Release going to a Hugging Face model
repo, include a representative episode from the exact Policy as root `replay.mp4`
in the verified bundle. For game uploads, add the model repo to the owner's
canonical environment collection, creating it if absent.
Create or refresh the game dashboard after verifying the model revision.
Prepare its update through the [dashboard procedure](references/dashboard.md).
Follow the [model upload procedure](references/operations.md#publish-a-model-on-hugging-face)
for authorization, transport, and verification. When the applicable outbound
policy authorizes the selected Checkpoint upload from the owner's explicit
request, proceed after verification without a separate draft approval.

For accepted Releases, resolve the catalog Checkpoint, open `gradlab play --run <id>`,
capture/render the current completed episode, run credential preflight, preview
metadata, and admit once after the required public draft approval.
Retain the job ID/fingerprint. The player service/worker owns generated metadata,
resumable uploads, HF commits/tags, and final privacy. Quality fixes belong in the
renderer or verified inputs. There is no standalone headless HF-release CLI.

## Recover

Use `gradlab jobs show <job-id>` and existing-job recovery controls. Reconcile
uncertain uploads with persisted sessions/markers before any repeat submission.
Use the durable publisher rather than ad hoc HF commits or duplicate uploads.
Report unsupported input or missing evidence; cancellation/cleanup needs its own request.

## Finish

For Hugging Face model uploads, verify the published revision, the loaded Video
Preview on the model page, and game collection membership where applicable.
For game uploads, verify the dashboard row and the Space's first position in the
collection. Report a verified model publication and any blocked dashboard update
separately so a retry can reconcile the existing model.
For accepted Releases, also require job success and the reference's Release
checklist. Report the result identity, public links, and applicable job and
evidence details. Distinguish Checkpoint Publication from accepted Release.
