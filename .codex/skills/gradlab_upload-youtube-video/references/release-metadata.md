# Research Release Preview Contract

Project code generates the evidence-bearing release metadata in
`gradlab.player_publication.generated_metadata`. Treat that output as locked
within an admitted request; correct its sources or renderer rather than
hand-authoring a competing description.

## Identity and content

- Use the generated goal repository and next sequential immutable HF `vN` tag.
  Both platforms publish the same admitted representative `replay.mp4`.
- The title identifies the materialized goal, actual trainer, algorithm, and
  compact environment-step count. Use the canonical environment/goal fallback
  if required by the provider limit; fail rather than truncate identities.
  Keep Acceptance and replay outcomes out of the title.
- The description identifies the exact Run/Checkpoint and evaluation
  protocol/count, all Acceptance outcomes, applicable ranking metrics, and the
  separately labeled representative replay result. Playback is not evaluation
  evidence. Do not substitute a generic win-rate claim for goal-defined metrics.
- Include direct immutable HF, public Run/telemetry or applicable service, R2
  manifest, source/recipe, project, environment collection, and playlist links.
  Avoid shorteners. New-schema public telemetry comes from the verified journal;
  do not require private MLflow access to read public evidence.
- Use human display names where generated, and at most three visible hashtags:
  reinforcement learning, algorithm, and game. Hidden tags identify the actual
  backend/provider; SB3 and Stable Retro tags apply only when used.
- Optional editorial text is labeled `Operator note (not verified evidence)`.
  It cannot change verified outcomes, identities, links, or admitted replay bytes.

## Admission and completion

- Admit only the current completed episode after replay encoding succeeds.
  Beginning a new episode invalidates a prior capture for a new request.
- Freeze channel, requested final privacy, metadata, replay hashes, and the
  hidden `gradlab-publication-<request-fingerprint>` marker at admission.
- Persist the resumable session before uploading bytes. Reconcile uncertainty
  through the existing request/session; never blindly repeat an upload.
- Upload unlisted and require successful processing plus admitted channel and
  metadata verification before any HF mutation.
- Complete and verify HF, the environment collection and playlist, cross-links,
  and thumbnail before changing to the requested final privacy.
- Environment containers are named `GradLab — <canonical environment ID>`.
  Preserve historical Releases and independent contributions. The current
  player does not admit Featured Research curation or destination opt-outs.

Hugging Face previews use root `replay.mp4`; do not duplicate that video in the
card body when the reinforcement-learning widget is available. YouTube processing
or replay success cannot establish Acceptance or Promotion.
