# Publication

This specification applies to GradLab's current publication providers and public research indexes.

## Requirements

- Hugging Face must remain the current Policy and Release publication provider.
- YouTube must remain the current representative-replay publication provider.
- Hugging Face must use one stable model repository for each Research Goal identity.
- Goal Revisions, Goal Variants, and Policy Lineages must remain explicit within immutable Release tags instead of creating one repository for each lineage.
- Each environment-scoped Hugging Face collection and YouTube playlist must be named `GradLab — <canonical environment ID>`.
- Only a curated index that spans environments may use a name that is not an environment name.
- Each publication-enabled Run must declare a Publication Policy that selects its publishable periodic and final Checkpoints.
- Checkpoint Publication must remain independent of evaluation status and must not imply Acceptance, Promotion, or Release.
- A Release for a visual or interactive Policy must include its representative replay as root `replay.mp4` when the publication provider supports that preview.
- Public model, video, source, and telemetry surfaces for one result must link to each other and use the same stable result identity.
