# GradLab RGB trajectories

A snapshot of recorded Policy execution for trajectory and neural-emulator research.
Full RGB, including HUD pixels, is preserved as lossless PNG. Each image is stored
once in `frames`; ordered `transitions` reference source/successor frame IDs.
`episodes` includes initial frames, seeds, completion status and session IDs.
`sessions` contains collection settings and portable checkpoint/runtime provenance.
`manifest.json` records the collection contract. No checkpoint is needed to read it.

## Loading

```python
from datasets import load_dataset
repo = "YOUR_NAMESPACE/YOUR_DATASET"  # Or the prepared local dataset directory.
steps = load_dataset(repo, "transitions", split="train")
frames = load_dataset(repo, "frames", split="assets")
episodes = load_dataset(repo, "episodes", split="train")
# Join source_frame_id / successor_frame_id to frames.frame_id.
# Frame IDs are identifiers, not zero-based row offsets.
```

Each configuration is a separate table in the Hugging Face Dataset Viewer.
The `frames` table previews images; the transition table shows IDs and numerical
facts. The Hub does not automatically render frame-ID joins as image columns.
Use `(episode_id, step)` for ordering and `session_id` for provenance.
Action JSON columns preserve scalar/vector shape. Nullable fields mean unavailable.
`record_json` retains every original fact and exact NumPy dtype/shape: parse its
JSON, parse `structure` for the tagged tree, and base64-decode each `arrays[].data`
using the declared NumPy dtype/shape. Array references index that array list.
The tree tags are scalar, array, dict (key/value pairs), list, and tuple.

## Splits and limitations

Episode assignments are preserved as `train` and `heldout`; absent splits are
omitted. `frames/assets` is a shared image pool, NOT an independent training split.
Only join images referenced by your chosen episode split; sharing image bytes does
not permit fitting on held-out trajectories, labels, or histories. Incomplete
prefixes remain marked in `episodes` and must not be treated as game-over events.
Boundaries and temperature changes are recorded in session settings; Counterfactual
Playback remains ineligible as Acceptance or Promotion evidence. Capture cadence
comes from the manifest, and omitted native intermediate frames are unavailable.
Image uniqueness does not measure hidden simulator-state coverage.

The collector does not assign a dataset license. Set the applicable license and
source attribution in this card before publication. `upload.json` identifies the
snapshot and hashes its files; this is collection evidence, not model performance.
