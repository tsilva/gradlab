# Player episode trajectories

Use **Download episode** in Player to save the current episode as `episode.gradtraj`.
Capture starts automatically when an exact Checkpoint is loaded. Downloading fixes
an immutable transition cutoff; live Playback may continue while the archive is
prepared. A completed episode remains available until another episode or
Checkpoint replaces it. An unfinished download does not invent a termination.

Use **Import episode** to open an archive in Player, including from its source
selection screen. Imported Playback reads stored images, Policy inputs, decisions,
and transition facts. It does not construct an environment, require a ROM, load
or execute the bundled Checkpoint, or resolve remote artifacts. Use the transition
number, previous/next buttons, timeline, play/pause, and speed setting to navigate.
Other viewers in the same Playback Session share the imported cursor.

The recording preserves Faithful Playback, Evaluation Reproduction, and
Counterfactual Playback classifications. Inspection is never Acceptance,
Promotion, or Training Success evidence. Heavy attribution maps, gradients, and
activation tensors are omitted even when their live panels were enabled. Panels
distinguish missing recordings, unsupported Policy capabilities, and scientific
incomparability. Recording requests no additional Policy computations.

## Archive layout and versioning

`.gradtraj` is a ZIP container with these exact members:

```text
manifest.json
metadata.json
checkpoint/model.zip
checkpoint/model.json
checkpoint/recipe.json
data/train-00000-of-00001.parquet
```

Version 1 uses a JSON manifest containing `format_version: 1` and a `files`
mapping from relative member name to its byte `size` and hex `sha256`. The
manifest inventories every other member. The checkpoint file is the exact
immutable file staged for the active Playback Session. Its existing model and
recipe documents bind its identity, configuration, Research Goal, producer
provenance, and required execution contracts. Archive inspection never invokes a
Checkpoint deserializer. Integrity hashes establish byte consistency, not trust
in an archive's author or scientific claims.

`metadata.json` contains an independent episode UUID, version, first captured
step, transition count, original episode number and seed, classification,
`scientific_evidence: false`, `complete`, resolved environment configuration,
Playback contract, execution/runtime provenance, discount factor and semantics,
and initial presentation.
`complete` means an actual episode boundary was captured. It does not mean the
environment succeeded. A first step greater than one identifies an absent prefix.
Original training contracts remain in `checkpoint/recipe.json`; active Playback
configuration, including permitted boundary overrides, is in metadata.
Credential-like configuration keys and private paths/locations are omitted from
recorded presentation and transition metadata. Existing portable model-document
validation applies to checkpoint sidecars.

## Parquet features

The Arrow schema carries `gradlab.trajectory.version = 1`. Each ordered transition
is one row and one indexed row group, compressed with Zstandard. Physical row
order is authoritative. Step numbers remain those of the original episode;
`sequence` remains the original Playback Session decision sequence.

| Columns | Arrow type | Meaning |
| --- | --- | --- |
| `episode_id` | string | Joins the episode metadata |
| `sequence`, `step`, `seed` | int64 | Original decision order, episode step, start seed |
| `start_id`, `action_source` | nullable string | Recorded task/start identity and action producer |
| `reward`, `return` | float64 | Policy-facing step reward and undiscounted episode return |
| `terminated`, `truncated`, `boundary` | bool | Original environment/task boundary facts |
| `classification` | string | Scientific classification at this decision |
| `next_observation_status`, `after_image_status` | string | `available`, `terminal_missing`, or `unavailable` |
| `observation`, `next_observation` | structured value | Exact before/after Policy inputs |
| `selected_action`, `executed_action` | structured value | Policy selection versus action sent to the provider |
| `before_image`, `after_image` | structured value | Lossless rendered images, separate from Policy inputs |
| `observation_frames` | structured value | Input display planes for this decision |
| `policy_outputs` | structured value | Small outputs already computed, preserving absent values |
| `facts` | structured value | Provider info, tasks, available reward/event diagnostics, bootstrap facts |
| `presentation` | string containing JSON | Versioned Player projection, including per-transition diagnostic availability |

A structured value is an Arrow struct with `structure: string` and
`arrays: list<struct<dtype: string, shape: list<int64>, data: binary>>`. `structure`
is a tagged JSON tree: `["dict", [[key, child], ...]]`, `["tuple", [child, ...]]`,
`["list", [child, ...]]`, `["scalar", value]`, or `["array", leaf_index]`. Numeric
leaves use NumPy dtype strings (including byte order), original dimensions, and
C-order bytes. No pickle or executable schema is used to decode these values.
Dictionary keys, tuple/list topology, exact numeric representations, and batch
axes are preserved. Do not flatten structured Policy inputs or normalize them a
second time. The checkpoint/environment contracts describe their preprocessing.

At boundaries, `next_observation` contains the provider's terminal observation
when available, with the original Policy conditioning. It never substitutes the
next episode's reset observation. A missing terminal observation or image is
null and explicitly marked `terminal_missing`. Discounting applies per Policy
decision, not per rendered frame; action-repeat/frame-skip configuration and
available runtime facts remain in the recorded environment contract and facts.
Termination stops the return; truncation requires the declared terminal bootstrap
when used for value comparison. A recording cutoff has neither effect by itself.

## Local storage and limits

Runtime capture uses an append-only disk stream and a fixed-width index; Parquet
encoding happens during download preparation. The pending write budget is 64 MiB
plus at most one transition of up to 32 MiB. Temporary serialization copies and
the existing bounded Player history are additional memory. If a write fails or
the budget is exhausted, Playback pauses visibly at a decision boundary and keeps
the captured prefix. **Retry recording** retries a failed write before advancing.
An unencodable transition remains held and blocks further capture; the accepted
prefix can still be downloaded, and replacing the episode discards that failed
capture explicitly.

Downloads pin the Checkpoint and cutoff independently of active source changes.
At most two prepared/in-flight download archives are retained. Unclaimed download
capabilities expire after five minutes on the next preparation or shutdown;
completed/cancelled transfers remove their temporary files. Old episode stores
are retired on replacement; Player shutdown removes active temporary storage.
Imported recordings use disk indexing, so seek does not load the full episode.

Import validates member names, duplicates, hashes, model/recipe bindings, version,
Arrow features, numeric shapes/dtypes, ordering, images, boundary/missingness,
classification, and presentation consistency. Version 1 limits archives and
expanded Parquet data to 32 GiB, individual records to 32 MiB, metadata files to
8 MiB, the manifest to 64 KiB, the Parquet footer to 128 MiB, and transition count
to ten million. It rejects object arrays and unknown archive members.

## Later Hugging Face publication

The extracted `data/train-00000-of-00001.parquet` is a standard typed Parquet
`train` split, ready to place in a dataset repository with the manifest, metadata,
and checkpoint directory. A future dataset card should declare:

```yaml
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*.parquet
```

Document the structured-value encoding above in that dataset card. Numeric image
bytes are explicit tensor features, not Hugging Face's automatically decoded
`Image` feature. Keep artifact licensing and provenance with a future published
copy. This feature does not upload datasets or provide a training integration.

See [capture measurements](player-trajectory-performance.md) for the activation
choice and reproducible workload.
