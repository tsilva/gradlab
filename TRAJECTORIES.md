# Trajectory schema v1

This contract governs the four-table checkpoint trajectory datasets published by
GradLab. It is independent of metrics schema versions, package versions, HF
commit revisions, PNG monitoring recordings and the separate Gymrec-compatible
`dataset_format_version = 3` recorder.

`gradlab.trajectory_format.TRAJECTORY_SCHEMA_VERSION` is the current integer
version. `trajectory_schema_contract()` returns the complete machine-readable
definition: ordered Arrow fields, types, nullability, and semantic rules.
`trajectory_schema_identity()` returns the version and SHA-256 of its canonical
JSON. A frozen fingerprint test catches contract changes without an intentional
version update.

## Declarations and validation

Publication receipts, view pointers, queued WebP publication requests, and split
manifests declare `trajectory_schema_version` and `trajectory_schema_sha256`.
Every Parquet file carries these keys in Arrow metadata with the `gradlab.` prefix,
plus `gradlab.trajectory_table`. Frame files also retain HF Image feature metadata.
Each published snapshot includes `schema.json` and a `publication.json` inventory
of table paths, kinds, row counts, and file SHA-256 hashes.

Ordinary consumers accept only the exact current version and fingerprint.
Absent declarations, booleans, numeric strings, older/future versions, wrong table
identities, and structural mismatches are errors. Do not infer compatibility from
a filename, matching columns alone, the producer's version, or an HF repository name.

Use `open_trajectory_parquet(path, "transitions")` (or `episodes`, `frames`,
`sessions`) to validate a shard's footer before accessing rows. It rejects added,
removed, reordered, retyped or differently nullable fields even if the version
metadata claims compatibility. Generic third-party Parquet/HF readers do not
automatically enforce this GradLab contract.

```bash
gradlab dataset verify-trajectories /path/to/snapshot
```

This validates declarations, the contract document, physical schemas, the full
shard inventory, hashes and row counts. It is a format/integrity check, not a new
scientific evaluation or replay of the environment.

## Meaning of v1

| Table | Identity and relationships | Meaning |
| --- | --- | --- |
| `transitions` | `episode_id`, zero-based `step`, `session_id`; source/successor frame IDs join `frames` | One ordered action-conditioned transition at the recorded action cadence |
| `episodes` | `episode_id`, `session_id`, `initial_frame_id` | Complete episode boundaries, length, seed and optional split assignment |
| `frames` | `frame_id`, RGB `sha256` | Deduplicated lossless WebP, full unmasked 210×160 RGB canvas |
| `sessions` | `session_id` | Tagged `record_json` containing original monitoring episode provenance |

IDs are snapshot-local identifiers, not row offsets or cross-snapshot identities.
Successor frames precede reset; initial and terminal frames remain explicit.
`selected_action_json` is policy-requested, `effective_action_json` follows any
override, and `native_action_json` is the submitted provider action index. The
internal emulator encoding remains in the original monitoring record.

`policy_reward` is shaped reward and `native_reward` is provider reward.
`task_reward` and other unavailable separately recorded facts remain null.
`terminated`/`truncated` describe scientific episode boundaries; provider and task
flags retain their distinct meanings. Missing values are never replaced by zero.

Brick annotations use `breakout-bricks-v1`. Transition grids describe successor
frames, source flags describe source frames, and initial episode annotations
describe the initial frame. RGB hashes cover the canonical v2 image header plus
uncompressed pixels. `record_json` retains the tagged-tree encoding with base64
numeric array leaves.

Unassigned episode splits are null. Explicit derived splits preserve complete
episodes and group reused evaluation seeds across checkpoints. Shared frame
assets do not imply permission to fit preprocessing on held-out frames.

## Version changes and migrations

Change the version whenever a field, type, nullability, relationship or field
meaning changes. An additive column also requires a new version under the current
strict compatibility policy. New data, shard boundaries or seed assignments alone
do not change the schema. The HF commit identifies the snapshot; the schema version
identifies how to interpret it.

New migrations must register an explicit source-to-target edge with a stable
migration identity and tests for its validation/transformation. Do not make ordinary
readers guess historical semantics or simply rewrite a version label. Conversion
must report unavailable information instead of inventing it.

The initial registered migration is
`gradlab.trajectories.webp.v1-unversioned` → schema 1. It accepts a local legacy
WebP snapshot with its original `publication.json`, unmasked-image declaration,
and all four table directories. It does not accept the older HUD-masked collector,
PNG exports, mixed-version files or an entire HF repository history tree.

```bash
gradlab dataset migrate-trajectories /path/to/legacy-snapshot /path/to/new-snapshot \
  --source-format gradlab.trajectories.webp.v1-unversioned \
  --source-revision efd5726b5786252b89242cb9fd104518628c1718
```

For a downloaded HF repository, point to the specific
`trajectories/<snapshot-id>/` directory containing the receipt and four tables.
The source revision is the full immutable HF commit from which those files were
obtained. The command records that operator-supplied provenance; it does not
contact HF to authenticate the local checkout. It does not migrate a derived split
view selected elsewhere by the repository README. Prepare such views against a
versioned source using the split rules below.

Migration creates a new directory, never overwrites an existing destination,
and leaves source files intact. It streams the rewrite, compares every decoded
value (including image bytes) with its source, validates the new snapshot, and
records source file hashes, source revision, migration identity, target contract,
and verification results in the new receipt. Failed staging output is removed.
No HF publication happens during migration.

## Publication and derived splits

New exports stamp the version and preserve it through compaction and resume.
Appending to an existing WebP view requires its current schema declarations.
Unversioned queued WebP jobs and incompatible views fail explicitly; recreate the
request against a compatible destination after migration. PNG jobs retain their
separate format.

A prepared split manifest must carry the current identity, `source_revision`
equal to the queued immutable parent, and a `tables` object keyed by each uploaded
repository-relative Parquet path. Each entry contains `table`, `rows` and `sha256`.
Both transitions and episodes must be present with stamped schemas. Validate input
shards with `open_trajectory_parquet` before splitting, and preserve schema metadata
in output. The durable split publisher validates the prepared files and source
view before upload. Schema checks do not replace the grouping and performance
balance checks recorded by the split producer.
