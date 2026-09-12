# Experimental RGB dataset collector

Issue [#47](https://github.com/tsilva/gradlab/issues/47). This directory contains the
collector, its Pygame debugger, offline reader, validator, tests, and usage notes.
It adds no package CLI command and changes no shared GradLab runtime or Player code.

## Delivery status

Storage, collection controls, offline inspection, and native Breakout fidelity
tests are implemented. The collector reuses GradLab's existing shared checkpoint
loader unchanged, including for PPO/A2C, with staged bundle verification and the
shared Policy runtime. This is the agreed loading scope for this experiment;
it adds no new model format or shared loader changes.

PPO checkpoint loading and collection are integration-tested with exploration
both disabled and enabled. Action-program and cell-graph artifacts retain their
existing action-selection modes and do not support temperature exploration.
The 10 GiB pilot has not been authorized or run.

## Commands

Run from the repository root after `uv sync --frozen`. Supply a local immutable
bundle directory containing `model.zip`, `model.json`, and `recipe.json`, or a
Checkpoint file with its bound `.model.json` and `.recipe.json` sidecars.

```bash
uv run --frozen python experiments/dataset-collector/collector.py collect \
  ~/.config/gradlab/runs/dataset-pilot /path/to/checkpoint-bundle \
  --max-gib 10 --max-steps 10000 --max-seconds 3600
```

The same command appends to an existing compatible dataset. Each invocation uses
one Checkpoint. Later sessions may use a different compatible Checkpoint, with
separate provenance. Seed-allocation settings and the effective environment,
action, provider-version, RGB, and cadence contracts must remain compatible.
The script refuses to initialize a nonempty directory without its manifest.

Add `--debug` to start a local Pygame window paused. Space toggles play/pause and
Right advances exactly one contracted environment step. Esc or closing the window
stops collection. Paused redraws never call the Policy or environment. Debug mode
flushes each step and compares a copied live successor RGB against that exact
transition's frame decoded from the committed store. The UI reports mismatches or
read failures; it does not display a second copy of the live buffer as evidence.

```bash
uv run --frozen python experiments/dataset-collector/collector.py validate /path/to/dataset
uv run --frozen python experiments/dataset-collector/collector.py progress /path/to/dataset
uv run --frozen python experiments/dataset-collector/collector.py inspect /path/to/dataset --episode 1
uv run --frozen python experiments/dataset-collector/collector.py preview /path/to/dataset \
  /path/to/preview.mp4 --episode 1 --max-steps 600 --fps 30
```

Offline inspection starts at the episode's initial image. Left/Right navigate
stored transitions, Space plays/pauses, and Page Up/Page Down select episodes.
These commands need neither a Checkpoint nor a running environment. The preview
is a display artifact at the requested playback FPS; it is not a lossless copy of
the dataset and does not synthesize omitted native frames. Existing preview files
are never overwritten. Offline validation verifies stored data, not its equality
to a no-longer-available live source image.

## Contracts and exploration

Collection preserves the full uint8 HWC provider-rendered RGB, including HUD
pixels, at the provider's dimensions. It stores the initial frame and one successor
per contracted step. The Checkpoint determines frame skip, preprocessing, actions,
and structured Policy inputs. Processed observations and frame stacks are not
stored. The current native Breakout provider does not expose actual native frames
elapsed for each transition, so that field is explicitly absent rather than guessed
from configured frame skip.

The original episode contract and action-selection mode are preserved by default.
`--full-game --episode-steps N` explicitly replaces task termination rules with a
finite duration cap in contracted environment steps. Native termination remains
active; rewards, Policy inputs, action overrides, and cadence remain unchanged.
The original and effective contracts are recorded. Task termination, task timeout,
native termination, native truncation, and an external collection cutoff remain
separate facts. The collector requires terminal RGB from the pre-autoreset
single-lane diagnostics. Missing evidence fails recording and preserves an
incomplete prefix.

The proposed pilot settings are:

```text
--full-game --episode-steps <explicit cap>
--explore --temperatures 0.75 1.0 1.25 --probabilities 0.20 0.60 0.20
--temperature-block 256 --schedule-seed <explicit seed>
```

A block counts Policy decisions and resets its position at each episode start.
The schedule uses a separate NumPy Generator derived from its seed and the durable
episode ID. It never consumes environment or Policy RNG. Disabled exploration
uses temperature one and preserves stochastic selection. Invalid probabilities,
temperatures, block lengths, or ineffective exploration modes fail before stepping.
Boundary overrides or an enabled schedule classify the session as Counterfactual
Playback. Dataset collection never constitutes Training Success, Acceptance, or
Promotion, and emits no W&B metrics.

## Format and recovery

Format version 1 is experimental. Migration is not guaranteed. Existing datasets
are preserved; use a new directory when changing the format or effective contract.

- `manifest.json` records the immutable dataset contract and seed allocation.
- `index.sqlite` holds the global frame hash index, committed file bindings, episode
  allocation, session references, transition batch lookup, and exact counters.
- `rgb-*.bin` packs zlib-compressed full RGB images, with multiple images per batch.
- `steps-*.parquet` stores ordered transitions with compact frame IDs and numerical
  values serialized by GradLab's existing data-only tree codec.
- `episode-*.parquet` records immutable episode snapshots. SQLite selects the latest
  committed snapshot. `session-*.parquet` records Checkpoint/model/recipe identities,
  source/runtime provenance, collection settings, and classification once per session.
- `progress.json` is a replaceable progress snapshot. The index remains authoritative.

Frame identity is SHA-256 of a canonical image-format header plus raw RGB bytes.
Matching hashes require byte equality. Identical images with different histories,
actions, labels, or outcomes share RGB storage but keep every transition occurrence.
Frame identity says nothing about hidden simulator-state identity.

One process holds an exclusive advisory writer lock. Collection writes synchronous
bounded batches, so storage latency applies backpressure. The batch target defaults
to 128 transitions. Encoded records and compressed images trigger an 8 MiB flush
threshold; at most one additional bounded image and record can cross it. Each RGB
image is limited to 4 MiB and each metadata record to 8 MiB. SQLite's page cache is
bounded. There is no dataset-sized in-memory frame index or asynchronous write queue.

Files are created with unique names, synced, and their parent directory synced
before SQLite commits their visibility and counters. The disk budget includes all
dataset files, unfinished output, a conservative rollback-journal reservation,
and index/progress headroom. SQLite page growth is capped by the remaining budget.
A failed write stops collection. Files written before a failed commit are retained,
count toward disk usage, and remain invisible to readers. Committed data is never
rolled back to make room. Increasing a dataset's byte budget is an explicit command
option.

Episode IDs and seed assignments are committed before environment reset. Resume
marks an interrupted active episode incomplete and starts a new episode. It does
not restore the simulator or mid-episode Policy RNG. By default every fifth episode
is held out. Training episode seeds start at `gradlab.seeds.TRAIN_SEED_MIN`; held-out
seeds start at `gradlab.seeds.EVAL_SEED_START`. Allocation never repeats a seed after
an interrupted reservation. Training allocation stops at the end of the established
training range. Sharing physical image bytes does not permit training on held-out
labels or histories.

## Progress interpretation

Counts describe committed data. Each initial frame and successor contributes one
captured occurrence; source-frame references do not add another occurrence.
Cumulative reuse is `1 - unique_frames / captured_occurrences` and is unavailable
before the first capture. Pending transitions, captures, and buffer bytes are
reported separately during collection.

Collection reports recent image discovery per second and per occurrence, cumulative
unique RGB, complete/incomplete episode counts, throughput, elapsed time, actual
bytes, SQLite overhead, peak process RSS, and storage projections. Compression
savings compare compressed bytes with raw bytes of unique images. Duplicate reuse
is a separate fraction. Projections extrapolate observed growth and are not capacity
guarantees. Headless progress updates every five seconds and at shutdown; display
pacing changes throughput and time-limited sample counts.

HUD-only changes count as new images. Discovery attribution depends on collection
order. A novelty plateau does not prove exhaustive state or action coverage.

## Hugging Face export

`export-hf` creates a standalone, upload-ready snapshot without changing the
collection format. Stop the writer first; the export holds a shared dataset lock,
validates the source, and writes into a temporary sibling directory. Only a
completed export becomes the requested output directory. Existing destinations
are refused. Failed exports clean up their own temporary files.

```bash
uv run --frozen python experiments/dataset-collector/collector.py export-hf \
  /path/to/dataset /path/to/huggingface-export --max-gib 10
```

The export contains four Hugging Face configurations:

| Configuration | Contents | Splits |
| --- | --- | --- |
| `frames` | One lossless PNG per unique frame, image feature and SHA-256 | `assets` |
| `transitions` (default) | Ordered frame references, actions, rewards, temperatures, boundaries | `train`, `heldout` when present |
| `episodes` | Initial frames, lengths, seeds, session IDs, completion status | Original episode splits |
| `sessions` | Portable checkpoint provenance and collection settings | `metadata` |

PNG bytes are embedded in Parquet using Hugging Face's image feature metadata.
The Hub viewer can display the `frames` images; it does not automatically join
transition frame IDs to images. Consumers join `source_frame_id` and
`successor_frame_id` to `frames.frame_id`. IDs are not array offsets. The shared
image pool is not a training split: select episodes first, then their referenced
images, preserving held-out isolation. Incomplete episodes remain marked.

The numerical columns and JSON action columns are directly readable without
GradLab. `record_json` additionally preserves every original transition fact and
exact array dtypes/shapes using the documented tree structure and base64 array
bytes. The generated card describes decoding and loading. Manifest and session
metadata use the existing portable metadata filter. No Policy weights, SQLite,
private source paths, or custom Hugging Face loading script are required.

Parquet row groups are bounded by 128 records and an 8 MiB target (one bounded
record may exceed it). `--shard-rows` defaults to 4096. `--max-gib` limits the
completed export, separately from collection storage; a failed private staging
write can temporarily exceed it by a bounded row group and Parquet footer.
`export.json` records source identity, counts and file checksums. Its hashes
exclude `export.json` itself. This is a snapshot export, not incremental sync.

Load locally, or substitute a Hub dataset ID after upload:

```python
from datasets import load_dataset
steps = load_dataset("/path/to/huggingface-export", "transitions", split="train")
frames = load_dataset("/path/to/huggingface-export", "frames", split="assets")
episodes = load_dataset("/path/to/huggingface-export", "episodes", split="train")
```

The exporter uses existing dependencies. The `datasets` library is only needed by
consumers; it is not added to GradLab. Upload is separate and requires a chosen
repository and visibility. After setting the appropriate license and attribution
in the generated dataset card, an example for a **new** dataset repository is:

```bash
uv run --frozen hf upload YOUR_NAMESPACE/YOUR_DATASET /path/to/huggingface-export . \
  --type dataset --private
```

Do not overlay a shorter export on old shards: files absent locally are not
removed by this command. Use a fresh repository or explicitly manage obsolete
files when publishing a replacement snapshot. The collector never uploads data
or creates a Hub repository automatically.

## Validation and pilot

```bash
uv run --frozen pytest -q experiments/dataset-collector/test_collector.py
uv run --frozen ruff check experiments/dataset-collector
uv run --frozen ruff format --check experiments/dataset-collector
```

Tests use real temporary stores and scripted execution at the agreed boundary.
They cover mutable buffers, exact RGB reuse, distinct terminal/reset frames, missing
terminal evidence, stochastic baseline fidelity, temperature blocks, paused disk
readback, crash prefixes, fresh seed allocation, splits, writer locking, append
compatibility, disk limits, corrupt records, and bounded memory. A native Breakout
integration compares an in-memory stochastic Policy's ordinary execution with both
headless and debug recording over equal transition prefixes, including a task timeout. A separate integration saves and reloads a PPO Checkpoint
through the shared loader, collects native RGB with exploration off and on, and
validates terminal images, provenance, and the resulting dataset.

The 10 GiB pilot has **not run**. Before launch, obtain explicit pilot authorization
and record the selected Checkpoint, episode
cap, compute target, seed allocation, exact source, and resource limits under the
operator's run directory. Follow `COMPUTE.md` and the private operator inventory.
Use the explicit full-game override and exploratory schedule after fidelity checks
pass. Keep generated datasets, previews, and measurements outside tracked source.

The pilot report must include integrity results, a reconstructed trajectory preview,
full-RGB novelty, compression and reuse separately, index overhead, storage projections,
throughput, and memory observations. Those measurements validate the collector only.
