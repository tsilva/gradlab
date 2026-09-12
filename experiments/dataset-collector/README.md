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
Right advances one step per active environment. Esc or closing the window
stops collection. Paused redraws never call the Policy or environment. Debug mode
flushes each step and compares a copied live successor RGB against that exact
transition's frame decoded from the committed store. The UI reports mismatches or
read failures; it does not display a second copy of the live buffer as evidence.

### Vector collection

Add `--n-envs 16` to batch PPO/A2C inference across sixteen independently seeded
environments. The default is one; supported counts are 1–64. One model serves
all lanes. Environments step sequentially in lane order within the same process;
batching accelerates policy inference.
Lanes with the same current exploration temperature share one forward pass.
Stateful action-program/cell-graph execution and state-dependent exploration
remain single-environment only.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen python \
  experiments/dataset-collector/collector.py collect \
  ~/.config/gradlab/runs/dataset-vector /path/to/checkpoint-bundle \
  --n-envs 16 --max-gib 1 --max-steps 100000 --max-seconds 120
```

Benchmark lane counts and CPU thread settings on the target machine. Larger
batches trade inference overhead against image encoding, environment stepping,
storage latency and memory; more environments need not improve total throughput.

`--max-steps` counts transitions across all lanes. The last batch shrinks to the
remaining count, so it never steps extra lanes past the cap. The time limit is
checked between batches; final writes can extend wall time slightly. Each lane
keeps its own episode, seed, temperature-block position, and terminal frame.
All lanes share frame deduplication and the bounded writer. Stopping or restarting
marks every unfinished episode incomplete. `lane_id` is in each transition's
exact `record_json`; environment count and batching/RNG rules are in session provenance.

Vector policy sampling uses a single session RNG stream seeded by the first
reserved episode seed. Resetting one environment never reseeds that stream or
resets another lane. Repeatability requires the same checkpoint, runtime, initial
allocation, lane count and step budget. Changing lane count changes stochastic
trajectories; it does not change the recorded action-selection rule. Single-lane
collection retains its original per-episode policy seeding.

In vector debug mode, each Right press advances a batch and shows the last lane
stepped, including its lane and episode IDs. All lanes are recorded and available
in the offline inspector. Pausing still consumes no policy RNG or environment steps.

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

Format version 2 is experimental. Version 1 datasets are preserved and rejected by
this version; use the earlier collector revision to inspect them. Migration is not guaranteed. Existing datasets
are preserved; use a new directory when changing the format or effective contract.

- `manifest.json` records the immutable dataset contract and seed allocation.
- `index.sqlite` holds the global frame hash index, committed file bindings, episode
  allocation, session references, transition batch lookup, and exact counters.
- `frames-*.parquet` stores lossless PNG images with the Hugging Face image feature,
  frame IDs and RGB hashes, with multiple images per batch.
- `steps-*.parquet` stores ordered transitions with typed numerical columns, action
  JSON, compact frame IDs, and exact structured facts in `record_json`.
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
to 128 transitions per environment; reaching it flushes all pending lanes together.
Encoded records and compressed images across all lanes trigger an 8 MiB flush
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
savings compare complete committed frame-Parquet file bytes (including their IDs
and metadata) with raw bytes of unique images. `non_rgb_bytes` covers everything
outside those frame files, including SQLite, transition metadata and orphan files. Duplicate reuse
is a separate fraction. Projections extrapolate observed growth and are not capacity
guarantees. Headless progress updates every five seconds and at shutdown; display
pacing changes throughput and time-limited sample counts.

HUD-only changes count as new images. Discovery attribution depends on collection
order. A novelty plateau does not prove exhaustive state or action coverage.

## Direct Hugging Face storage

Collection writes the final Hugging Face-compatible Parquet tables directly.
There is no separate image format, data conversion, or second dataset copy.
SQLite remains a local index for deduplication, random access and crash recovery.
Each batch closes and syncs its Parquet files before SQLite commits visibility.
Readers and the debugger use those same committed files.

After stopping the writer, prepare the existing directory for Hugging Face:

```bash
uv run --frozen python experiments/dataset-collector/collector.py prepare-hf /path/to/dataset
```

This validates the dataset and generates `README.md` and `upload.json` in place.
It does not rewrite or copy any Parquet bytes. The card selects exact committed
files, including only the latest snapshot of each episode. Abandoned files and
superseded snapshots remain locally preserved and are excluded from publication.

| Configuration | Contents | Splits |
| --- | --- | --- |
| `frames` | One PNG per unique RGB image, frame ID and hash | `assets` |
| `transitions` (default) | Ordered frame references, actions, rewards and boundaries | `train`, `heldout` when present |
| `episodes` | Initial frame, length, seed, session ID and completion status | Original episode splits |
| `sessions` | Portable checkpoint provenance and collection settings | `metadata` |

```python
from datasets import load_dataset
steps = load_dataset("/path/to/dataset", "transitions", split="train")
frames = load_dataset("/path/to/dataset", "frames", split="assets")
episodes = load_dataset("/path/to/dataset", "episodes", split="train")
```

The Hub displays images in `frames`; transition IDs do not automatically render
as images. Join source/successor frame IDs to `frames.frame_id`, not row offsets.
Select episode splits before their referenced images. The shared `assets` pool is
not a training split. Incomplete episodes remain marked. `record_json` retains
all original facts and exact NumPy dtypes/shapes using the documented tree format
with base64 array bytes; ordinary typed columns need no GradLab decoder.

`upload.json` is the upload-readiness receipt: it lists file paths, sizes and
SHA-256 checksums. Re-preparation invalidates the previous receipt before changing
the card and publishes a new receipt last. If preparation is interrupted, rerun
it before uploading. Upload only its listed
files plus `upload.json`, excluding SQLite and stale/uncommitted files. Preparation
records one snapshot: subsequent collection does not change its immutable data
files, but rerun preparation to include new commits. Preparation regenerates the
card and file list; it does not choose a license, repository or visibility, or
upload anything. Select the applicable license and attribution before publishing,
and refresh the card's checksum if you edit it after preparation. Do not overlay
a replacement snapshot on obsolete remote shards without managing their removal.

The exporter command `export-hf` has been removed. No new dependencies are needed
for collection or preparation; consumers can use the standard `datasets` library.
Preparation metadata is capped at 8 MiB and does not rewrite the collection's byte
budget. Leave room for the card and upload list outside that collection budget.

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
validates terminal images, provenance, and the resulting dataset. Vector tests
cover partial final batches, interleaved frame discovery, temperature-group action
alignment, recovery across all lanes, a 64-lane storage budget, native headless/debug
equivalence across lane resets, early rejection of unsupported execution, and
Hugging Face split/frame/record integrity.

The 10 GiB pilot has **not run**. Before launch, obtain explicit pilot authorization
and record the selected Checkpoint, episode
cap, compute target, seed allocation, exact source, and resource limits under the
operator's run directory. Follow `COMPUTE.md` and the private operator inventory.
Use the explicit full-game override and exploratory schedule after fidelity checks
pass. Keep generated datasets, previews, and measurements outside tracked source.

The pilot report must include integrity results, a reconstructed trajectory preview,
full-RGB novelty, compression and reuse separately, index overhead, storage projections,
throughput, and memory observations. Those measurements validate the collector only.
