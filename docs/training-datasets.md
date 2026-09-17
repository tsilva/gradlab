# Collect training trajectories

Training collection is opt-in for SB3 PPO/A2C with the exact native Breakout
0.5.13 action contract. Correctness is covered by provider and lifecycle tests.
The matched training-throughput acceptance benchmark is still required before
claiming the at-most-2% overhead target. No live training is started by tests.

Add this runtime setting to a recipe or its launch overrides:

```yaml
runtime:
  trajectory_collection:
    enabled: true
```

Resolved defaults are 10 GiB of total reserved contribution per Run, 256 MiB of
capture and delivery working memory, 512 MiB of local spool, 32 MiB per chunk, one active
recording, 5% reset sampling probability, 20 planned-budget stages, and a
120-second dataset drain. `scratch_headroom_bytes` defaults to 1 GiB; the encoder
also retains at least 6% filesystem headroom, ahead of the global scratch guard.
All values are configurable through `trajectory_collection` and are validated
before execution. An HF target or credential is not a training input.

Capture runs at the contracted action cadence. GradLab retains the Policy
request, effective action, submitted provider action, native encoding, and
`auto_serve` override reason. The execution guarantee is specific to this
verified provider and action table. Requested and executed action integers can
use different tables; inspect the stored contract instead of comparing integers
without their meaning.

The encoder samples at reset with a separate random stream. Admission also
depends on budget, memory and disk availability. These recordings are not an
unbiased sample of all training experience. A chunk reserves its entire maximum
allowance at admission; smaller compressed chunks do not replenish that
allowance. Local deletion also does not replenish it. Verified remote
reservations carry across Attempts. Each Attempt also reserves 128 KiB for its
producer and finalization metadata; exhausted Runs skip further capture without
creating new dataset artifacts. Admission allowance grows against the
original planned training duration, even if training stops early.

Each episode produces a bounded ZIP of full unmasked PNG frames and a JSONL
transition table, plus a separate episode manifest. `frames/0.png` is the initial
frame; transition `i` leads to `frames/i+1.png`. The true terminal frame is captured
before reset. A recording that reaches chunk capacity, 8,192 transitions, buffer
pressure, or training shutdown becomes an explicit contiguous prefix. Its
`prefix_return` is not a full-episode result. Complete episodes retain individual
return, outcome, native score and brick facts, including normalized progress and
its provider metadata. Per-transition steps and Policy updates avoid attributing
an evolving training Policy to one immutable Checkpoint.

The supervisor uploads chunks under `datasets/runs/<run>/attempts/<attempt>/` in
the configured model artifact R2 bucket. It verifies bytes and manifests before
reclaiming local ZIPs. Manifests, reservations and canonical R2 sources remain.
Successful terminal drain requires a verified inventory in `dataset_delivery`.
A timeout preserves spool evidence and reports failure. Forced host loss can
still destroy unuploaded data; this feature does not reconstruct lost experience.

# Publish selected Runs

```bash
gradlab dataset publish-runs \
  --run gradlab-0123456789abcdef0123456789abcdef \
  --repo owner/training-dataset \
  --stage-min 0.25 --score-min 20
```

Repeat `--run` to select multiple finalized Runs. Optional bounds are `--stage-min`,
`--stage-max`, `--return-min`, `--return-max`, `--score-min`, `--score-max`,
`--bricks-min`, and `--bricks-max`. Stage filters use episode start progress
against the planned budget. Return filters use the captured shaped return;
score and brick filters use the last captured authoritative provider facts.
Missing requested facts exclude an episode. `--include-prefixes` explicitly
allows incomplete recordings; the default includes only complete episodes.

The command admits a durable local job and reports its identity. Use
`gradlab jobs` to inspect or retry work. Publication is local and uses local HF
credentials. Ending training never initiates this command. Admission pins R2
inventory hashes; retries cannot silently switch to a later Attempt.

HF receives immutable episode indexes, PNG ZIP assets, and Parquet transition
rows with archive/member references. `transition_json` preserves the complete
transition facts. Each episode's files appear in one expected-parent commit.
Repeated or overlapping selections reuse the same episode identity. Conflicting
content or execution contracts are rejected. Contribution metadata records the
Run inventories, predicates and selected episodes at an immutable HF revision.
Only bounded chunks are downloaded and converted; the entire remote dataset is
not downloaded to append.

R2 sources are retained after HF publication. Whole-Run or seed-group holdouts
are a useful starting point for downstream training; no train/validation split
is assigned. Dataset holdout is unrelated to Policy Acceptance evaluation.
Existing isolated-collector datasets keep their existing workflow and format.
