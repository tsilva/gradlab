# Monitor immutable checkpoints

Checkpoint monitoring is opt-in for native Breakout with `gradlab.ppo`, `sb3.ppo`,
and `sb3.a2c`. It evaluates every unique periodic, final, and interrupted
Checkpoint on same-host CPU workers. It does not attach recorders to the learner.
`trajectory_collection` is no longer accepted. Historical training recordings
retain their original meaning and are not current monitoring evidence.

Monitoring defaults to 400 stochastic episodes per Checkpoint, all recorded.
`train.checkpoint_monitoring.episodes` sets the evaluation count;
`record_episodes` independently selects how many to record (0 through `episodes`;
omitted or null means all). The first N manifest ordinals are recorded, independent
of outcomes and worker scheduling. Every episode still contributes to metrics.
Unrecorded episodes retain lightweight results without frame/action chunks. The frozen
manifest gives each episode separate environment and policy seeds, disjoint from
training and Acceptance. Episode identity and randomness are independent of
worker count, order, and retries. Monitoring is observational: training-only
Goals remain training-only, and these results do not stop learning or promote
policies. Goal-owned Modal Acceptance remains separate.

FirstWall's `ppo.yaml` keeps monitoring disabled and configures 100 evaluation
episodes, 50 recordings, and `contribution_bytes: 30000000000` (30 GB, decimal)
per Run. This cumulative monitoring allowance includes retained chunks, metadata
and representative media across checkpoints and retries; it excludes policy
checkpoint bytes and is not replenished by deleting local files. Changing either
count or the byte allowance invalidates prior calibration.

Enablement normally requires a supported calibration report bound to the resolved
configuration. For an explicitly uncalibrated run, pass
`--set train.checkpoint_monitoring.allow_uncalibrated=true` together with
`--set train.checkpoint_monitoring.enabled=true`. This hash-bound exception bypasses
calibration admission only; provider compatibility, resource limits, full task CPU
allocation and deadline checks remain enforced. It makes no throughput-support
claim and does not change Acceptance or promotion. No checked-in recipe is enabled by this change. The calibration operation reports incomplete, infeasible, or unproven evidence explicitly.
To execute a separately authorized campaign, use `gradlab monitor calibrate --campaign
campaign.json --output calibration.json`. This launches counterbalanced off/on pairs
through the ordinary dstack supervisor, resumes from a durable local campaign journal,
and waits for private R2 terminal receipts. It never grants support from process
separation or an untrained short rollout.

The campaign document declares `launch_args` (an argument list containing the
recipe, operator-selected compute target and explicit `--max-duration`), 3–20 distinct
training `seeds`, full monitoring `settings`, `warmup_updates`, and `representatives`.
Each early/intermediate/stronger/long-episode selector contains a campaign `seed` and
`checkpoint_step`. Set worker/shared/media budgets and `task_cpus` to the actual
allocation; there is no automatic hardware choice or resource downgrade. The operation
measures native inference, RGB capture/encoding, R2 transfer, full-video encoding,
W&B delivery, memory/spool peaks, retained bytes and matched learner throughput.
It checks host-load observations, equivalent resolved work and actual completion
time, and retains all campaign Run identities. Throughput uses total post-warm-up
transitions divided by their total measured time, including intermittent slowdowns.
Timestamped inference must overlap those learner intervals; monitoring performed
only during final drain cannot establish active-training overhead. Missing, interrupted, weak or short
representative evidence leaves calibration incomplete.

To assess an already-collected measurement document without launching work:

```bash
gradlab monitor calibrate --measurements campaign.json --output calibration.json
```

Measurements must cover early, intermediate, stronger, and long-episode policies,
full episode sets and delivery phases, and at least three matched independent
training-seed pairs with warm-up excluded. A report is not evidence of measurements
that were never run. The live throughput and browser-playback campaign requires
separate operator authorization; ordinary tests do not establish the 2% target.

The `train.checkpoint_monitoring` contract freezes evaluation and recording counts, checkpoint
cadence through `checkpoint_freq`, active workers, full task CPU allocation,
per-worker and shared memory/spool limits, cumulative retained bytes, dedicated W&B media memory/spool, chunk size,
scratch headroom, watchdog, and a finite whole-Run deadline. Unavailable resources
or incomplete evaluation cause an operational failure, never a scientific failure
or a reduced episode count. Post-training finalization expands to the declared
CPU allocation within the same budgets. Complete episodes are reused on the
single execution retry; diagnostic tails use content-addressed chunk names; video and W&B delivery do not rerun completed episodes.

Each recorded trajectory stores full lossless 210×160 RGB initial and true terminal frames,
requested/effective/executed/native actions, conditional overrides, provider and
shaped rewards, facts and episode boundaries. Bounded ZIP chunks preserve a
continuous step index and a shared boundary frame; long episodes cross chunks
without becoming complete prefixes. Verified R2 objects are canonical. Local
chunk files are reclaimed only after chunk and manifest verification. Durable
reservations count all retained contributions across Attempts and are not refunded
by local deletion.

Complete-set results include success rate and a two-sided 95% Wilson interval,
mean/median normalized brick progress (denominator 216), native score, shaped
return, episode length and count. Metrics with training counterparts use the same
suffix under `eval/`; median, Wilson bounds and video retain `eval/monitor/`.
All monitoring metrics use `eval/step`, the immutable
Checkpoint step, while W&B delivery uses monotone `ops/sequence`. The recorded episode
nearest the full evaluation-set median normalized progress, with manifest-order tie breaking, supplies
one original-frame video. The supervisor alone publishes its canonical R2 video
to W&B through the durable outbox. With zero recordings, selection/video are absent
and the supervisor publishes scalar metrics only. A complete terminal receipt requires all saved
Checkpoints, complete results, delivery and worker quiescence.

# Publish selected Runs

```bash
gradlab dataset publish-runs \
  --run gradlab-0123456789abcdef0123456789abcdef \
  --repo owner/checkpoint-dataset \
  --stage-min 0.25 --score-min 20
```

Add `--completed-snapshot` to publish a fixed snapshot of completed, verified
evaluations before a Run finishes. Unfinished evaluations and later completions
are excluded; training continues unchanged. Repeat the command to append a later
snapshot to a compatible repository.

Repeat `--run` to select multiple finalized Runs. Optional lower and upper bounds
cover `stage`, `return`, `score`, and `bricks`. Stage is immutable Checkpoint step
divided by planned training steps. Unrecorded episodes are excluded from dataset
publication. Selection reads episode metadata before frame
transfer. The command admits an explicit durable local publication job; use
`gradlab jobs` to inspect it. Training never starts HF publication.

HF receives the existing Breakout trajectories table format: deduplicated lossless
WebP images embedded in `frames` Parquet, typed `transitions`, `episodes`, and
`sessions` tables, tagged `record_json`, and `breakout-bricks-v1` annotations.
RGB remains full and unmasked. Transitions and episodes use one `all` container;
the episode `split` column is null. There is no train/heldout assignment. The
provider-submitted action index remains `native_action_json`; the original
internal emulator encoding is retained in `record_json.monitoring_record`.
Unavailable separately recorded task rewards/boundaries, temperature and native
frame timing remain null. Session records retain complete original episode
provenance, including Run, training seed, Checkpoint and evaluation identity.

Conversion resumes at completed episode boundaries, with a SQLite frame-deduplication
index, an 8 GiB temporary spool cap and 1 GiB free-space reserve. Images are checked
for exact decoded RGB equality after WebP encoding. The dataset keeps original
R2 recordings and previous HF files/revisions. Later contributions merge earlier
complete episodes into a new immutable snapshot; the current dataset view switches
atomically only after all snapshot files are staged. IDs join tables within a
pinned snapshot and must not be interpreted as row offsets or compared across
snapshot revisions.

The publisher preuploads binary content with two upload threads and publishes
one expected-parent commit, bounded to 100 files. Larger snapshots fail before
publication rather than expose incomplete data. A lost commit acknowledgement
is reconciled using the immutable receipt. Previously queued PNG-shard jobs
retain their original export format; new publications default to trajectory tables.

Dataset publication jobs share a persistent budget of 10 commit attempts per
rolling hour and 60 SDK operations per five minutes, with one writer per repository.
SDK operations can contain multiple underlying HTTP requests; this conservative
budget leaves headroom but is not a guarantee against remote throttling. A 429
pauses the shared publisher until the longest applicable reset plus a margin;
repository-commit throttling waits at least an hour. No immediate commit retry
is issued after throttling. Ambiguous responses are reconciled before another
attempt. Jobs defer durably rather than exhaust retries while quota is unavailable.

Use whole-Run or training-seed-group holdouts to avoid leakage across correlated
Checkpoints; GradLab assigns no downstream split. The isolated trajectory
collector remains a separate workflow.

## Explicit derived split views

An operator may explicitly request train/validation/test assignments after publication.
These are immutable derived views of a pinned dataset revision, never a new default
for monitoring collection. Keep complete episodes and reused environment/policy seeds
in one partition across checkpoints. Balance episode brick-progress distributions
first, then check returns, lengths and checkpoint coverage; publish the actual
statistics, grouping, deterministic assignment and thresholds with the split manifest.
Shared frame assets remain unchanged. Only training-referenced frame IDs may be used
to fit models or preprocessing; identical RGB states can recur across episodes.

Prepared split tables and their hashes use the durable `dataset-split-publication`
handler, the shared publication budget and one expected-parent commit of at most
100 files. A changed parent or changed prepared bytes blocks publication. The Hub
README selects the derived tables; previous unsplit files and revisions remain valid.
