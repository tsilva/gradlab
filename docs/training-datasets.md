# Monitor immutable checkpoints

Checkpoint monitoring is opt-in for native Breakout with `gradlab.ppo`, `sb3.ppo`,
and `sb3.a2c`. It evaluates every unique periodic, final, and interrupted
Checkpoint on same-host CPU workers. It does not attach recorders to the learner.
`trajectory_collection` is no longer accepted. Historical training recordings
retain their original meaning and are not current monitoring evidence.

Monitoring defaults to 400 stochastic episodes per Checkpoint. The frozen
manifest gives each episode separate environment and policy seeds, disjoint from
training and Acceptance. Episode identity and randomness are independent of
worker count, order, and retries. Monitoring is observational: training-only
Goals remain training-only, and these results do not stop learning or promote
policies. Goal-owned Modal Acceptance remains separate.

Enablement requires a supported calibration report bound to the resolved
configuration. No checked-in recipe is enabled by this change. The calibration
assessment command reports incomplete, infeasible, or unproven evidence explicitly:

```bash
gradlab monitor calibrate --measurements campaign.json --output calibration.json
```

Measurements must cover early, intermediate, stronger, and long-episode policies,
full episode sets and delivery phases, and at least three matched independent
training-seed pairs with warm-up excluded. A report is not evidence of measurements
that were never run. The live throughput and browser-playback campaign requires
separate operator authorization; ordinary tests do not establish the 2% target.

The `train.checkpoint_monitoring` contract freezes episode count, checkpoint
cadence through `checkpoint_freq`, active workers, full task CPU allocation,
per-worker and shared memory/spool limits, cumulative retained bytes, chunk size,
scratch headroom, watchdog, and a finite whole-Run deadline. Unavailable resources
or incomplete evaluation cause an operational failure, never a scientific failure
or a reduced episode count. Post-training finalization expands to the declared
CPU allocation within the same budgets. Complete episodes are reused on the
single execution retry; video and W&B delivery do not rerun completed episodes.

Each trajectory stores full lossless 210×160 RGB initial and true terminal frames,
requested/effective/executed/native actions, conditional overrides, provider and
shaped rewards, facts and episode boundaries. Bounded ZIP chunks preserve a
continuous step index and a shared boundary frame; long episodes cross chunks
without becoming complete prefixes. Verified R2 objects are canonical. Local
chunk files are reclaimed only after chunk and manifest verification. Durable
reservations count all retained contributions across Attempts and are not refunded
by local deletion.

Complete-set results include success rate and a two-sided 95% Wilson interval,
mean/median normalized brick progress (denominator 216), native score, shaped
return, episode length and count. `eval/monitor/*` uses `eval/step`, the immutable
Checkpoint step, while W&B delivery uses monotone `ops/sequence`. The full episode
nearest median normalized progress, with manifest-order tie breaking, supplies
one original-frame video. The supervisor alone publishes its canonical R2 video
to W&B through the durable outbox. A complete terminal receipt requires all saved
Checkpoints, complete results, delivery and worker quiescence.

# Publish selected Runs

```bash
gradlab dataset publish-runs \
  --run gradlab-0123456789abcdef0123456789abcdef \
  --repo owner/checkpoint-dataset \
  --stage-min 0.25 --score-min 20
```

Repeat `--run` to select multiple finalized Runs. Optional lower and upper bounds
cover `stage`, `return`, `score`, and `bricks`. Stage is immutable Checkpoint step
divided by planned training steps. Selection reads episode metadata before frame
transfer. The command admits an explicit durable local publication job; use
`gradlab jobs` to inspect it. Training never starts HF publication.

HF receives bounded immutable PNG ZIP assets and Parquet transition tables. Each
complete episode index commits only after all its chunks. Run, training seed,
Checkpoint, evaluation and episode identities remain queryable. Repeated and
overlapping selections reuse identities, and expected-parent commits reconcile
concurrent appenders. An incompatible or historic dataset is rejected. R2 sources
remain canonical and are retained after publication.

Use whole-Run or training-seed-group holdouts to avoid leakage across correlated
Checkpoints; GradLab assigns no downstream split. The isolated trajectory
collector remains a separate workflow.
