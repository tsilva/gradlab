# 2026-06-27 B87-B102 Progress Snapshot

Goal metric: peak `train/info/level_complete/rate/min/last`, requiring both
seeds in the same two-seed batch to exceed `0.80` within the 5M training cap.
All decisions below use full-row W&B history scans, not final run summaries.

## Live Status

Queue snapshot after adding B101 and B102:

- Running: 5 train jobs on beast-3.
- Pending: 27 train jobs.
- Succeeded: 5 B86 incumbent runs.
- Fleet plan: keep existing beast-3 RTX4090 container with 5 workers on digest
  `c672be38cd0f`; no reconcile action required.

Running jobs:

- `b87_l11l12_lowpress_s198_20260627T110807Z`
- `b87_l11l12_lowpress_s199_20260627T110807Z`
- `b88_l11l12_l12bias_s200_20260627T110819Z`
- `b88_l11l12_l12bias_s201_20260627T110819Z`
- `b89_l11l12_complete25_s202_20260627T110832Z`

Next pending jobs start with B89 seed 203, then B90/B91. B92-B102 are queued
behind those jobs.

## Current W&B High-Watermarks

B90-B102 are still pending and have no W&B run history yet. The current active
tranche has advanced to about 4.4M steps without a new solved batch.

| Group | Run | State | Peak min-rate | Peak step | Last min-rate | Last L1-1 | Last L1-2 | Last step | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B86 | `b86_l11l12_b74current_s193_20260627T091726Z` | finished | `0.33` | `1,992,720` | `0.08` | `0.78` | `0.08` | `5,005,312` | incumbent support seed |
| B86 | `b86_l11l12_b74current_s194_20260627T091726Z` | finished | `0.16` | `2,911,312` | `0.01` | `0.75` | `0.01` | `5,005,312` | L1-2 collapse |
| B86 | `b86_l11l12_b74current_s195_20260627T091726Z` | finished | `0.80` | `4,070,256` | `0.13` | `0.87` | `0.13` | `5,005,312` | near-miss; not strict `>0.80` |
| B86 | `b86_l11l12_b74current_s196_20260627T091726Z` | finished | `0.11` | `2,894,560` | `0.00` | `0.50` | `0.00` | `5,005,312` | L1-2 collapse |
| B86 | `b86_l11l12_b74current_s197_20260627T091726Z` | finished | `0.16` | `3,372,064` | `0.06` | `0.76` | `0.06` | `5,005,312` | L1-2 collapse |
| B87 | `b87_l11l12_lowpress_s198_20260627T110807Z` | running | `0.09` | `2,121,920` | `0.01` | `0.92` | `0.01` | `4,451,488` | strong L1-1, L1-2 bottleneck |
| B87 | `b87_l11l12_lowpress_s199_20260627T110807Z` | running | `0.12` | `2,512,032` | `0.01` | `0.71` | `0.01` | `4,414,128` | strong L1-1, L1-2 bottleneck |
| B88 | `b88_l11l12_l12bias_s200_20260627T110819Z` | running | `0.07` | `3,968,832` | `0.02` | `0.85` | `0.02` | `4,424,672` | sampling bias alone not enough |
| B88 | `b88_l11l12_l12bias_s201_20260627T110819Z` | running | `0.09` | `3,084,848` | `0.02` | `0.31` | `0.02` | `4,412,736` | weak balanced lift |
| B89 | `b89_l11l12_complete25_s202_20260627T110832Z` | running | `0.41` | `2,512,208` | `0.02` | `0.81` | `0.02` | `4,416,432` | best active lift; L1-2 still collapses |

Decision: no batch is solved. The current active evidence still points to
Level1-2 collapse as the bottleneck. Lower update pressure alone and Level1-2
sampling bias alone are not sufficient. Completion reward remains the only arm
that materially raised the balanced high-watermark.

## Added Backfills

B101 and B102 stay on the B89 completion-reward branch and target post-peak
collapse without changing the task or evaluation protocol.

### B101

- Spec: `specs/b101-complete25-slowent-l12soft-l11l12-two-seed.json`
- Seeds: `198,199`
- W&B group: `b101-l11l12-complete25-slowent-l12soft-two-seed`
- Delta from B97: add `state_probs=[0.45,0.55]`
- Jobs:
  - `59`: `b101_l11l12_complete25_slowent_l12soft_s198_20260627T121303Z`
  - `60`: `b101_l11l12_complete25_slowent_l12soft_s199_20260627T121303Z`

B101 tests whether the B89 completion reward needs both slower entropy decay
and soft Level1-2 coverage, without changing PPO update pressure.

### B102

- Spec: `specs/b102-lowpress-complete25-slowent-l12soft-l11l12-two-seed.json`
- Seeds: `200,201`
- W&B group: `b102-l11l12-lowpress-complete25-slowent-l12soft-two-seed`
- Delta from B100: add slower entropy decay and higher entropy floor,
  `ent_coef_final=0.001`, `ent_coef_schedule_timesteps=4000000`
- Jobs:
  - `61`: `b102_l11l12_lowpress_complete25_slowent_l12soft_s200_20260627T121304Z`
  - `62`: `b102_l11l12_lowpress_complete25_slowent_l12soft_s201_20260627T121304Z`

B102 tests the compact full anti-collapse bundle: `completion_reward=25`,
`target_kl=0.16`, late LR decay to `0.0001`, slower entropy decay, higher
entropy floor, and `state_probs=[0.45,0.55]`.

Both specs validated with `rlab.job_queue.load_spec_document` and were enqueued
profileless for `rtx4090` with runtime image digest `c672be38cd0f`.
