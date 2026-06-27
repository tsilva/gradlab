# 2026-06-27 B89-B104 Progress Snapshot

Goal metric: peak `train/info/level_complete/rate/min/last`, requiring both
seeds in the same two-seed batch to exceed strict `0.80` within the 5M training
cap. This snapshot uses full-row W&B history scans for peak detection.

## Live Status

Queue after adding B103 and B104:

- Succeeded train jobs: `10`
- Running train jobs: `5`
- Pending train jobs: `26`
- Eval jobs: none

Active beast-3 workers:

- `job=36`: `b89_l11l12_complete25_s203_20260627T110832Z`
- `job=37`: `b90_l11l12_lowpress_l12bias_s204_20260627T112001Z`
- `job=38`: `b90_l11l12_lowpress_l12bias_s205_20260627T112001Z`
- `job=39`: `b91_l11l12_lowpress_complete25_s206_20260627T112013Z`
- `job=40`: `b91_l11l12_lowpress_complete25_s207_20260627T112013Z`

Fleet plan: keep the existing beast-3 RTX4090 container with five workers on
digest `c672be38cd0f`; no reconcile action required.

## Current W&B High-Watermarks

No two-seed batch has solved the goal.

| Group | Peaks | Decision |
| --- | ---: | --- |
| B86 | `0.16`, `0.33`, `0.80`, `0.16`, `0.11` | near-miss only; no strict `>0.80` and no two-seed success |
| B87 | `0.09`, `0.12` | failed; low update pressure alone did not fix Level1-2 |
| B88 | `0.07`, `0.09` | failed; Level1-2 sampling bias alone did not fix Level1-2 |
| B89 | `0.41`, `0.00` so far | seed 202 is best active signal but below target; seed 203 still running |
| B90 | `0.00`, `0.00` so far | too early; low pressure plus Level1-2 bias has no signal yet |
| B91 | `0.00`, `0.00` so far | too early; low pressure plus completion reward has no signal yet |

Current active run detail at scan time:

| Run | Step | Peak min-rate | Last min-rate | Last L1-1 | Last L1-2 | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `b89_l11l12_complete25_s203_20260627T110832Z` | `647,232` | `0.00` | `0.00` | `0.00` | `0.00` | paired B89 seed still early |
| `b90_l11l12_lowpress_l12bias_s204_20260627T112001Z` | `557,968` | `0.00` | `0.00` | `0.01` | `0.00` | too early |
| `b90_l11l12_lowpress_l12bias_s205_20260627T112001Z` | `581,632` | `0.00` | `0.00` | `0.00` | `0.00` | too early |
| `b91_l11l12_lowpress_complete25_s206_20260627T112013Z` | `557,776` | `0.00` | `0.00` | `0.00` | `0.00` | too early |
| `b91_l11l12_lowpress_complete25_s207_20260627T112013Z` | `564,336` | `0.00` | `0.00` | `0.02` | `0.00` | too early |

## Added Backfills

Completion reward is still the only ingredient that materially moved the
high-watermark, so the next backfills cover the missing stronger-completion
branch while staying within legal reward/hyperparameter/state-sampling levers.

### B103

- Spec: `specs/b103-complete50-l12soft-l11l12-two-seed.json`
- W&B group: `b103-l11l12-complete50-l12soft-two-seed`
- Seeds: `204,205`
- Delta from B98: add `state_probs=[0.45,0.55]`
- Jobs:
  - `65`: `b103_l11l12_complete50_l12soft_s204_20260627T122924Z`
  - `66`: `b103_l11l12_complete50_l12soft_s205_20260627T122924Z`

B103 tests whether doubling the true clean-completion bonus and adding soft
Level1-2 coverage raises the bottleneck level without starving Level1-1.

### B104

- Spec: `specs/b104-complete50-slowent-l12soft-l11l12-two-seed.json`
- W&B group: `b104-l11l12-complete50-slowent-l12soft-two-seed`
- Seeds: `206,207`
- Delta from B103: add slower entropy decay and higher entropy floor,
  `ent_coef_final=0.001`, `ent_coef_schedule_timesteps=4000000`
- Jobs:
  - `63`: `b104_l11l12_complete50_slowent_l12soft_s206_20260627T122923Z`
  - `64`: `b104_l11l12_complete50_slowent_l12soft_s207_20260627T122923Z`

B104 tests whether the stronger clean-completion reward needs retained
exploration to avoid the B89-style post-peak Level1-2 collapse.

Both specs validated with `rlab.job_queue.load_spec_document` and were enqueued
profileless for `rtx4090` with runtime image digest `c672be38cd0f`.
