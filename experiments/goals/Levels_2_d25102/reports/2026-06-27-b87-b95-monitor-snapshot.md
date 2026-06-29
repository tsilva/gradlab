# 2026-06-27 B87-B95 Monitor Snapshot

Goal metric: peak `train/info/level_complete/rate/min/last`, requiring both
seeds in the same two-seed batch to exceed `0.80` within 5M training steps.

## Live Status

Queue snapshot after adding B94/B95:

- Running: 5 train jobs on beast-3.
- Pending: 13 train jobs.
- Succeeded: 5 B86 incumbent runs.
- Fleet plan: keep existing beast-3 RTX4090 container with 5 workers on digest
  `c672be38cd0f`; no reconcile action required.

Running jobs:

- `b87_l11l12_lowpress_s198_20260627T110807Z`
- `b87_l11l12_lowpress_s199_20260627T110807Z`
- `b88_l11l12_l12bias_s200_20260627T110819Z`
- `b88_l11l12_l12bias_s201_20260627T110819Z`
- `b89_l11l12_complete25_s202_20260627T110832Z`

## Current W&B High-Watermarks

Full row scans over the five active runs, at about 1.55M-1.59M observed steps:

| Run | Peak min-rate | Peak step | Last min-rate | Last L1-1 | Last L1-2 | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `b87_l11l12_lowpress_s198_20260627T110807Z` | `0.04` | `1,571,168` | `0.04` | `0.06` | `0.04` | best current early signal is low-pressure |
| `b87_l11l12_lowpress_s199_20260627T110807Z` | `0.05` | `1,490,560` | `0.05` | `0.05` | `0.07` | best current early signal is low-pressure |
| `b88_l11l12_l12bias_s200_20260627T110819Z` | `0.03` | `1,362,112` | `0.01` | `0.01` | `0.05` | 40/60 bias may be making L1-1 the bottleneck early |
| `b88_l11l12_l12bias_s201_20260627T110819Z` | `0.01` | `1,131,728` | `0.00` | `0.00` | `0.08` | 40/60 bias may be making L1-1 the bottleneck early |
| `b89_l11l12_complete25_s202_20260627T110832Z` | `0.01` | `968,832` | `0.00` | `0.00` | `0.07` | completion reward has not shown early balanced lift yet |

Decision: no batch is solved. All active runs remain far below the strict
`>0.80` high-watermark, but this is still early relative to the B86 near-miss,
which peaked near 4.07M steps. PPO diagnostics did not show a crash or obvious
optimizer failure; keep the active jobs running.

## Added Backfill

B94/B95 extend the current search around the best early signal:

- B87 low-pressure is the strongest active early signal so far.
- B86 showed late high-watermark behavior followed by collapse.
- B88's 40/60 Level1-2 bias improves L1-2 early but can make L1-1 the min-rate
  bottleneck, so use a softer bias for one follow-up.

Added specs:

- `specs/b94-lowpress-slowent-l11l12-two-seed.yaml`
  - Seeds: `202,203`
  - Deltas from B86: `target_kl=0.16`,
    `learning_rate_final=0.0001`,
    `learning_rate_schedule_timesteps=4000000`,
    `ent_coef_final=0.001`,
    `ent_coef_schedule_timesteps=4000000`
  - W&B group: `b94-l11l12-lowpress-slowent-two-seed`
  - Jobs: `47`, `48`
- `specs/b95-lowpress-slowent-l12soft-l11l12-two-seed.yaml`
  - Seeds: `204,205`
  - Deltas from B94: `state_probs=[0.45,0.55]`
  - W&B group: `b95-l11l12-lowpress-slowent-l12soft-two-seed`
  - Jobs: `45`, `46`

Both specs validated with `rlab.job_queue.load_spec_document` and were enqueued
profileless for `rtx4090` with runtime image digest `c672be38cd0f`.
