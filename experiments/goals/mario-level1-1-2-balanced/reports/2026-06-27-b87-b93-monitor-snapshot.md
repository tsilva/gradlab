# 2026-06-27 B87-B93 Monitor Snapshot

Goal metric: peak `train/info/level_complete/rate/min/last`, requiring both
seeds in the same two-seed batch to exceed `0.80` within 5M training steps.

## Live Status

Queue snapshot:

- Running: 5 train jobs on beast-3.
- Pending: 9 train jobs after adding B92/B93.
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

Full row scans over the five active runs, at about 1.15M-1.20M observed steps:

| Run | Peak min-rate | Peak step | Last min-rate | Notes |
| --- | ---: | ---: | ---: | --- |
| `b87_l11l12_lowpress_s198_20260627T110807Z` | `0.00` | `82,864` | `0.00` | no balanced signal yet |
| `b87_l11l12_lowpress_s199_20260627T110807Z` | `0.02` | `994,736` | `0.00` | tiny early signal only |
| `b88_l11l12_l12bias_s200_20260627T110819Z` | `0.00` | `57,792` | `0.00` | no balanced signal yet |
| `b88_l11l12_l12bias_s201_20260627T110819Z` | `0.01` | `1,131,728` | `0.01` | tiny early signal only |
| `b89_l11l12_complete25_s202_20260627T110832Z` | `0.01` | `968,832` | `0.00` | tiny early signal only |

PPO diagnostics were normal for early training: approximate KL roughly
`0.008-0.016`, explained variance roughly `0.86-0.93`, and no evidence of a
crash or queue/fleet issue. Decision: keep running; too early to stop or promote.

## Added Backfill

B92/B93 test a distinct hypothesis from B87-B91: B86 briefly found a strong
balanced policy and then collapsed, so a slower entropy decay with a higher
entropy floor might preserve enough exploration to make high-watermark crossings
more reproducible.

Added specs:

- `specs/b92-slowent-l11l12-two-seed.yaml`
  - Seeds: `198,199`
  - Deltas from B86: `ent_coef_final=0.001`,
    `ent_coef_schedule_timesteps=4000000`
  - W&B group: `b92-l11l12-slowent-two-seed`
  - Jobs: `43`, `44`
- `specs/b93-slowent-l12bias-l11l12-two-seed.yaml`
  - Seeds: `200,201`
  - Deltas from B86: B92 entropy schedule plus `state_probs=[0.4,0.6]`
  - W&B group: `b93-l11l12-slowent-l12bias-two-seed`
  - Jobs: `41`, `42`

Both specs validated with `rlab.job_queue.load_spec_document` and were enqueued
profileless for `rtx4090` with runtime image digest `c672be38cd0f`.
