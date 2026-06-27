# 2026-06-27 B87-B97 Progress Snapshot

Goal metric: peak `train/info/level_complete/rate/min/last`, requiring both
seeds in the same two-seed batch to exceed `0.80` within 5M training steps.

## Live Status

Queue snapshot after adding B97:

- Running: 5 train jobs on beast-3.
- Pending: 17 train jobs.
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

Full row scans over the five active runs, at about 2.57M-2.61M observed steps:

| Run | Peak min-rate | Peak step | Last min-rate | Last L1-1 | Last L1-2 | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `b87_l11l12_lowpress_s198_20260627T110807Z` | `0.09` | `2,121,920` | `0.05` | `0.40` | `0.05` | L1-2 bottleneck |
| `b87_l11l12_lowpress_s199_20260627T110807Z` | `0.12` | `2,512,032` | `0.08` | `0.36` | `0.08` | L1-2 bottleneck |
| `b88_l11l12_l12bias_s200_20260627T110819Z` | `0.06` | `1,925,408` | `0.02` | `0.09` | `0.02` | not promising |
| `b88_l11l12_l12bias_s201_20260627T110819Z` | `0.04` | `2,590,224` | `0.04` | `0.07` | `0.04` | not promising |
| `b89_l11l12_complete25_s202_20260627T110832Z` | `0.41` | `2,512,208` | `0.05` | `0.43` | `0.05` | best active peak; collapse is now L1-2-limited |

Decision: no batch is solved. B89 seed 202 produced the best active
high-watermark so far, but the paired seed is still pending and the value is
well below the strict `>0.80` target.

## Added Backfill

B97 tests the most direct stabilization hypothesis for B89:

- Keep `completion_reward=25`, because it produced the best observed active
  high-watermark.
- Change only entropy scheduling to slow decay and preserve a higher entropy
  floor, targeting the post-peak Level1-2 collapse.

Added spec:

- `specs/b97-complete25-slowent-l11l12-two-seed.json`
  - Seeds: `198,199`
  - Deltas from B89: `ent_coef_final=0.001`,
    `ent_coef_schedule_timesteps=4000000`
  - W&B group: `b97-l11l12-complete25-slowent-two-seed`
  - Jobs: `51`, `52`

The spec validated with `rlab.job_queue.load_spec_document` and was enqueued
profileless for `rtx4090` with runtime image digest `c672be38cd0f`.
