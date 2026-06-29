# 2026-06-27 B87-B99 Progress Snapshot

Goal metric: peak `train/info/level_complete/rate/min/last`, requiring both
seeds in the same two-seed batch to exceed `0.80` within 5M training steps.

## Live Status

Queue snapshot after adding B99:

- Running: 5 train jobs on beast-3.
- Pending: 21 train jobs.
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

Full row scans over the five active runs, at about 3.40M-3.44M observed steps:

| Run | Peak min-rate | Peak step | Last min-rate | Last L1-1 | Last L1-2 | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `b87_l11l12_lowpress_s198_20260627T110807Z` | `0.09` | `2,121,920` | `0.05` | `0.63` | `0.05` | L1-2 bottleneck |
| `b87_l11l12_lowpress_s199_20260627T110807Z` | `0.12` | `2,512,032` | `0.02` | `0.51` | `0.02` | L1-2 bottleneck |
| `b88_l11l12_l12bias_s200_20260627T110819Z` | `0.06` | `1,925,408` | `0.01` | `0.54` | `0.01` | L1-2 bottleneck |
| `b88_l11l12_l12bias_s201_20260627T110819Z` | `0.09` | `3,084,848` | `0.03` | `0.18` | `0.03` | weak balanced lift |
| `b89_l11l12_complete25_s202_20260627T110832Z` | `0.41` | `2,512,208` | `0.00` | `0.85` | `0.00` | best peak so far; late L1-2 collapse |

Decision: no batch is solved. B89 seed 202 remains the best active
high-watermark, but it is well below the strict `>0.80` target and the paired
B89 seed is still pending.

## Added Backfill

B99 targets the late B89 imbalance:

- Keep `completion_reward=25`, because B89 produced the best observed active
  high-watermark.
- Add a soft Level1-2 sampling bias, `state_probs=[0.45,0.55]`, because the
  current B89 run retains high Level1-1 rate while Level1-2 collapses.

Added spec:

- `specs/b99-complete25-l12soft-l11l12-two-seed.yaml`
  - Seeds: `202,203`
  - Deltas from B89: `state_probs=[0.45,0.55]`
  - W&B group: `b99-l11l12-complete25-l12soft-two-seed`
  - Jobs: `55`, `56`

The spec validated with `rlab.job_queue.load_spec_document` and was enqueued
profileless for `rtx4090` with runtime image digest `c672be38cd0f`.
