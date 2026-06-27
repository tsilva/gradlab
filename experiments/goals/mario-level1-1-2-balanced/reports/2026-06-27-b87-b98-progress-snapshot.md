# 2026-06-27 B87-B98 Progress Snapshot

Goal metric: peak `train/info/level_complete/rate/min/last`, requiring both
seeds in the same two-seed batch to exceed `0.80` within 5M training steps.

## Live Status

Queue snapshot after adding B98:

- Running: 5 train jobs on beast-3.
- Pending: 19 train jobs.
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

Full row scans over the five active runs, at about 3.04M-3.09M observed steps:

| Run | Peak min-rate | Peak step | Last min-rate | Last L1-1 | Last L1-2 | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `b87_l11l12_lowpress_s198_20260627T110807Z` | `0.09` | `2,121,920` | `0.08` | `0.81` | `0.08` | strong L1-1, L1-2 bottleneck |
| `b87_l11l12_lowpress_s199_20260627T110807Z` | `0.12` | `2,512,032` | `0.05` | `0.49` | `0.05` | L1-2 bottleneck |
| `b88_l11l12_l12bias_s200_20260627T110819Z` | `0.06` | `1,925,408` | `0.00` | `0.39` | `0.00` | not promising |
| `b88_l11l12_l12bias_s201_20260627T110819Z` | `0.09` | `3,084,848` | `0.09` | `0.18` | `0.09` | no strong lift yet |
| `b89_l11l12_complete25_s202_20260627T110832Z` | `0.41` | `2,512,208` | `0.01` | `0.79` | `0.01` | best peak so far; collapsed on L1-2 |

Decision: no batch is solved. B89 seed 202 remains the best active
high-watermark, but it is well below the strict `>0.80` target and the paired
B89 seed is still pending.

## Added Backfill

B98 tests the most direct reward-scale follow-up from B89:

- Keep the same environment, termination, state sampling, optimizer, and metric
  contract.
- Increase only the true clean-completion bonus from `25` to `50`.
- This is not a proxy reward; it rewards the semantic clean clear event counted
  by the target metric.

Added spec:

- `specs/b98-complete50-l11l12-two-seed.json`
  - Seeds: `200,201`
  - Deltas from B89: `completion_reward=50`
  - W&B group: `b98-l11l12-complete50-two-seed`
  - Jobs: `53`, `54`

The spec validated with `rlab.job_queue.load_spec_document` and was enqueued
profileless for `rtx4090` with runtime image digest `c672be38cd0f`.
