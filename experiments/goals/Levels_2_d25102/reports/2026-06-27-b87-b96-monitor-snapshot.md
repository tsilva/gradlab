# 2026-06-27 B87-B96 Monitor Snapshot

Goal metric: peak `train/info/level_complete/rate/min/last`, requiring both
seeds in the same two-seed batch to exceed `0.80` within 5M training steps.

## Live Status

Queue snapshot after adding B96:

- Running: 5 train jobs on beast-3.
- Pending: 15 train jobs.
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

Full row scans over the five active runs, at about 1.92M-1.96M observed steps:

| Run | Peak min-rate | Peak step | Last min-rate | Last L1-1 | Last L1-2 | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `b87_l11l12_lowpress_s198_20260627T110807Z` | `0.05` | `1,598,576` | `0.03` | `0.03` | `0.09` | still far below goal |
| `b87_l11l12_lowpress_s199_20260627T110807Z` | `0.07` | `1,603,792` | `0.02` | `0.21` | `0.02` | best active peak, but L1-2 bottleneck |
| `b88_l11l12_l12bias_s200_20260627T110819Z` | `0.06` | `1,925,408` | `0.05` | `0.06` | `0.05` | modest balanced lift from 40/60 bias |
| `b88_l11l12_l12bias_s201_20260627T110819Z` | `0.02` | `1,842,464` | `0.02` | `0.02` | `0.19` | L1-1 bottleneck under L1-2 bias |
| `b89_l11l12_complete25_s202_20260627T110832Z` | `0.04` | `1,925,664` | `0.04` | `0.04` | `0.57` | completion reward strongly lifted L1-2, not L1-1 |

Decision: no batch is solved. The active runs remain far below the strict
`>0.80` high-watermark, but they are still much earlier than the B86 near-miss
peak near 4.07M. Keep the active jobs running.

## Added Backfill

B96 tests the most actionable imbalance in the latest snapshot: the B89
completion-reward arm has a high early Level1-2 rate (`0.57`) but very low
Level1-1 rate (`0.04`). A slight Level1-1 sampling bias may preserve the
completion-reward benefit while raising the bottleneck.

Added spec:

- `specs/b96-complete25-l11soft-l11l12-two-seed.yaml`
  - Seeds: `206,207`
  - Deltas from B89: `state_probs=[0.55,0.45]`
  - W&B group: `b96-l11l12-complete25-l11soft-two-seed`
  - Jobs: `49`, `50`

The spec validated with `rlab.job_queue.load_spec_document` and was enqueued
profileless for `rtx4090` with runtime image digest `c672be38cd0f`.
