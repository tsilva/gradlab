# 2026-06-27 B87-B96 Progress Snapshot

Goal metric: peak `train/info/level_complete/rate/min/last`, requiring both
seeds in the same two-seed batch to exceed `0.80` within 5M training steps.

## Live Status

Queue snapshot:

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

Full row scans over the five active runs, at about 2.31M-2.35M observed steps:

| Run | Peak min-rate | Peak step | Last min-rate | Last L1-1 | Last L1-2 | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `b87_l11l12_lowpress_s198_20260627T110807Z` | `0.09` | `2,121,920` | `0.04` | `0.25` | `0.04` | L1-2 bottleneck |
| `b87_l11l12_lowpress_s199_20260627T110807Z` | `0.10` | `2,258,656` | `0.05` | `0.28` | `0.05` | L1-2 bottleneck |
| `b88_l11l12_l12bias_s200_20260627T110819Z` | `0.06` | `1,925,408` | `0.03` | `0.05` | `0.03` | no clear lift yet |
| `b88_l11l12_l12bias_s201_20260627T110819Z` | `0.02` | `1,842,464` | `0.02` | `0.02` | `0.67` | L1-1 bottleneck under L1-2 bias |
| `b89_l11l12_complete25_s202_20260627T110832Z` | `0.22` | `2,322,400` | `0.19` | `0.19` | `0.50` | best active signal; completion reward is promising |

Decision: no batch is solved. The best active run is B89 seed 202, but it is
still far below the strict `>0.80` target and the paired B89 seed is still
pending. Keep the active jobs running because previous runs peaked much later
near 4M steps.

## Queue Decision

Do not enqueue another batch in this snapshot. The queue already contains 15
pending jobs, including:

- B89 seed 203, the paired completion-reward seed needed to judge the B89 arm.
- B91, combining low update pressure with `completion_reward=25`.
- B96, testing B89 plus a slight Level1-1 sampling bias to address the current
  B89 imbalance.

The next useful action is to re-scan after either B89 seed 202 approaches the
late-training region or beast-3 rolls into B89 seed 203/B90+.
