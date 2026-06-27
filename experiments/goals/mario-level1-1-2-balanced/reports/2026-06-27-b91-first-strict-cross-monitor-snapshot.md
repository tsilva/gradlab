# 2026-06-27 B91 First Strict Cross Monitor Snapshot

Goal metric: peak `train/info/level_complete/rate/min/last`, requiring both
seeds in the same two-seed batch to exceed strict `0.80` within the 5M training
cap. This snapshot uses full-row W&B history scans for changed active runs.

## Live Queue

No new jobs were added in this iteration.

- Succeeded train jobs: `10`
- Running train jobs: `5`
- Pending train jobs: `36`
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

No two-seed batch has solved the goal yet.

| Run | State | Step | Peak min-rate | Peak L1-1 | Peak L1-2 | Last min-rate | Last L1-1 | Last L1-2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `b89_l11l12_complete25_s203_20260627T110832Z` | running | `1,879,968` | `0.28` | `0.31` | `0.28` | `0.01` | `0.84` | `0.01` |
| `b90_l11l12_lowpress_l12bias_s204_20260627T112001Z` | running | `2,614,944` | `0.04` | `0.04` | `0.04` | `0.02` | `0.20` | `0.02` |
| `b90_l11l12_lowpress_l12bias_s205_20260627T112001Z` | running | `2,632,064` | `0.08` | `0.08` | `0.08` | `0.03` | `0.04` | `0.03` |
| `b91_l11l12_lowpress_complete25_s206_20260627T112013Z` | running | `3,790,400` | `0.82` | `0.82` | `0.83` | `0.82` | `0.82` | `0.83` |
| `b91_l11l12_lowpress_complete25_s207_20260627T112013Z` | running | `3,001,392` | `0.67` | `0.76` | `0.67` | `0.03` | `0.84` | `0.03` |

Interpretation:

- B91 produced the first strict crossing in this search. Seed 206 reached
  `0.82` on the primary min-rate metric and has both per-level rates above
  `0.80`.
- The same B91 batch is not solved because seed 207 has not exceeded `0.80`.
  Its high-watermark remains `0.67`, and its latest Level1-2 rate has collapsed
  to `0.03`.
- This strengthens the case that the recipe is close and that the blocking
  failure mode is Level1-2 retention on the second seed, not absence of a
  learnable mixed policy.

## Decision

No new batch was queued in this iteration.

Reason:

- B91 seed 207 can still recover before 5M, and the goal explicitly does not
  allow early stopping on collapse.
- The queue already contains the most direct stabilizer arms for this failure:
  B100 for soft Level1-2 sampling, B108 for slower entropy decay, and B109 for
  gentler PPO updates. B107 also tests the opposite sampling tilt.
- Adding another arm before those queued variants start would dilute the
  evidence more than it would improve coverage.

Next monitor condition:

- Continue scanning B91 until it finishes or seed 207 recovers past strict
  `0.80`.
- If B91 finishes unsolved, prioritize the already queued B100/B107/B108/B109
  stabilizers as soon as beast-3 rolls to them.
