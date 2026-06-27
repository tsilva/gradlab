# 2026-06-27 B91 0.86 Monitor Snapshot

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
| `b89_l11l12_complete25_s203_20260627T110832Z` | running | `1,879,968` | `0.28` | `0.31` | `0.28` | `0.02` | `0.96` | `0.02` |
| `b90_l11l12_lowpress_l12bias_s204_20260627T112001Z` | running | `2,614,944` | `0.04` | `0.04` | `0.04` | `0.03` | `0.26` | `0.03` |
| `b90_l11l12_lowpress_l12bias_s205_20260627T112001Z` | running | `2,632,064` | `0.08` | `0.08` | `0.08` | `0.01` | `0.15` | `0.01` |
| `b91_l11l12_lowpress_complete25_s206_20260627T112013Z` | running | `3,944,464` | `0.86` | `0.86` | `0.86` | `0.81` | `0.83` | `0.81` |
| `b91_l11l12_lowpress_complete25_s207_20260627T112013Z` | running | `3,001,392` | `0.67` | `0.76` | `0.67` | `0.22` | `0.84` | `0.23` |

Interpretation:

- B91 seed 206 is now clearly over the strict threshold, with a peak min-rate
  of `0.86` and both per-level rates at `0.86` at the peak.
- B91 seed 207 remains the blocker for this same-batch recipe. Its peak
  min-rate remains `0.67`, and latest Level1-2 is still weak.
- B89 and B90 remain noncompetitive. B89 has become a Level1-1 specialist and
  B90 never produced balanced two-level competence.

## Decision

No new batch was queued in this iteration.

Reason:

- The same B91 batch is still running; seed 207 can recover before 5M and the
  goal is explicitly high-watermark based.
- The pending B100/B107/B108/B109 arms already cover the obvious fixes for
  seed-207 Level1-2 retention and one-level collapse.
- The best next action is to continue monitoring until B91 finishes or rolls to
  the queued stabilizer arms.

Next monitor condition:

- If B91 seed 207 crosses strict `0.80`, perform a completion audit for B91.
- If B91 finishes with seed 207 below threshold, promote B100/B107/B108/B109 as
  the next evidence source before adding further variants.
