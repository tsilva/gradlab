# 2026-06-27 B91 0.76 Monitor Snapshot

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
| `b89_l11l12_complete25_s203_20260627T110832Z` | running | `1,879,968` | `0.28` | `0.31` | `0.28` | `0.02` | `0.88` | `0.02` |
| `b90_l11l12_lowpress_l12bias_s204_20260627T112001Z` | running | `2,614,944` | `0.04` | `0.04` | `0.04` | `0.03` | `0.16` | `0.03` |
| `b90_l11l12_lowpress_l12bias_s205_20260627T112001Z` | running | `2,632,064` | `0.08` | `0.08` | `0.08` | `0.04` | `0.04` | `0.08` |
| `b91_l11l12_lowpress_complete25_s206_20260627T112013Z` | running | `3,535,696` | `0.76` | `0.76` | `0.76` | `0.73` | `0.73` | `0.78` |
| `b91_l11l12_lowpress_complete25_s207_20260627T112013Z` | running | `3,001,392` | `0.67` | `0.76` | `0.67` | `0.05` | `0.71` | `0.05` |

Interpretation:

- B91 seed 206 is very close but not solved: it moved from `0.72` to `0.76`
  and currently has both per-level latest rates near the peak.
- B91 seed 207 remains the blocker for the same-batch success criterion. Its
  best min-rate is `0.67`, and its latest Level1-2 rate has collapsed to `0.05`.
- B89 and B90 remain noncompetitive branches.

## Decision

No new batch was queued in this iteration.

Reason:

- The current B91 pair has not reached the 5M cap.
- The next useful variants are already queued but not yet running: B100 for
  soft Level1-2 sampling, B107 for soft Level1-1 sampling, B108 for slower
  entropy decay, and B109 for gentler PPO updates.
- Another immediate variant would be less informative than letting the queued
  stabilizer arms start.

Next monitor condition:

- If B91 seed 206 crosses `0.80`, continue scanning B91 seed 207 for paired
  strict success.
- If B91 seed 207 does not recover, prioritize B100 and B107-B109 when they
  start before adding more variants.
