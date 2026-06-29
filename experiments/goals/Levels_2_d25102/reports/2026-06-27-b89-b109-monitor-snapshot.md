# 2026-06-27 B89-B109 Monitor Snapshot

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
| `b89_l11l12_complete25_s203_20260627T110832Z` | running | `1,879,968` | `0.28` | `0.31` | `0.28` | `0.02` | `0.71` | `0.02` |
| `b90_l11l12_lowpress_l12bias_s204_20260627T112001Z` | running | `2,614,944` | `0.04` | `0.04` | `0.04` | `0.00` | `0.01` | `0.00` |
| `b90_l11l12_lowpress_l12bias_s205_20260627T112001Z` | running | `2,632,064` | `0.08` | `0.08` | `0.08` | `0.06` | `0.06` | `0.07` |
| `b91_l11l12_lowpress_complete25_s206_20260627T112013Z` | running | `2,753,920` | `0.58` | `0.58` | `0.65` | `0.30` | `0.49` | `0.30` |
| `b91_l11l12_lowpress_complete25_s207_20260627T112013Z` | running | `3,001,392` | `0.67` | `0.76` | `0.67` | `0.52` | `0.67` | `0.52` |

Interpretation:

- B91 remains the clear leading branch. Seed 207 improved from `0.46` to `0.67`
  peak, and seed 206 still holds a `0.58` peak.
- B91 still does not solve the goal because neither seed exceeds strict `0.80`,
  but it is now close enough that the next best action is to continue monitoring
  it toward the 5M cap rather than blindly adding more speculative backfills.
- B107, B108, and B109 already cover the most direct B91-derived stabilizers:
  soft Level1-1 sampling, slower entropy decay, and gentler PPO updates.
- B89 and B90 are no longer leading branches. B89 continues to specialize
  toward Level1-1 with Level1-2 collapse, while B90 remains weak on the balanced
  min metric.

## Decision

No new batch was queued in this iteration. The queue is already deep, and the
best current evidence says to let B91 continue because it is still producing
new high-watermarks around 3M steps.

Next monitor condition:

- If either B91 seed crosses `0.80`, scan both B91 histories immediately and
  check whether the paired seed has also crossed strict `>0.80`.
- If B91 collapses without crossing, prioritize the already queued B107-B109
  stabilizer arms as the next evidence source before creating further variants.
