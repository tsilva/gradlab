# 2026-06-27 B91 Near-Miss Monitor Snapshot

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
| `b89_l11l12_complete25_s203_20260627T110832Z` | running | `1,879,968` | `0.28` | `0.31` | `0.28` | `0.09` | `0.82` | `0.09` |
| `b90_l11l12_lowpress_l12bias_s204_20260627T112001Z` | running | `2,614,944` | `0.04` | `0.04` | `0.04` | `0.01` | `0.05` | `0.01` |
| `b90_l11l12_lowpress_l12bias_s205_20260627T112001Z` | running | `2,632,064` | `0.08` | `0.08` | `0.08` | `0.02` | `0.02` | `0.12` |
| `b91_l11l12_lowpress_complete25_s206_20260627T112013Z` | running | `3,337,680` | `0.72` | `0.72` | `0.78` | `0.69` | `0.72` | `0.69` |
| `b91_l11l12_lowpress_complete25_s207_20260627T112013Z` | running | `3,001,392` | `0.67` | `0.76` | `0.67` | `0.05` | `0.70` | `0.05` |

Interpretation:

- B91 is now a real near-miss branch. Seed 206 improved from `0.58` to `0.72`,
  and its per-level peak values are both close to the strict threshold.
- Seed 207 still has a useful `0.67` high-watermark but later lost Level1-2,
  so the leading failure mode remains one-level collapse after competence
  appears.
- B89 and B90 remain noncompetitive. B89 specializes toward Level1-1, and B90
  still fails to produce balanced competence.

## Decision

No new batch was queued in this iteration.

Reason:

- The live B91 pair is still improving around 3.3M steps and has not reached
  the 5M cap.
- The queue already contains the direct stabilizer arms for the observed B91
  failure mode:
  - B100: low pressure, completion reward, soft Level1-2 sampling.
  - B107: low pressure, completion reward, soft Level1-1 sampling.
  - B108: low pressure, completion reward, slower entropy decay.
  - B109: completion reward with gentler PPO updates.

Next monitor condition:

- If either B91 seed crosses `0.80`, immediately rescan both B91 histories for
  strict same-batch success.
- If B91 finishes or collapses below the target, prioritize B100 and B107-B109
  when those queued jobs start before adding further variants.
