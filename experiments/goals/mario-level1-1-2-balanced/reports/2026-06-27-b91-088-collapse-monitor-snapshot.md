# 2026-06-27 B91 0.88 Collapse Monitor Snapshot

Goal metric: peak `train/info/level_complete/rate/min/last`, requiring both
seeds in the same two-seed batch to exceed strict `0.80` within the 5M training
cap. This snapshot uses full-row W&B history scans, with per-level rates carried
forward to interpret sparse rows at each high-watermark.

Scan time: `2026-06-27T13:22:19Z`.

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
digest `c672be38cd0f`; no reconcile action required. Beast-2 still timed out
during fleet inspection, but the active goal target is beast-3/RTX4090.

## Current W&B High-Watermarks

No two-seed batch has solved the goal yet.

| Run | State | Latest step | Peak step | Peak min-rate | Peak L1-1 | Peak L1-2 | Latest min-rate | Latest L1-1 | Latest L1-2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `b89_l11l12_complete25_s203_20260627T110832Z` | running | `4,538,080` | `1,879,968` | `0.28` | `0.31` | `0.28` | `0.03` | `0.90` | `0.03` |
| `b90_l11l12_lowpress_l12bias_s204_20260627T112001Z` | running | `4,479,472` | `2,614,944` | `0.04` | `0.04` | `0.04` | `0.01` | `0.26` | `0.01` |
| `b90_l11l12_lowpress_l12bias_s205_20260627T112001Z` | running | `4,542,464` | `2,632,064` | `0.08` | `0.08` | `0.08` | `0.01` | `0.05` | `0.01` |
| `b91_l11l12_lowpress_complete25_s206_20260627T112013Z` | running | `4,462,560` | `4,089,456` | `0.88` | `0.89` | `0.88` | `0.00` | `0.87` | `0.00` |
| `b91_l11l12_lowpress_complete25_s207_20260627T112013Z` | running | `4,476,976` | `3,001,392` | `0.67` | `0.76` | `0.67` | `0.04` | `0.84` | `0.04` |

W&B links:

- B91 seed 206: <https://wandb.ai/tsilva/SuperMarioBros-NES/runs/al1h7yu0>
- B91 seed 207: <https://wandb.ai/tsilva/SuperMarioBros-NES/runs/hdmyn6h3>

## Interpretation

- B91 seed 206 is the strongest individual mixed-level training signal so far:
  it reached `0.88` min-rate, with carried Level1-1 and Level1-2 rates of
  `0.89` and `0.88` at the peak.
- The same seed then collapsed completely on Level1-2 while retaining high
  Level1-1 competence. This is useful evidence for the user's warning: the
  high-watermark is the selection metric, but it is not a safe early-stop rule
  because collapse and recovery both happen during the remaining cap.
- B91 seed 207 remains the same-batch blocker. Its high-watermark is still
  `0.67`, and its current Level1-2 rolling completion is near zero.
- B89 and B90 are not competitive. B89 became a Level1-1 specialist, and B90
  never formed balanced competence.

## Decision

No additional batch was queued.

Reason:

- B91 is still running near 4.5M steps, and seed 207 still has remaining cap
  for a possible high-watermark recovery.
- The queue already has `36` pending jobs behind the current five runners.
- The queued stabilizer arms already target the observed failure mode:
  - B100: low pressure, completion reward, soft Level1-2 sampling.
  - B107: low pressure, completion reward, soft Level1-1 sampling.
  - B108: low pressure, completion reward, slower entropy decay.
  - B109: completion reward with gentler PPO pressure.

Next monitor condition:

- If B91 seed 207 crosses strict `0.80`, perform a completion audit for the
  B91 spec immediately.
- If B91 finishes with seed 207 below threshold, let beast-3 advance into the
  queued stabilizer arms and scan their full histories before adding more
  variants.
