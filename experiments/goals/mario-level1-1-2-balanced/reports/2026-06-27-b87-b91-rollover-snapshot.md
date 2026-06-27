# 2026-06-27 B87-B91 Rollover Snapshot

Goal metric: peak `train/info/level_complete/rate/min/last`, requiring both
seeds in the same two-seed batch to exceed strict `0.80` within the 5M training
cap. W&B history high-watermark remains the selection source; final values are
diagnostic only.

## Queue And Fleet State

After the first B87/B88/B89 tranche reached the cap, beast-3 rolled into the
next jobs.

Queue status:

- Succeeded train jobs: `10`
- Running train jobs: `5`
- Pending train jobs: `22`
- Eval jobs: none

Active beast-3 workers:

- `job=36`: `b89_l11l12_complete25_s203_20260627T110832Z`
- `job=37`: `b90_l11l12_lowpress_l12bias_s204_20260627T112001Z`
- `job=38`: `b90_l11l12_lowpress_l12bias_s205_20260627T112001Z`
- `job=39`: `b91_l11l12_lowpress_complete25_s206_20260627T112013Z`
- `job=40`: `b91_l11l12_lowpress_complete25_s207_20260627T112013Z`

Fleet plan still requires no action: keep the existing beast-3 RTX4090
container with five workers on digest `c672be38cd0f`.

## Completed Tranche Result

The first B87/B88/B89 tranche did not solve the goal.

| Group | Run | Final/last step | Peak min-rate | Peak step | Last min-rate | Last L1-1 | Last L1-2 | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B87 | `b87_l11l12_lowpress_s198_20260627T110807Z` | `4,999,584` | `0.09` | `2,121,920` | `0.01` | `0.62` | `0.01` | failed; L1-2 bottleneck |
| B87 | `b87_l11l12_lowpress_s199_20260627T110807Z` | `4,956,160` | `0.12` | `2,512,032` | `0.01` | `0.65` | `0.01` | failed; L1-2 bottleneck |
| B88 | `b88_l11l12_l12bias_s200_20260627T110819Z` | `4,964,352` | `0.07` | `3,968,832` | `0.01` | `0.91` | `0.01` | failed; L1-2 sampling bias alone not enough |
| B88 | `b88_l11l12_l12bias_s201_20260627T110819Z` | `4,943,104` | `0.09` | `3,084,848` | `0.04` | `0.58` | `0.04` | failed; weak balanced lift |
| B89 | `b89_l11l12_complete25_s202_20260627T110832Z` | `5,005,312` | `0.41` | `2,512,208` | `0.02` | `0.84` | `0.02` | failed but best active signal |

Interpretation:

- Lower PPO update pressure alone did not improve the balanced bottleneck.
- A Level1-2 sampling bias alone did not improve the balanced bottleneck.
- `completion_reward=25` remains the only tested lever that materially raised
  the goal metric, but it still collapsed on Level1-2 and did not approach the
  strict `>0.80` target.

## Newly Active Tranche

B89 seed 203 and the B90/B91 two-seed batches have just started, so their W&B
histories are too young for a useful rolling 100-attempt judgment.

| Group | Run | Step observed | Peak min-rate | Last min-rate | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| B89 | `b89_l11l12_complete25_s203_20260627T110832Z` | `148,336` | `0.00` | `0.00` | paired seed just started |
| B90 | `b90_l11l12_lowpress_l12bias_s204_20260627T112001Z` | `90,384` | `0.00` | `0.00` | too early; lowpress + L1-2 bias |
| B90 | `b90_l11l12_lowpress_l12bias_s205_20260627T112001Z` | `90,208` | n/a | n/a | too early; no full rate window yet |
| B91 | `b91_l11l12_lowpress_complete25_s206_20260627T112013Z` | `71,088` | `0.00` | `0.00` | too early; lowpress + completion reward |
| B91 | `b91_l11l12_lowpress_complete25_s207_20260627T112013Z` | `65,536` | n/a | n/a | too early; no full rate window yet |

## Decision

Do not add another batch from this snapshot. B100-B102 already queue the most
compact completion-reward anti-collapse combinations, and the active B90/B91
tranche is just beginning. The next useful action is to monitor B89 seed 203,
B90, and B91 until their histories have enough attempt-window coverage, then
choose whether to continue emphasizing completion reward, stronger completion
bonus, entropy preservation, or sampling bias.
