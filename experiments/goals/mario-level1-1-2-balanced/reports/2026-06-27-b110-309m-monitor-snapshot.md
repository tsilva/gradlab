# 2026-06-27 B110 3.09M Monitor Snapshot

Goal metric: peak `train/info/level_complete/rate/min/last`, requiring both
seeds in the same two-seed batch to exceed strict `0.80` within the 5M training
cap. This snapshot uses full-row W&B history scans and judges by high-watermark,
not final value.

Scan time: `2026-06-27T14:18:43Z`.

## Live Queue And Fleet

beast-3 remains healthy and already reconciled.

- Train jobs: `6` failed, `10` succeeded, `5` running, `40` pending.
- Eval jobs: none.
- Fleet plan: no action; keep the existing beast-3 RTX4090 managed runner.
- Runner digest: `c672be38cd0f`.
- Disk check: `/` has about `40G` free and is at `82%` use.

Active beast-3 workers:

- `job=77`: `b110_l11l12_gentlerpress_complete25_l12soft_s206_20260627T133104Z`
- `job=78`: `b110_l11l12_gentlerpress_complete25_l12soft_s207_20260627T133104Z`
- `job=44`: `b92_l11l12_slowent_s199_20260627T112706Z`
- `job=41`: `b93_l11l12_slowent_l12bias_s200_20260627T112704Z`
- `job=42`: `b93_l11l12_slowent_l12bias_s201_20260627T112704Z`

Queued high-priority follow-ups remain:

- `job=79`: `b111_l11l12_gentlerpress_complete25_l11soft_s206_20260627T135253Z`
- `job=80`: `b111_l11l12_gentlerpress_complete25_l11soft_s207_20260627T135253Z`
- `job=81`: `b112_l11l12_lowpress_complete25_l12micro_s206_20260627T140401Z`
- `job=82`: `b112_l11l12_lowpress_complete25_l12micro_s207_20260627T140401Z`
- `job=83`: `b113_l11l12_midpress_complete25_l12micro_s206_20260627T141012Z`
- `job=84`: `b113_l11l12_midpress_complete25_l12micro_s207_20260627T141012Z`
- `job=85`: `b114_l11l12_lowpress_complete25_slowent_l12micro_s206_20260627T141519Z`
- `job=86`: `b114_l11l12_lowpress_complete25_slowent_l12micro_s207_20260627T141519Z`

## W&B High-Watermarks

No same-batch candidate has solved the goal yet.

| Run | Step | Peak min-rate | Peak step | Latest min-rate | Latest L1-1 rate | Latest L1-2 rate | L1-1 clears | L1-2 clears | FPS | Reward | Approx KL | Clip frac | EV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `b110_l11l12_gentlerpress_complete25_l12soft_s206_20260627T133104Z` | `3,088,384` | `0.17` | `2,007,472` | `0.02` | `0.35` | `0.02` | `148` | `85` | `1,212` | `2325.056` | `0.0098` | `0.0677` | `0.9708` |
| `b110_l11l12_gentlerpress_complete25_l12soft_s207_20260627T133104Z` | `3,072,000` | `0.17` | `1,786,272` | `0.01` | `0.65` | `0.01` | `375` | `96` | `1,200` | `2233.491` | `0.0113` | `0.1257` | `0.9705` |
| `b92_l11l12_slowent_s199_20260627T112706Z` | `3,056,272` | `0.05` | `2,630,960` | `0.02` | `0.72` | `0.02` | `1219` | `19` | `1,193` | `2236.753` | `0.0220` | `0.1640` | `0.9768` |
| `b93_l11l12_slowent_l12bias_s200_20260627T112704Z` | `3,048,272` | `0.20` | `2,022,256` | `0.08` | `0.38` | `0.08` | `426` | `121` | `1,190` | `2054.848` | `0.0150` | `0.1192` | `0.9551` |
| `b93_l11l12_slowent_l12bias_s201_20260627T112704Z` | `3,063,808` | `0.12` | `2,607,216` | `0.03` | `0.31` | `0.03` | `181` | `161` | `1,190` | `1930.516` | `0.0204` | `0.1396` | `0.9671` |

B91 near-miss reference from the same scan:

| Run | State | Step | Peak min-rate | Peak step | Latest min-rate | Peak L1-1 | Peak L1-2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `b91_l11l12_lowpress_complete25_s206_20260627T112013Z` | finished | `4,599,488` | `0.88` | `4,089,456` | `0.00` | `0.89` | `0.88` |
| `b91_l11l12_lowpress_complete25_s207_20260627T112013Z` | crashed | `4,600,464` | `0.67` | `3,001,392` | `0.03` | `0.76` | `0.67` |

## Interpretation

- B110 has not improved after the `2.70M` snapshot. Both seeds still peak at
  only `0.17`, and the latest bottleneck is Level1-2 (`0.02` and `0.01`).
- B110 seed 207 has strong Level1-1 movement (`0.65` latest) while Level1-2 is
  nearly absent, matching the specialization/collapse pattern from earlier
  failed mixed-level batches.
- B92 is even more Level1-1-specialized: latest Level1-1 is `0.72`, but
  Level1-2 is `0.02`.
- B93 seed 200 remains the strongest active run at `0.20`, but seed 201 is only
  `0.12`; this does not satisfy the same-batch two-seed requirement.
- PPO health metrics do not indicate a catastrophic optimizer failure. The
  problem is still balanced competence/retention, not W&B summary drift or a
  metric-reading issue.

## Decision

No new batch was queued in this pass.

Reason:

- The high-priority queue already contains four focused follow-ups that target
  the B91 clue and the observed Level1-2 retention failure:
  - B111: gentler pressure plus soft Level1-1 retention.
  - B112: B91 low-pressure recipe plus tiny `48/52` Level1-2 nudge.
  - B113: midpoint PPO pressure plus tiny `48/52` Level1-2 nudge.
  - B114: B112 plus slower entropy decay through the 2-4M window.
- Adding another recipe before any of B111-B114 starts would mostly deepen the
  backlog without using new evidence from those targeted arms.

Next monitor condition:

- Keep scanning B110/B92/B93 until the current slots finish.
- Once B111-B114 begin, compare them against B91 by peak min-rate around `3.0M`,
  seed 207's ability to exceed `0.67`, and whether Level1-2 avoids collapsing
  after the first peak.
