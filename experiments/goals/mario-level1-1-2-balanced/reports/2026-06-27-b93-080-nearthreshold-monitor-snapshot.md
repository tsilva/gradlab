# 2026-06-27 B93 0.80 Near-Threshold Monitor Snapshot

Goal metric: peak `train/info/level_complete/rate/min/last`, requiring both
seeds in the same two-seed batch to exceed strict `0.80` within the 5M training
cap. This snapshot uses full-row W&B history scans and judges by high-watermark,
not final value.

Scan time: `2026-06-27T14:28:26Z`.

## Live Queue And Fleet

beast-3 remains healthy and already reconciled.

- Train jobs: `6` failed, `10` succeeded, `5` running, `42` pending.
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

Next queued arm:

- `job=87`: `b115_l11l12_slowent_l12bias_complete25_s200_20260627T142436Z`
- `job=88`: `b115_l11l12_slowent_l12bias_complete25_s201_20260627T142436Z`

## W&B High-Watermarks

No same-batch candidate has solved the goal yet. B93 is now the clear live
leader, but seed 201 reached exactly `0.80`, not strict `>0.80`, and seed 200 is
still at `0.72`.

| Run | Step | Peak min-rate | Peak step | Latest min-rate | Latest L1-1 rate | Latest L1-2 rate | L1-1 clears | L1-2 clears | FPS | Reward | Approx KL | Clip frac | EV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `b110_l11l12_gentlerpress_complete25_l12soft_s206_20260627T133104Z` | `3,798,560` | `0.17` | `2,007,472` | `0.02` | `0.72` | `0.02` | `332` | `92` | `1,213` | `2452.188` | `0.0059` | `0.0413` | `0.9876` |
| `b110_l11l12_gentlerpress_complete25_l12soft_s207_20260627T133104Z` | `3,776,512` | `0.17` | `1,786,272` | `0.00` | `0.92` | `0.00` | `635` | `106` | `1,201` | `2445.978` | `0.0118` | `0.0792` | `0.9631` |
| `b92_l11l12_slowent_s199_20260627T112706Z` | `3,776,512` | `0.16` | `3,352,448` | `0.13` | `0.83` | `0.13` | `1701` | `62` | `1,195` | `2224.098` | `0.0279` | `0.1771` | `0.9392` |
| `b93_l11l12_slowent_l12bias_s200_20260627T112704Z` | `3,768,320` | `0.72` | `3,766,560` | `0.72` | `0.74` | `0.72` | `736` | `363` | `1,191` | `2566.310` | `0.0171` | `0.1493` | `0.9825` |
| `b93_l11l12_slowent_l12bias_s201_20260627T112704Z` | `3,790,160` | `0.80` | `3,756,288` | `0.71` | `0.76` | `0.71` | `501` | `475` | `1,193` | `2560.309` | `0.0178` | `0.1595` | `0.9766` |

B91 near-miss reference from the same scan:

| Run | State | Step | Peak min-rate | Peak step | Latest min-rate | Peak L1-1 | Peak L1-2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `b91_l11l12_lowpress_complete25_s206_20260627T112013Z` | finished | `4,599,488` | `0.88` | `4,089,456` | `0.00` | `0.89` | `0.88` |
| `b91_l11l12_lowpress_complete25_s207_20260627T112013Z` | crashed | `4,600,464` | `0.67` | `3,001,392` | `0.03` | `0.76` | `0.67` |

## Interpretation

- B93 is the best live recipe family so far. It has one seed at exact `0.80`
  and the other at `0.72`, both with balanced per-level rates at their peaks.
- This is not a completion proof: the goal requires strict `>0.80` on both
  seeds in the same batch. Exact `0.80` is a near miss, not a pass.
- B93's latest values are still high enough to justify continuing to cap. Seed
  200 is currently at its peak (`0.72`), and seed 201 is still latest `0.71`.
- B110 and B92 are not comparable leaders despite strong Level1-1 rates; they
  remain Level1-2-limited.
- B115 is the right queued follow-up because it preserves B93's slow-entropy
  `40/60` Level1-2-biased shape and adds only B91's `completion_reward=25`.

## Decision

No additional batch was queued in this pass.

Reason:

- B93 is still actively improving near the target threshold; interrupting or
  replacing it would throw away exactly the late-recovery evidence the goal is
  designed to capture.
- B115 is already queued next and directly tests the best current hypothesis:
  B93 retention plus B91's clean-completion reward.
- More backlog before B93 or B115 finishes would mostly dilute the signal.

Next monitor condition:

- Keep scanning B93 through cap. If both B93 seeds cross strict `>0.80`, run the
  completion audit immediately.
- If B93 remains a near miss, compare B115 against B93 at roughly `3.0M-4.0M`
  by paired peak min-rate and whether seed 200 clears the current `0.72`
  ceiling.
