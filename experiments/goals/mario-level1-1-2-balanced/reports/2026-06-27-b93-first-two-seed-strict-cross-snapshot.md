# 2026-06-27 B93 First Two-Seed Strict Cross Snapshot

Goal metric: peak `train/info/level_complete/rate/min/last`, requiring both
seeds in the same two-seed batch to exceed strict `0.80` within the 5M training
cap. This snapshot uses full-row W&B history scans and judges by high-watermark,
not final value.

Scan time: `2026-06-27T14:34:31Z`.

## Decision Status

B93 is the first observed same-spec two-seed batch to cross the goal threshold
on both seeds.

This is strong threshold evidence, but the goal should remain open until the two
B93 jobs finish normally at the 5M cap, because the checked-in goal contract says
not to early-stop on the first high-watermark crossing and the success protocol
expects completed runs.

## Live Queue And Fleet

beast-3 is healthy and already reconciled.

- Train jobs: `6` failed, `10` succeeded, `5` running, `42` pending.
- Eval jobs: none.
- Fleet plan: no action; keep the existing beast-3 RTX4090 managed runner.
- Runner digest: `c672be38cd0f`.
- Disk check: `/` has about `39G` free and is at `82%` use.

Active beast-3 workers:

- `job=77`: `b110_l11l12_gentlerpress_complete25_l12soft_s206_20260627T133104Z`
- `job=78`: `b110_l11l12_gentlerpress_complete25_l12soft_s207_20260627T133104Z`
- `job=44`: `b92_l11l12_slowent_s199_20260627T112706Z`
- `job=41`: `b93_l11l12_slowent_l12bias_s200_20260627T112704Z`
- `job=42`: `b93_l11l12_slowent_l12bias_s201_20260627T112704Z`

Next queued arm:

- `job=87`: `b115_l11l12_slowent_l12bias_complete25_s200_20260627T142436Z`
- `job=88`: `b115_l11l12_slowent_l12bias_complete25_s201_20260627T142436Z`

## B93 High-Watermark Evidence

Spec:
`experiments/goals/mario-level1-1-2-balanced/specs/b93-slowent-l12bias-l11l12-two-seed.json`

W&B group: `b93-l11l12-slowent-l12bias-two-seed`

| Run | State | Latest step | Peak min-rate | Peak step | Peak L1-1 | Peak L1-2 | Latest min-rate | Latest L1-1 | Latest L1-2 | L1-1 clears | L1-2 clears |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `b93_l11l12_slowent_l12bias_s200_20260627T112704Z` | running | `4,157,968` | `0.81` | `3,986,208` | `0.81` | `0.82` | `0.72` | `0.72` | `0.82` | `899` | `626` |
| `b93_l11l12_slowent_l12bias_s201_20260627T112704Z` | running | `4,177,920` | `0.85` | `3,984,944` | `0.87` | `0.85` | `0.69` | `0.83` | `0.69` | `721` | `817` |

Both B93 seeds have now exceeded strict `>0.80` by full-history W&B scan:

- Seed `200`: peak `0.81` at `3,986,208` steps.
- Seed `201`: peak `0.85` at `3,984,944` steps.

The later drop in latest values is not disqualifying for this goal, because the
selection metric is explicitly the training high-watermark. It is still a useful
stability diagnostic.

## Other Active Arms

| Run | Latest step | Peak min-rate | Peak step | Latest min-rate | Latest L1-1 | Latest L1-2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `b110_l11l12_gentlerpress_complete25_l12soft_s206_20260627T133104Z` | `4,268,032` | `0.17` | `2,007,472` | `0.04` | `0.75` | `0.04` |
| `b110_l11l12_gentlerpress_complete25_l12soft_s207_20260627T133104Z` | `4,232,432` | `0.17` | `1,786,272` | `0.01` | `0.77` | `0.01` |
| `b92_l11l12_slowent_s199_20260627T112706Z` | `4,231,824` | `0.16` | `3,352,448` | `0.06` | `0.90` | `0.06` |

B110 and B92 are not contenders; they remain Level1-2-limited despite decent
Level1-1 competence.

## Config Delta

B93 differs from the B86 near-miss incumbent in two interpretable ways:

- State sampling moved from `state_probs=[0.5, 0.5]` to
  `state_probs=[0.4, 0.6]`, giving Level1-2 more training attempts.
- Entropy decays more slowly and ends higher:
  `ent_coef_final=0.001` over `4,000,000` schedule steps instead of
  `ent_coef_final=0.0003` over `2,000,000` schedule steps.

Everything else important stayed conservative: `target_kl=0.2`,
`clip_range=0.15`, constant `learning_rate=0.00015`, task conditioning,
per-task advantage normalization, and no `completion_reward`.

## Next Action

Final audit completed at `2026-06-27T14:46:49Z`.

Queue status marked both B93 results as `succeeded`, and W&B marked both runs as
`finished`.

Final full-history peaks:

| Run | Final W&B state | Latest step | Peak min-rate | Peak step | Peak L1-1 | Peak L1-2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `b93_l11l12_slowent_l12bias_s200_20260627T112704Z` | finished | `5,005,312` | `0.87` | `4,653,360` | `0.87` | `0.87` |
| `b93_l11l12_slowent_l12bias_s201_20260627T112704Z` | finished | `5,005,312` | `0.94` | `4,967,360` | `0.94` | `0.95` |

B93 is the winning recipe for this goal. The reusable recipe note is
`experiments/goals/mario-level1-1-2-balanced/recipes/b93-slowent-l12bias-two-seed-success.md`.
