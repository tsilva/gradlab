# 2026-06-27 B110 0.5M Monitor Snapshot

Goal metric: peak `train/info/level_complete/rate/min/last`, requiring both
seeds in the same two-seed batch to exceed strict `0.80` within the 5M training
cap. This snapshot uses full-row W&B history scans and is not a recipe decision.

Scan time: `2026-06-27T13:42:56Z`.

## Live Queue And Fleet

beast-3 remains healthy.

- Train jobs: `6` failed, `10` succeeded, `5` running, `32` pending.
- Eval jobs: none.
- Fleet plan: no action; keep the existing beast-3 RTX4090 managed runner.
- Runner digest: `c672be38cd0f`.
- Disk check: `/` has about `42G` free and is at `81%` use.

Active beast-3 workers:

- `job=77`: `b110_l11l12_gentlerpress_complete25_l12soft_s206_20260627T133104Z`
- `job=78`: `b110_l11l12_gentlerpress_complete25_l12soft_s207_20260627T133104Z`
- `job=44`: `b92_l11l12_slowent_s199_20260627T112706Z`
- `job=41`: `b93_l11l12_slowent_l12bias_s200_20260627T112704Z`
- `job=42`: `b93_l11l12_slowent_l12bias_s201_20260627T112704Z`

## W&B High-Watermarks

No active run has meaningful balanced completion yet.

| Run | Step | Peak min-rate | Latest L1-1 rate | Latest L1-2 rate | L1-1 clears | L1-2 clears | FPS | Latest reward | Approx KL | Clip frac | EV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `b110_l11l12_gentlerpress_complete25_l12soft_s206_20260627T133104Z` | `494,368` | `0.00` | `0.01` | `0.00` | `3` | `0` | `1,231` | `699.190` | `0.0061` | `0.1480` | `0.753` |
| `b110_l11l12_gentlerpress_complete25_l12soft_s207_20260627T133104Z` | `482,944` | `0.00` | `0.00` | `0.00` | `2` | `0` | `1,199` | `611.007` | `0.0048` | `0.1492` | `0.708` |
| `b92_l11l12_slowent_s199_20260627T112706Z` | `471,792` | `0.00` | `0.00` | `0.00` | `2` | `0` | `1,172` | `697.927` | `0.0086` | `0.1488` | `0.759` |
| `b93_l11l12_slowent_l12bias_s200_20260627T112704Z` | `459,552` | `0.00` | `0.00` | `0.00` | `1` | `0` | `1,143` | `675.704` | `0.0116` | `0.1327` | `0.758` |
| `b93_l11l12_slowent_l12bias_s201_20260627T112704Z` | `481,072` | `0.00` | `0.01` | `0.00` | `2` | `0` | `1,153` | `761.460` | `0.0077` | `0.1563` | `0.776` |

B110 W&B links:

- Seed 206: <https://wandb.ai/tsilva/SuperMarioBros-NES/runs/u9qhp8y3>
- Seed 207: <https://wandb.ai/tsilva/SuperMarioBros-NES/runs/rmypl7nc>

## Interpretation

- B110 is still too early to judge at about `0.5M` steps. Neither seed has a
  Level1-2 clean clear yet, and both still have peak min-rate `0.00`.
- PPO health is not alarming: KL, clip fraction, explained variance, reward,
  and throughput are all plausible for early training.
- The no-space runner issue remains fixed; disk and heartbeats are healthy.

## Decision

No new recipe was queued in this iteration.

Reason:

- B110 has not yet reached the training region where B91 showed the decisive
  signal, so adding another arm now would not use new evidence.
- The queue already contains the direct follow-up stabilizers B100, B107, B108,
  and B109.

Next monitor condition:

- Re-scan B110 after about `1M` steps or after either seed emits positive
  Level1-2 clear-rate movement.
- If both B110 seeds cross strict `0.80`, run a completion audit and report the
  winning spec, W&B group, run names, peak values, peak steps, and config
  deltas.
