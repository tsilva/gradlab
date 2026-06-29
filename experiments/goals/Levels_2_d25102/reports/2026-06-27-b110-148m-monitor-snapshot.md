# 2026-06-27 B110 1.48M Monitor Snapshot

Goal metric: peak `train/info/level_complete/rate/min/last`, requiring both
seeds in the same two-seed batch to exceed strict `0.80` within the 5M training
cap. This snapshot uses full-row W&B history scans and judges by high-watermark.

Scan time: `2026-06-27T13:56:24Z`.

## Live Queue And Fleet

beast-3 is healthy.

- Train jobs: `6` failed, `10` succeeded, `5` running, `34` pending.
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

No active same-batch candidate has solved the goal yet, but B110 is now the best
active signal because both seeds have nonzero bottleneck rates and both-level
clears.

| Run | Step | Peak min-rate | Peak step | Latest min-rate | Latest L1-1 rate | Latest L1-2 rate | L1-1 clears | L1-2 clears | FPS | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `b110_l11l12_gentlerpress_complete25_l12soft_s206_20260627T133104Z` | `1,479,376` | `0.04` | `1,406,640` | `0.01` | `0.01` | `0.03` | `10` | `18` | `1,221` | `1429.885` |
| `b110_l11l12_gentlerpress_complete25_l12soft_s207_20260627T133104Z` | `1,458,176` | `0.03` | `1,454,000` | `0.03` | `0.03` | `0.03` | `9` | `6` | `1,203` | `1362.630` |
| `b92_l11l12_slowent_s199_20260627T112706Z` | `1,467,888` | `0.00` | `79,632` | `0.00` | `0.10` | `0.00` | `90` | `0` | `1,197` | `1164.399` |
| `b93_l11l12_slowent_l12bias_s200_20260627T112704Z` | `1,455,824` | `0.00` | `54,288` | `0.00` | `0.12` | `0.00` | `29` | `0` | `1,186` | `1511.340` |
| `b93_l11l12_slowent_l12bias_s201_20260627T112704Z` | `1,457,232` | `0.02` | `1,184,368` | `0.01` | `0.01` | `0.13` | `10` | `60` | `1,187` | `1618.725` |

B110 W&B links:

- Seed 206: <https://wandb.ai/tsilva/SuperMarioBros-NES/runs/u9qhp8y3>
- Seed 207: <https://wandb.ai/tsilva/SuperMarioBros-NES/runs/rmypl7nc>

## Interpretation

- B110 is still far below the strict `>0.80` success threshold, so there is no
  completion audit yet.
- B110 is now meaningfully alive: both seeds have clears on both Level1-1 and
  Level1-2, and both have a positive bottleneck high-watermark.
- The current bottleneck is still Level1-1 retention under the soft Level1-2
  sampling bias. Seed 206 peaked at `0.04` and then dropped to `0.01`; seed 207
  is balanced but only at `0.03`.
- B92 and B93 are weaker comparison arms. They mostly specialize, with B92 and
  B93 seed 200 clearing Level1-1 only, and B93 seed 201 showing Level1-2
  movement but weak Level1-1.

## Decision

No additional recipe was queued in this iteration.

Reason:

- B110 is improving before 1.5M and still has most of the 5M cap available.
- B111 is already queued as the symmetric Level1-1-tilted counter-arm for the
  observed B110 bottleneck.
- Adding another arm now would duplicate the already-queued stabilizer queue
  without incorporating later B110 evidence.

Next monitor condition:

- Continue full-history scans of B110 while it runs; do not early-stop on weak
  current values because the goal selects by peak and late recoveries matter.
- If both B110 seeds cross strict `0.80`, run the completion audit immediately.
- When B111 starts, compare whether the Level1-1 tilt improves the bottleneck
  without losing Level1-2 clears.
