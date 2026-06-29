# 2026-06-27 B110 Early Health Snapshot

Goal metric: peak `train/info/level_complete/rate/min/last`, requiring both
seeds in the same two-seed batch to exceed strict `0.80` within the 5M training
cap. This snapshot is an early health check, not a recipe decision.

Scan time: `2026-06-27T13:40:00Z`.

## Live Queue And Fleet

beast-3 is healthy after the earlier disk cleanup and runner restart.

- Train jobs: `6` failed, `10` succeeded, `5` running, `32` pending.
- Eval jobs: none.
- Fleet plan: no action; keep the existing beast-3 RTX4090 managed runner.
- Runner digest: `c672be38cd0f`.
- Disk check: `/` has about `42G` free and is at `80%` use.

Active beast-3 workers:

- `job=77`: `b110_l11l12_gentlerpress_complete25_l12soft_s206_20260627T133104Z`
- `job=78`: `b110_l11l12_gentlerpress_complete25_l12soft_s207_20260627T133104Z`
- `job=44`: `b92_l11l12_slowent_s199_20260627T112706Z`
- `job=41`: `b93_l11l12_slowent_l12bias_s200_20260627T112704Z`
- `job=42`: `b93_l11l12_slowent_l12bias_s201_20260627T112704Z`

## W&B Early History

The B110 runs are live in W&B:

- Seed 206: <https://wandb.ai/tsilva/SuperMarioBros-NES/runs/u9qhp8y3>
- Seed 207: <https://wandb.ai/tsilva/SuperMarioBros-NES/runs/rmypl7nc>

| Run | Step | Peak min-rate | Latest L1-1 rate | Latest L1-2 rate | L1-1 clears | L1-2 clears | FPS | Latest reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `b110_l11l12_gentlerpress_complete25_l12soft_s206_20260627T133104Z` | `275,392` | `0.00` | `0.00` | `0.00` | `1` | `0` | `1,240` | `570.881` |
| `b110_l11l12_gentlerpress_complete25_l12soft_s207_20260627T133104Z` | `260,448` | `0.00` | `0.00` | `0.00` | `2` | `0` | `1,172` | `487.442` |
| `b92_l11l12_slowent_s199_20260627T112706Z` | `269,472` | `0.00` | `0.00` | `0.00` | `2` | `0` | `1,133` | `552.270` |
| `b93_l11l12_slowent_l12bias_s200_20260627T112704Z` | `262,144` | `0.00` | `0.00` | `0.00` | `1` | `0` | `1,109` | `546.174` |
| `b93_l11l12_slowent_l12bias_s201_20260627T112704Z` | `262,144` | `0.00` | `0.00` | `0.00` | `1` | `0` | `1,108` | `618.309` |

## Interpretation

- B110 is too young to judge. It has only reached about `260k-275k` steps, with
  almost no clean clears and no positive rolling min-rate yet.
- The operational path is healthy: W&B is logging, queue heartbeats are fresh,
  throughput is normal for the current five-worker beast-3 shape, and disk is no
  longer blocking Docker mounts.
- No new recipe was added. Adding more queue depth now would not incorporate
  evidence from B110; the existing queue already contains the direct B91
  stabilizer arms B100, B107, B108, and B109.

## Next Monitor Condition

- Re-scan B110 after it reaches at least about `1M` steps or after either seed
  emits a meaningful positive `train/info/level_complete/rate/min/last`.
- If both B110 seeds cross strict `0.80`, run the completion audit against the
  goal contract and report the winning spec, W&B group, run names, peak values,
  peak steps, and config deltas.
- If B110 remains flat while B92/B93 also remain flat, keep the queue moving
  toward the already queued completion-reward stabilizers rather than adding a
  speculative new arm.
