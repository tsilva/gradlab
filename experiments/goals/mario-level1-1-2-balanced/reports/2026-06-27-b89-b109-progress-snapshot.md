# 2026-06-27 B89-B109 Progress Snapshot

Goal metric: peak `train/info/level_complete/rate/min/last`, requiring both
seeds in the same two-seed batch to exceed strict `0.80` within the 5M training
cap. This snapshot uses full-row W&B history scans for changed active runs.

## Live Queue

After adding B109:

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
| `b89_l11l12_complete25_s202_20260627T110832Z` | finished | `2,512,208` | `0.41` | `0.41` | `0.48` | `0.02` | `0.84` | `0.02` |
| `b89_l11l12_complete25_s203_20260627T110832Z` | running | `1,879,968` | `0.28` | `0.31` | `0.28` | `0.06` | `0.86` | `0.06` |
| `b90_l11l12_lowpress_l12bias_s204_20260627T112001Z` | running | `2,614,944` | `0.04` | `0.04` | `0.04` | `0.03` | `0.05` | `0.03` |
| `b90_l11l12_lowpress_l12bias_s205_20260627T112001Z` | running | `2,632,064` | `0.08` | `0.08` | `0.08` | `0.05` | `0.05` | `0.14` |
| `b91_l11l12_lowpress_complete25_s206_20260627T112013Z` | running | `2,753,920` | `0.58` | `0.58` | `0.65` | `0.56` | `0.56` | `0.70` |
| `b91_l11l12_lowpress_complete25_s207_20260627T112013Z` | running | `2,514,832` | `0.46` | `0.73` | `0.46` | `0.24` | `0.75` | `0.24` |

Interpretation:

- B91 is still the only live recipe with both seeds showing meaningful balanced
  competence. Seed 206 improved from `0.38` to `0.58` since the previous
  snapshot and remains near peak. Seed 207 peaked at `0.46` but has partially
  lost Level1-2.
- B89 and B90 continue to show specialization or weak balanced learning and are
  no longer the leading branch.
- The immediate failure mode to probe is weaker-level drift after useful
  two-level competence appears, not absence of all learning.

## Added Backfill

B109 covers the missing B91-derived gentler-update-pressure interaction.

- Spec: `specs/b109-gentlerpress-complete25-l11l12-two-seed.json`
- W&B group: `b109-l11l12-gentlerpress-complete25-two-seed`
- Seeds: `206,207`
- Delta from B91:
  - `target_kl=0.12` instead of `0.16`
  - `clip_range=0.12` instead of `0.15`
  - `learning_rate_final=0.000075` instead of `0.0001`
  - keep `state_probs=[0.5,0.5]`, `completion_reward=25`, and the same
    entropy schedule
- Jobs:
  - `75`: `b109_l11l12_gentlerpress_complete25_s206_20260627T130011Z`
  - `76`: `b109_l11l12_gentlerpress_complete25_s207_20260627T130011Z`

B109 is legal under the goal constraints because it changes only PPO update
hyperparameters while preserving the same states, task conditioning,
termination contract, reward mode, and W&B high-watermark selection metric. The
spec validated with `rlab.job_queue.load_spec_document` and was enqueued
profileless for `rtx4090` with runtime image digest `c672be38cd0f`.

## Next Monitor

Keep watching B91 through the 5M cap. It is not solved yet, but seed 206 is now
above the halfway mark to the strict threshold on both levels, and seed 207's
earlier Level1-2 peak gives a concrete stabilizer target for B107-B109.
