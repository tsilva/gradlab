# 2026-06-27 B89-B108 Progress Snapshot

Goal metric: peak `train/info/level_complete/rate/min/last`, requiring both
seeds in the same two-seed batch to exceed strict `0.80` within the 5M training
cap. This snapshot uses full-row W&B history scans for changed active runs.

## Live Queue

After adding B108:

- Succeeded train jobs: `10`
- Running train jobs: `5`
- Pending train jobs: `34`
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
| `b89_l11l12_complete25_s203_20260627T110832Z` | running | `1,879,968` | `0.28` | `0.31` | `0.28` | `0.03` | `0.66` | `0.03` |
| `b90_l11l12_lowpress_l12bias_s204_20260627T112001Z` | running | `1,137,504` | `0.01` | `0.01` | `0.01` | `0.01` | `0.01` | `0.54` |
| `b90_l11l12_lowpress_l12bias_s205_20260627T112001Z` | running | `2,395,184` | `0.02` | `0.02` | `0.04` | `0.02` | `0.02` | `0.02` |
| `b91_l11l12_lowpress_complete25_s206_20260627T112013Z` | running | `2,287,872` | `0.38` | `0.38` | `0.70` | `0.27` | `0.27` | `0.70` |
| `b91_l11l12_lowpress_complete25_s207_20260627T112013Z` | running | `2,393,408` | `0.44` | `0.74` | `0.44` | `0.44` | `0.75` | `0.44` |

Interpretation:

- B89 has not recovered; both seeds now show Level1-1 strengthening while
  Level1-2 collapses.
- B90's hard Level1-2 sampling bias remains a bad tradeoff. It can lift Level1-2
  on one seed but destroys the balanced min metric through Level1-1.
- B91 is now the clear active leader. Both seeds are moving, and the bottlenecks
  are complementary: seed 206 has strong Level1-2 with Level1-1 lagging, while
  seed 207 has strong Level1-1 with Level1-2 lagging.

## Added Backfill

B108 covers the missing B91-derived slow-entropy interaction at neutral 50/50
state sampling.

- Spec: `specs/b108-lowpress-complete25-slowent-l11l12-two-seed.yaml`
- W&B group: `b108-l11l12-lowpress-complete25-slowent-two-seed`
- Seeds: `204,205`
- Delta from B91:
  - `ent_coef_final=0.001` instead of `0.0003`
  - `ent_coef_schedule_timesteps=4000000` instead of `2000000`
  - keep `state_probs=[0.5,0.5]`, `completion_reward=25`, and low update
    pressure
- Jobs:
  - `73`: `b108_l11l12_lowpress_complete25_slowent_s204_20260627T125455Z`
  - `74`: `b108_l11l12_lowpress_complete25_slowent_s205_20260627T125455Z`

B108 is legal under the goal constraints because it changes only PPO entropy
hyperparameters while preserving the same states, task conditioning,
termination contract, reward mode, and W&B high-watermark selection metric. The
spec validated with `rlab.job_queue.load_spec_document` and was enqueued
profileless for `rtx4090` with runtime image digest `c672be38cd0f`.

## Next Monitor

Keep watching B91 through the 5M cap. It is still below the strict `>0.80`
target, but at roughly 2.4M steps it is the first active two-seed recipe where
both seeds have meaningful balanced high-watermarks.
