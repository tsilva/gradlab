# 2026-06-27 B89-B107 Progress Snapshot

Goal metric: peak `train/info/level_complete/rate/min/last`, requiring both
seeds in the same two-seed batch to exceed strict `0.80` within the 5M training
cap. This snapshot uses full-row W&B history scans for changed active runs.

## Live Queue

After adding B107:

- Succeeded train jobs: `10`
- Running train jobs: `5`
- Pending train jobs: `32`
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
| `b89_l11l12_complete25_s203_20260627T110832Z` | running | `1,879,968` | `0.28` | `0.31` | `0.28` | `0.08` | `0.29` | `0.08` |
| `b90_l11l12_lowpress_l12bias_s204_20260627T112001Z` | running | `1,137,504` | `0.01` | `0.01` | `0.01` | `0.00` | `0.00` | `0.15` |
| `b90_l11l12_lowpress_l12bias_s205_20260627T112001Z` | running | `1,082,864` | `0.01` | `0.01` | `0.02` | `0.00` | `0.00` | `0.05` |
| `b91_l11l12_lowpress_complete25_s206_20260627T112013Z` | running | `1,977,952` | `0.22` | `0.22` | `0.56` | `0.18` | `0.18` | `0.54` |
| `b91_l11l12_lowpress_complete25_s207_20260627T112013Z` | running | `2,004,928` | `0.28` | `0.31` | `0.28` | `0.28` | `0.31` | `0.28` |

Interpretation:

- B89 confirms `completion_reward=25` can produce mid-run balanced competence,
  but seed 202 later specialized toward Level1-1 while Level1-2 collapsed.
- B90 shows that a hard Level1-2 tilt with low update pressure is too damaging
  to Level1-1 in the current recipe.
- B91 is the best active signal. Seed 206 has Level1-2 around `0.54` to `0.56`
  but remains Level1-1 bottlenecked, while seed 207 is more balanced and still
  rising around `0.28`.

## Added Backfill

B107 covers the missing low-pressure plus completion-reward plus soft Level1-1
sampling interaction.

- Spec: `specs/b107-lowpress-complete25-l11soft-l11l12-two-seed.json`
- W&B group: `b107-l11l12-lowpress-complete25-l11soft-two-seed`
- Seeds: `200,201`
- Delta from B91: `state_probs=[0.55,0.45]` instead of `0.50/0.50`
- Jobs:
  - `71`: `b107_l11l12_lowpress_complete25_l11soft_s200_20260627T124958Z`
  - `72`: `b107_l11l12_lowpress_complete25_l11soft_s201_20260627T124958Z`

B107 is legal under the goal constraints because it changes only state sampling
weights while preserving the same states, task conditioning, termination
contract, reward mode, and W&B high-watermark selection metric. The spec
validated with `rlab.job_queue.load_spec_document` and was enqueued profileless
for `rtx4090` with runtime image digest `c672be38cd0f`.
