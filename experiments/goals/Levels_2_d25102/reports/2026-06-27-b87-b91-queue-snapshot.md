# 2026-06-27 B87-B91 Queue Snapshot

Goal: find a two-seed Level1-1/Level1-2 PPO recipe where both seeds in the same
batch reach peak `train/info/level_complete/rate/min/last > 0.80` within the 5M
training cap.

## Incumbent

- Spec: `b86-b74current-l11l12-latest-five-seed`
- Best run: `b86_l11l12_b74current_s195_20260627T091726Z`
- Full-history scan peak: `0.800` at step `4,070,256`
- Decision: near miss only; threshold is strict `> 0.80`, and no same two-seed
  batch solved the goal.

## Running Wave

Live queue snapshot after B90/B91 enqueue:

- Running: jobs 31-35
  - `b87_l11l12_lowpress_s198_20260627T110807Z`
  - `b87_l11l12_lowpress_s199_20260627T110807Z`
  - `b88_l11l12_l12bias_s200_20260627T110819Z`
  - `b88_l11l12_l12bias_s201_20260627T110819Z`
  - `b89_l11l12_complete25_s202_20260627T110832Z`
- Pending: jobs 36-40
  - `b89_l11l12_complete25_s203_20260627T110832Z`
  - `b90_l11l12_lowpress_l12bias_s204_20260627T112001Z`
  - `b90_l11l12_lowpress_l12bias_s205_20260627T112001Z`
  - `b91_l11l12_lowpress_complete25_s206_20260627T112013Z`
  - `b91_l11l12_lowpress_complete25_s207_20260627T112013Z`

Bounded W&B snapshot for active runs showed sampled peak min-rate `0` at about
0.88M-0.93M steps, so the current wave is still too early to judge.

## Next Recipes

- B90 combines B87 lower update pressure with B88 40/60 Level1-2 sampling bias.
- B91 combines B87 lower update pressure with B89 `completion_reward=25`.

Both specs were validated with `rlab.job_queue.load_spec_document` and enqueued
with the same immutable train image digest currently serving the beast-3 runner:
`c672be38cd0f`.

Fleet plan after enqueue kept the existing beast-3 deployment: one RTX4090
container, five workers, no reconcile action required. The beast-2 SSH timeout
warning is unrelated to this goal path.
