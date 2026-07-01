# Mario Level1-2 100/100 Goal

This folder is the durable capsule for the active `Level1-2`
research objective.

Primary optimization metric: peak
`train/info/level_complete/rate/min/last`. For this single-level goal it should
match the Level1-2 source rate once the rolling source window is full; external
eval remains the promotion check.

- `_goal.yaml`: current goal contract, metric, seed protocol, cap, runtime, and
  promotion policy.
- `specs/`: checked-in queue payloads for candidate runs.
- Shared archived recipe fragments referenced by specs live in `../../history/recipes/`.
- `reports/`: checked-in summaries or analysis reports for this goal.
- `decisions/`: checked-in decision records that should outlive chat history.

Generated checkpoints, W&B files, videos, and raw local logs should stay out of
source control. Prefer the repo-level ignored `runs/`, `logs/`, and `models/`
trees unless a goal-local ignored scratch directory is explicitly useful.

```bash
UV_CACHE_DIR=.uv-cache uv run rlab train \
  --spec-file experiments/goals/super-mario-bros-nes-v0/Level1-2/specs/b85-b46-post21-five-seed-l12-confirm.yaml \
  --runtime-image-ref-file rlab-train-image.json
```
