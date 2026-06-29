# 2026-06-27 B116 B93 Repro Launch Snapshot

Goal metric: peak `train/info/level_complete/rate/min/last`, judged by
full-history high-watermark within the 5M cap, not final value.

Launch time: `2026-06-27T15:01:01Z`.

## Purpose

Reproduce the winning B93 recipe across five fresh seeds beyond the solved
`200/201` pair.

## Spec

- Spec:
  `experiments/goals/mario-level1-1-2-balanced/specs/b116-b93-repro-five-seed.yaml`
- Parent recipe:
  `experiments/goals/mario-level1-1-2-balanced/specs/b93-slowent-l12bias-l11l12-two-seed.yaml`
- Stage: `confirm`
- W&B group: `b116-b93-repro-five-seed`
- Seeds: `208`, `209`, `210`, `211`, `212`
- Target: `rtx4090` / beast-3
- Runtime image:
  `docker:ghcr.io/tsilva/rlab/rlab-train@sha256:c672be38cd0fb7b5505d4d7b902ac10316ec979538c784838531098b4c1bf0e5`

## Queued Jobs

| Job | Run |
| ---: | --- |
| `89` | `b116_l11l12_b93repro_s208_20260627T150101Z` |
| `90` | `b116_l11l12_b93repro_s209_20260627T150101Z` |
| `91` | `b116_l11l12_b93repro_s210_20260627T150101Z` |
| `92` | `b116_l11l12_b93repro_s211_20260627T150101Z` |
| `93` | `b116_l11l12_b93repro_s212_20260627T150101Z` |

## Config

B116 keeps B93 exactly for the recipe levers under test:

- `state_probs=[0.4, 0.6]`
- `ent_coef=0.01`
- `ent_coef_final=0.001`
- `ent_coef_schedule_timesteps=4000000`
- `learning_rate=0.00015`
- `target_kl=0.2`
- `clip_range=0.15`
- `reward_mode=score`
- `terminal_reward=50`
- `death_penalty=25`
- no `completion_reward`
- `task_conditioning=true`
- `advantage_normalization=per-task`

## Queue And Fleet

After enqueue, queue status showed:

- Train jobs: `6` failed, `15` succeeded, `5` running, `42` pending.
- Active workers: B115 seeds `200/201`, B111 seeds `206/207`, and B112 seed
  `206`.
- B116 jobs `89-93` were the next pending jobs shown by `rlab-queue status`.
- Fleet plan required no action; beast-3 already had the matching
  `c672be38cd0f` container with five workers.

## Monitoring

For each B116 run, scan full W&B history for:

- peak `train/info/level_complete/rate/min/last`
- peak step
- peak `train/info/level_complete/from/0-0/rate`
- peak `train/info/level_complete/from/0-1/rate`
- latest value only as a stability diagnostic

The repro question is how many of the five fresh seeds exceed strict `0.80`.
