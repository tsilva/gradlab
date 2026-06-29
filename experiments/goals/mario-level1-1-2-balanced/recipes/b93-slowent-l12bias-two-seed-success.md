# B93 Slow-Entropy Level1-2-Biased Success Recipe

Status: success for `mario-level1-1-2-balanced`.

Final audit time: `2026-06-27T14:46:49Z`.

## Goal

Find a `SuperMarioBros-Nes-v0` Level1-1/Level1-2 mixed-policy PPO recipe where
both seeds in the same two-seed batch reach a peak
`train/info/level_complete/rate/min/last > 0.80` within the 5M-step cap.

## Winning Spec

- Spec:
  `experiments/goals/mario-level1-1-2-balanced/specs/b93-slowent-l12bias-l11l12-two-seed.yaml`
- W&B group: `b93-l11l12-slowent-l12bias-two-seed`
- Runtime target: `rtx4090` / beast-3 queue-backed fleet
- Runtime digest short id: `c672be38cd0f`
- Seeds: `200`, `201`

## Completion Evidence

Queue status marked both B93 train results as `succeeded`, and W&B marked both
runs as `finished`.

| Seed | Run | W&B id | Final W&B state | Latest step | Peak min-rate | Peak step | Peak L1-1 | Peak L1-2 |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `200` | `b93_l11l12_slowent_l12bias_s200_20260627T112704Z` | `tva7bwbh` | `finished` | `5,005,312` | `0.87` | `4,653,360` | `0.87` | `0.87` |
| `201` | `b93_l11l12_slowent_l12bias_s201_20260627T112704Z` | `vvmq6arm` | `finished` | `5,005,312` | `0.94` | `4,967,360` | `0.94` | `0.95` |

Both peak steps are below the 5M cap, and both peak min-rate values are strictly
greater than `0.80`.

The final/latest values were lower than the peaks, especially seed `201`, but
that is expected and not disqualifying for this goal because the contract ranks
by the training high-watermark rather than final checkpoint value.

## Recipe Delta

Parent near-miss:
`experiments/goals/mario-level1-1-2-balanced/specs/b86-b74current-l11l12-latest-five-seed.yaml`

B93 kept the B86 PPO/reward shape conservative and changed only the balancing
and exploration pressure:

- `state_probs`: from `[0.5, 0.5]` to `[0.4, 0.6]`, biasing attempts toward
  Level1-2.
- `ent_coef_final`: from `0.0003` to `0.001`.
- `ent_coef_schedule_timesteps`: from `2,000,000` to `4,000,000`.

Important unchanged settings:

- `learning_rate=0.00015` with effectively constant schedule.
- `target_kl=0.2`.
- `clip_range=0.15`.
- `reward_mode=score`.
- `terminal_reward=50`.
- `death_penalty=25`.
- No `completion_reward`.
- `task_conditioning=true`.
- `advantage_normalization=per-task`.

## Interpretation

B93 solved the same-policy balanced training high-watermark by preserving
Level1-1 competence while giving Level1-2 more sampling and keeping exploration
alive longer. The run family also confirms why this goal should not use the
final metric value as the deciding scalar: both seeds crossed cleanly and later
wobbled or collapsed from the peak.

## Playback

Use a checkpoint near each peak for visual inspection. Seed `201` peaked close
to the `step-5000000` checkpoint:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab-play \
  tsilva/SuperMarioBros-NES/b93_l11l12_slowent_l12bias_s201_20260627T112704Z-checkpoint:step-5000000 \
  --episodes 3 \
  --fps 30 \
  --scale 4
```

Seed `200` peaked between the `step-4600000` and `step-4700000` checkpoints.
Start with the later checkpoint:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab-play \
  tsilva/SuperMarioBros-NES/b93_l11l12_slowent_l12bias_s200_20260627T112704Z-checkpoint:step-4700000 \
  --episodes 3 \
  --fps 30 \
  --scale 4
```
