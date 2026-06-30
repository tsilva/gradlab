# 2026-06-29 B263-B264 Seed Population Launch

Goal: `Level2-1`

Primary metric: peak `train/info/level_complete/rate/min/last`

Status: B263 seed 90 finished weak; B264 launched the missing same-recipe
population seeds 91-94.

## Why B264 Exists

B263 initially launched only seed 90 of the B257-soft Level2-1 transfer. That
was too narrow for the goal search because seed variance is material in these
Mario PPO runs and beast-3 has enough screening capacity for a same-recipe seed
population. B264 corrects that launch shape without mutating the already queued
B263 spec or its recorded SHA.

Both specs use the same reusable recipe,
`experiments/recipes/mario/b257-softupdate-no-bonus.yaml`, and differ only in
run metadata and seed set. They keep `SuperMarioBros-Nes-v0`, `state=Level2-1`,
native-vector preprocessing, `action_set=simple`,
`done_on_events=life_loss,level_change`, W&B artifacts, and the strict early
stop `train/info/level_complete/rate/min/last > 0.99`.

## B263 Seed 90 Result

| Field | Value |
| --- | --- |
| Job | `278` |
| Run | `b263_l21_b257soft_s90_20260629T181652Z` |
| W&B | <https://wandb.ai/tsilva/SuperMarioBros-NES/runs/3g3uu6i4> |
| Process state | `finished` |
| Final step | `5,005,312` |
| Final `train/info/level_complete/rate/min/last` | `0.01` |
| Final `train/info/level_complete/from/1-0/rate` | `0.01` |
| Final `train/info/level_complete/from/1-0/count` | `93` |
| Final `rollout/ep_rew_mean` | `2063.0393` |
| Final `train/approx_kl` | `0.006546817` |
| Final `train/clip_fraction` | `0.13256836` |
| Final `train/explained_variance` | `0.98590934` |

Interpretation: seed 90 is not a goal success. It did find sparse clean clears,
so the recipe is not completely dead on World 2-1, but the rolling clear window
never approached the strict 100/100 target.

## B264 Four-Seed Launch

Checked-in spec:
`experiments/goals/Level2-1/specs/b264-b263-b257soft-four-seed-l21-screen.yaml`

The spec validated with `rlab.job_queue.load_spec_document` before enqueueing.
The repo-local `rlab-train-image.json` file was absent, so the queue again
resolved the current latest successful digest-pinned train image:
`docker:ghcr.io/tsilva/rlab/rlab-train@sha256:150bca7eba9c0999f638b6439861fd2a9017daeb5fb8e4ec75502231b9fedd6c`.

Queued jobs:

| Job | Seed | Run | W&B |
| ---: | ---: | --- | --- |
| `279` | `91` | `b264_l21_b257soft_s91_20260629T184517Z` | <https://wandb.ai/tsilva/SuperMarioBros-NES/runs/giu1hy3g> |
| `280` | `92` | `b264_l21_b257soft_s92_20260629T184517Z` | <https://wandb.ai/tsilva/SuperMarioBros-NES/runs/y16dri12> |
| `281` | `93` | `b264_l21_b257soft_s93_20260629T184517Z` | <https://wandb.ai/tsilva/SuperMarioBros-NES/runs/7xu3qu7f> |
| `282` | `94` | `b264_l21_b257soft_s94_20260629T184517Z` | <https://wandb.ai/tsilva/SuperMarioBros-NES/runs/isktxd8f> |

## Live Snapshot

At monitor refresh `2026-06-29T18:46:36+00:00`, all B264 jobs were running on
beast-3 with fresh worker heartbeats:

| Job | Seed | Step | Rate | Clears | FPS | Worker |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `279` | `91` | `73,728` | `0` | `0` | `1,573` | `3-ad5afc57` |
| `280` | `92` | `73,728` | `0` | `0` | `1,574` | `2-21a874f4` |
| `281` | `93` | `73,728` | `0` | `0` | `1,553` | `1-ff5068b7` |
| `282` | `94` | `73,728` | `0` | `0` | `1,547` | `0-b05ee1f4` |

Fleet state:

- `rlab fleet status` reported `pending=0 running=4` for profile `any`,
  target `rtx4090`, digest `150bca7eba9c`.
- `rlab fleet ps` showed one beast-3 managed container,
  `rlab-beast-3-rtx4090-any-profile-150bca7eba9c`, with four active workers
  leased to jobs `279-282`.
- beast-2 remained unreachable by SSH, which does not affect this RTX4090
  screen.

## Next Decision

Monitor B264 by `train/info/level_complete/rate/min/last` first. If one seed
reaches strict `100/100`, freeze this B257-soft recipe and launch confirmation
seeds before out-of-process promotion eval. If all seeds remain near seed 90's
weak `0.01` high-water, reject this recipe family for Level2-1 and backfill
with a documented legal reward or hyperparameter variant.
