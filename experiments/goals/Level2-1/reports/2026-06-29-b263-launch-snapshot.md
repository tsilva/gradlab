# 2026-06-29 B263 Level2-1 Launch Snapshot

Goal: `Level2-1`

Primary metric: peak `train/info/level_complete/rate/min/last`

Status: first Level2-1 screen launched and running.

## Hypothesis

B263 transfers the B257 soft-update no-completion-bonus recipe to World 2-1.
This preserves the B55/B250 score/progress reward semantics while lowering PPO
update pressure (`learning_rate=1.2e-4 -> 8e-5`, `clip_range=0.12`,
`target_kl=0.10`). This is a legal first screen because it changes only the
start state plus PPO hyperparameters/reward settings already captured in the
reusable recipe; it does not change the ROM, action set, observation pipeline,
termination semantics, or evaluation protocol.

## Spec

- Spec: `experiments/goals/Level2-1/specs/b263-b257soft-l21-screen.yaml`
- Stage: `screen`
- Seed: `90`
- Target: `rtx4090` / `beast-3`
- W&B group: `b263-l21-b257soft-screen`
- Run: `b263_l21_b257soft_s90_20260629T181652Z`
- W&B: <https://wandb.ai/tsilva/SuperMarioBros-NES/runs/3g3uu6i4>
- Job: `278`
- Runtime image: `docker:ghcr.io/tsilva/rlab/rlab-train@sha256:150bca7eba9c0999f638b6439861fd2a9017daeb5fb8e4ec75502231b9fedd6c`

The repo-local `rlab-train-image.json` file was absent, so the job was enqueued
without `--runtime-image-ref-file`; the queue resolved the latest successful
digest-pinned train image, matching the current repo workflow documented by
recent goal reports.

Validation used `rlab.job_queue.load_spec_document` before enqueueing. The
materialized config has `SuperMarioBros-Nes-v0`, `state=Level2-1`,
`done_on_events=life_loss,level_change`, and the strict early stop
`train/info/level_complete/rate/min/last > 0.99`.

## Launch State

`rlab jobs status --goal Level2-1` showed one active training job:

| Job | Status | Target | Run |
| ---: | --- | --- | --- |
| `278` | `running` | `rtx4090` | `b263_l21_b257soft_s90_20260629T181652Z` |

Fleet state:

- beast-3 container: `rlab-beast-3-rtx4090-any-profile-150bca7eba9c`
- Worker lease: `rlab-beast-3-rtx4090-any-profile-150bca7eba9c-1-ff5068b7`
- Host state: reachable, busy
- Monitor utilization sample: GPU `74%`, CPU `26%`, memory `4.8/60.9 GB`
- beast-2 was unreachable by SSH, but it is not needed for this RTX4090 screen.

## First Metrics Snapshot

At monitor refresh `2026-06-29T18:18:14+00:00`:

| Metric | Value |
| --- | ---: |
| `time/total_timesteps` | `163,840` |
| `time/fps` | `3,302` |
| `throughput/rollout_fps` | `4,290` |
| `train/info/level_complete/from/1-0/count` | `0` |
| `train/info/level_complete/from/1-0/rate` | `0` |
| `train/info/level_complete/rate/min/last` | `0` |
| `rollout/ep_rew_mean` | `400` |
| `train/approx_kl` | `0.0053554773` |
| `train/clip_fraction` | `0.158` |
| `train/explained_variance` | `0.745` |

## Next Decision

Monitor B263 by the goal metric first. If it reaches
`train/info/level_complete/rate/min/last > 0.99`, freeze the recipe and launch
confirmation seeds before promotion eval. If it plateaus with zero Level2-1
clean clears after a mature partial run, backfill with a documented legal
reward/hyperparameter variant rather than changing environment or evaluation
semantics.
