# Training schedules

The `gradlab.ppo`, `sb3.ppo`, and `sb3.a2c` backends support optional linear gamma
schedules in `train.backend.config`:

- `gamma` is the initial discount factor.
- `gamma_final` defaults to `null`, which disables scheduling.
- `gamma_schedule_timesteps` defaults to `0`, which uses the declared training
  timestep budget when a final gamma is supplied. A positive value sets an explicit
  duration in aggregate environment transitions, not native frames or episodes.

The discount interpolates from the initial value to the final value starting at
training step zero, then stays at the endpoint. Both endpoints must be finite and
within `[0, 1]`. A nonzero duration requires `gamma_final`.

Gamma is selected using the absolute training step at rollout start and held fixed
through collection, timeout bootstrapping, GAE, and optimization. An episode may
span multiple rollouts. Resuming uses the restored timestep count; it does not
restart the schedule. Learning-rate and entropy schedules remain independent, and
GAE lambda is unchanged.

`train/gamma` records the discount used for each rollout. Model checkpoints retain
the active gamma; the immutable recipe records the complete schedule. For a
changing schedule, critic metadata does not claim one stationary training discount.
Playback can inspect reward contributions using the loaded gamma, but fixed-discount
critic calibration is unavailable, including after the schedule reaches its endpoint.

No checked-in recipe enables gamma scheduling by default.

## Checkpoint schedules

Checkpoint cadence is a Run choice: goals provide frequency defaults, and recipes or launch
overrides may set `train.checkpoint_freq` and `train.checkpoint_steps`. Both use
absolute aggregate policy-environment transitions. A zero frequency disables
periodic saves; an explicit increasing list supplies phase milestones. Periodic
and explicit thresholds can be combined.

```yaml
train:
  checkpoint_freq: 0
  checkpoint_steps: [500000, 1000000, 2000000, 4000000, 8000000]
  checkpoint_candidates:
    metric: train/return/mean
    direction: maximize
    min_delta: 0.05
    delta_mode: relative
    start_after_steps: 500000
    min_interval_steps: 2000000
    max_checkpoints: 3
```

PPO/A2C saves occur after completed learner updates. Thresholds round upward to
that boundary; multiple crossed thresholds produce one snapshot. Resume skips
already crossed milestones and continues the absolute periodic cadence. The
terminal path always saves the final or early-stop Policy, and the periodic path
never duplicates the declared cap. Metric logging and early-stop decisions keep
their independent cadence. GradLab PPO's exact `checkpoint_update_steps` branch
points remain supported and must align with completed updates.

The optional learned-policy candidate shortlist saves at most the declared number
of meaningful training-proxy improvements per Attempt, in addition to scheduled
snapshots. The first eligible metric sample establishes the reference; the cooldown
limits save frequency. Candidates are ordinary immutable Checkpoints, retain the
configured evaluation backend, and do not prove optimality or restore an earlier
Policy. See `METRICS.md` for proxy semantics. Smoke/demo recipes disable the shortlist.

Go-Explore/JERK consume the same schedule after complete search batches. Go-Explore
also permits at most `train.backend.config.improvement_checkpoints` improved-completion
snapshots per Attempt (default three, zero disables them), spaced by at least the
checkpoint interval or one vector batch when the periodic interval is disabled.
The final route graph still includes the latest retained search result; previously
published evidence is never replaced or deleted.

Monitoring calibration binds frequency, milestones, candidate budget and trainer
configuration. Count estimates reserve candidate and interrupted artifacts. A
changed schedule invalidates calibration; metric-triggered candidates require an
explicit uncalibrated monitoring override until their potentially adjacent saves
can be supported by calibration. Monitoring remains disabled in bundled recipes.
