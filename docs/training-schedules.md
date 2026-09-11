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
