# Runner Profiles

Runner profiles are the durable source for long-lived SkyPilot queue workers.
They describe the remote runtime envelope: hardware shape, package pin, ROM
mounts, smoke checks, and the exact `train_jobs.profile_id` a worker may claim.

Use profiles when the queue already owns the experiment payload. Use
`experiments/launches/` manifests only for older direct SkyPilot batches that
embed concrete training runs in the rendered YAML.

Render a profile to ignored scratch YAML:

```bash
UV_CACHE_DIR=.uv-cache uv run stable-retro-ppo-skypilot render-runner \
  experiments/runner_profiles/mario_ppo_post19_task_conditioned_rtx4090.example.json \
  --output sky_train_runner_4090.yaml
```

Preflight a profile before launch:

```bash
UV_CACHE_DIR=.uv-cache uv run stable-retro-ppo-skypilot preflight-runner \
  experiments/runner_profiles/mario_ppo_post19_task_conditioned_rtx4090.example.json
```

Launch a queue runner:

```bash
UV_CACHE_DIR=.uv-cache uv run stable-retro-ppo-skypilot launch-runner \
  experiments/runner_profiles/mario_ppo_post19_task_conditioned_rtx4090.example.json \
  --output sky_train_runner_4090.yaml \
  --execute \
  --detach-run
```

Keep profile IDs coarse. Create a new profile when the runtime contract changes:
package pin, observation space or policy family, ROM/state mounts, hardware
shape, or queue-client compatibility. Seeds, hyperparameters, W&B tags, stop
criteria, and run descriptions belong in queued jobs.
