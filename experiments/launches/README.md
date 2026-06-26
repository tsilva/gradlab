# Launches

Root-level `sky_*.yaml` files are ignored scratch launch files. Promote a direct
batch launch shape here only when it is worth reviewing and reusing. For
long-lived queue workers, promote the runtime shape to
`experiments/runner_profiles/` instead.

Prefer names that describe intent instead of timestamps, and keep credentials in
SkyPilot secrets or environment variables rather than in YAML.

Run names should use `<batch>_<scope>_<arm>_s<seed>_<utc>`, for example
`b58_l11_lowkldecay_s108_20260623T142700Z`.

- `batch`: short campaign handle, usually `bNN`.
- `scope`: target or training/eval scope, such as `l11`, `l12`, or `l11l12`.
- `arm`: concise recipe or hypothesis label, not every hyperparameter.
- `s<seed>`: training seed.
- `utc`: UTC launch timestamp in `YYYYMMDDTHHMMSSZ` form.

Keep target/scope separate from the arm so the same recipe can be compared
across targets. Only fold scope into the arm when the recipe has no meaning
outside that target.

## Manifest-Driven Compute Launches

Use `rlab-compute` to launch a JSON experiment matrix on the configured target.
It dispatches SkyPilot targets to `rlab-skypilot` and Modal targets to
`modal run src/rlab/modal_app.py::launch_manifest`. Local `beast-2` and
`beast-3` targets are Docker fleet hosts; do not use these direct SkyPilot
launch manifests for them.

This path embeds concrete training runs in the rendered SkyPilot YAML. Prefer
runner profiles when jobs already live in the campaign queue and SkyPilot only
needs to host workers.

The example manifest is a template. Replace `game`, optional `state`, and
`rom_source` before expecting preflight or launch to succeed.
Default reusable manifests should target `stable-retro-turbo==1.0.0.post21`;
older runtime versions belong only in explicitly historical repro or comparison
manifests.

Render a task:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab-skypilot render \
  experiments/launches/rlab_rtx4090.example.json \
  --target runpod-rtx4090 \
  --output sky_rlab_generated_4090.yaml
```

Check env, descriptions, RTX4090 child count, `env_threads`, runtime pin, and
ROM source before launching on a SkyPilot-backed target:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab-compute preflight \
  experiments/launches/rlab_rtx4090.example.json \
  --target runpod-rtx4090
```

Print the provider launch command without running it:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab-compute launch \
  experiments/launches/rlab_rtx4090.example.json \
  --target runpod-rtx4090 \
  --output sky_rlab_generated_4090.yaml
```

Switch the same manifest to Modal without editing JSON:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab-compute launch \
  experiments/launches/rlab_rtx4090.example.json \
  --target modal-t4
```

Actually launch only after the rendered YAML and preflight output look right:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab-compute launch \
  experiments/launches/rlab_rtx4090.example.json \
  --target runpod-rtx4090 \
  --output sky_rlab_generated_4090.yaml \
  --execute \
  --sparse \
  --log-output logs/rlab_example_4090.sky.log
```

Add `--down-on-complete` when the cluster should be cleaned up automatically
after the launch command exits.

Summarize child logs after completion:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab-skypilot collect \
  logs/rlab_example_4090
```

Write a machine-readable report from a full SkyPilot launch log:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab-skypilot report \
  logs/rlab_example_4090.sky.log \
  --output reports/rlab_example_4090.report.json
```

Reproduce a W&B run config without hardcoding the ROM target:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab-skypilot repro-from-wandb \
  tsilva/SuperMarioBros-NES/lexxixz3 \
  --rom-source roms/your-game.rom \
  --target runpod-rtx4090 \
  --manifest-output experiments/launches/repro_lexxixz3.json \
  --output sky_repro_lexxixz3_4090.yaml
```

Use the local Mac SkyPilot API server as the default control plane:

```bash
UV_CACHE_DIR=.uv-cache uv run sky api start --host 127.0.0.1
UV_CACHE_DIR=.uv-cache uv run sky api info
```

Machine-readable hardware defaults may live in `experiments/instances.json` when
launch manifests need them. Keep `INSTANCES.md` as the human source of truth,
and update both files together when a benchmark-backed hardware fact changes.
