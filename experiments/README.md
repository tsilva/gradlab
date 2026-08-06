# Experiments

This directory holds active goal contracts, training recipes, benchmark
profiles, W&B view declarations, and experiment utilities.
Keep broad repo rules in the top-level runbooks:

- `../AGENTS.md` for repo rules and stable-retro runtime cautions.
- `../COMPUTE.md` for portable compute policy.
- `~/.config/gradlab/instances.md` for operator-local hardware inventory.

Use `goals/<env-id>/` for durable goal-family contracts and optional provider-specific
environment fragments named `_env-<provider>.yaml`. Goal-family report declarations
live beside those contracts as `_reports.yaml`; the shared project-workspace declaration
lives at `goals/_workspaces.yaml`. Active training recipes live under `recipes/`, while
benchmark profiles live under `benchmarks/`. Generated local run
outputs belong under `~/.config/gradlab/runs/`; other generated logs and models
belong under ignored `logs/` and `models/` paths.

Current research state:

- `goals/`: active goal contracts, optional environment fragments, and W&B view declarations.
- `recipes/`: active checked-in training recipes and presets.
- `benchmarks/`: reproducible benchmark profiles and supporting documentation.
- `scripts/`: active experiment utilities used by benchmarks and tooling.

## Provider-owned policy signal histories

Structured policy inputs use the existing `task.model_inputs.context` contract. A
transition-updated field may opt into the provider's image-frame-aligned history:

```yaml
task:
  signals:
    health: health
    selected_weapon: selected_weapon
  model_inputs:
    schema_version: 1
    context:
      health:
        signal: health
        update: transition
        history: provider_frame_stack
        encoding: {kind: continuous, scale: 0.01, offset: 0.0, low: 0.0, high: 2.0, clip: true}
      selected_weapon:
        signal: selected_weapon
        update: transition
        history: provider_frame_stack
        encoding: {kind: categorical, values: [1, 2, 3, 4, 5, 6]}

policy_model:
  schema_version: 2
  encoder: {kind: nature_cnn, features_dim: 512}
  info_history_encoder: {hidden_sizes: [128], activation: relu}
  fusion: {hidden_sizes: [256], activation: tanh}
  normalize_images: true
  orthogonal_init: true
```

The history depth always inherits the resolved image `frame_stack`. Entries are
provider-owned policy transitions ordered oldest to newest, not raw emulator tics;
GradLab neither appends nor reconstructs them when `frame_skip` is greater than one.
Continuous normalization and categorical vocabularies are shared across temporal
positions. At the policy, all configured history features for one transition are
concatenated before the next transition, flattened in temporal-major order, encoded
by `info_history_encoder`, and fused after the channel-stacked image encoder.

The provider still emits each base key as a current-transition task signal for reward,
events, termination, metrics, and diagnostics. A context field using
`provider_frame_stack` consumes only the provider's `<key>_frame_stack`; declare a
separate current-only context field only when the policy intentionally needs both.
Masked reset replaces history only in reset lanes. Terminal history remains paired
with the terminal observation used for value bootstrapping before those lanes reset.
Construction fails if the provider capability or declared reset/step schema does not
match the resolved image stack.

## Shared ViZDoom action head

Every checked-in `vizdoom-turbo` goal can opt into the same policy head without changing
its default recipe:

```yaml
action_profile: vizdoom-shared-multidiscrete-v1
```

The profile resolves that goal's exact provider action table, maps it into the common
`MultiDiscrete([3, 3, 10, 2, 2, 23])` coordinates, and restricts the joint categorical
distribution to those ordered legal tuples. Sampling, log probability, entropy, playback,
and evaluation therefore use only actions already supported by that goal. The profile is
part of effective-goal, environment, checkpoint, and publication provenance. Omitting
`action_profile` preserves the existing discrete action space and checkpoint behavior.
