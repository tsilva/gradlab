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

Portable experiment reports and supporting evidence live under `reports/`.
`gradlab validate` excludes that subtree from configuration-file checks and
counts; JSON files elsewhere in the experiment tree remain invalid configuration.

Current research state:

- `goals/`: active goal contracts, optional environment fragments, and W&B view declarations.
- `recipes/`: active checked-in training recipes and presets.
- `benchmarks/`: reproducible benchmark profiles and supporting documentation.
- `scripts/`: active experiment utilities used by benchmarks and tooling.

## Local training

```bash
gradlab train gradlab__bandit/ppo
gradlab env list
gradlab benchmark list
```

The bundled bandit recipe runs on CPU with a durable local metric journal.
For other recipes, `gradlab train <goal>/<recipe>` uses the checked-in goal and
training configuration. NES goals require a lawfully obtained ROM. Supply it
with `--rom-path` or register it with `gradlab rom sync`. Use
`gradlab rom status --json` to inspect registered ROMs. Keep ROMs and credentials
outside source control.

Visual PPO presets request CUDA and fail if it is unavailable. Classic control,
the discrete Gymnasium baseline, and the bundled bandit recipe select CPU.
Local training can establish Training Success. Acceptance requires a separate
evaluation under the goal's declared contract.

### Gymnasium recipes

The discrete Gymnasium goals use isolated vector lanes, masked resets, native
rewards, and RGB rendering. They are Training-Only goals. Use
`gymnasium:<environment-id>` for the qualified environment ID.

| Environment | PPO recipe |
| --- | --- |
| CartPole, MountainCar, Acrobot, LunarLander | `CartPole-v1/ppo`, `MountainCar-v0/ppo`, `Acrobot-v1/ppo`, `LunarLander-v3/ppo` |
| FrozenLake | `FrozenLake-v1/Default/ppo`, `FrozenLake-v1/Maze/ppo` |
| CliffWalking | `CliffWalking-v1/Default/ppo`, `CliffWalking-v1/Slippery/ppo` |
| Taxi, Blackjack | `Taxi-v3/ppo`, `Blackjack-v1/ppo` |

The FrozenLake recipes use deterministic movement on the standard 4×4 map
and the fixed maze, respectively. Blackjack retains categorical observations
using `MultiDiscrete([32, 11, 2])` for player sum, dealer card, and usable ace.

### Tracking and publication

Project recipes inherit [MLflow](tracking.yaml) with online delivery unless
overridden. Pass `--set tracking.backend=wandb` to select W&B for local or
queued training.
The bandit example selects `local_only` delivery. Use `--no-wandb` to keep
other local runs local.

- `gradlab publish-local <run-directory>` publishes a completed local run's
  checkpoints, journal, and public telemetry.
- `gradlab sync <run-directory>` syncs a completed local run to its frozen tracker.
- `gradlab sync-run <run-id>` syncs a queued local-only run's retained R2 journal
  to its frozen tracker.

See the [MLflow debugging guide](../ops/mlflow/README.md#debugging-training) for
checkpoints and execution diagnostics, and [compute operations](../COMPUTE.md)
for queued runs. For replay publishing credentials, see the
[YouTube operator guide](../ops/youtube/README.md).

## Shared configurations

FirstWall and TwoWalls inherit their common Breakout PPO policy, optimizer,
reward and budget from `recipes/_presets/breakout-ppo.yaml`; each keeps its own
checkpoint and occupancy settings. Mario mixed-state PPO inherits the common
Mario optimizer and checkpoint schedule, overriding its policy fusion and
advantage grouping. The Mario family owns its shared reward catalog.

The per-level Mario campaign recipes inherit either
[`mario-campaign-ppo`](recipes/_presets/mario-campaign-ppo.yaml) or
[`mario-campaign-ppo-sil`](recipes/_presets/mario-campaign-ppo-sil.yaml).
Each level declares its recipe ID and overrides only the fields it needs.
Presets can supply the description, reward shape, and seeds.

Mario reward shapes can opt in to `progress_x_signed_16bit: true` when the
provider's position wraps at the 16-bit boundary. The task decodes the position
before updating furthest progress and stall timing. This setting is part of the
reward shape's semantic hash. Existing shapes keep their original decoding
unless they opt in.

Classic-control PPO specializes the discrete Gymnasium CPU preset's entropy
coefficient. Health Gathering Supreme inherits the Health Gathering goal and
training preset, retaining its scenario identity and longer training budget.
Recipe identities and paths remain separate; composition records every shared
source in immutable provenance.

Checked-in PPO recipes use the native `gradlab.ppo` backend. Set
`train.backend.config.precision` to `fp32`, `amp-fp16`, or `amp-bf16`.
The `execution_profile` options are `sb3-parity`, `compiled-parity`,
`compiled-fused-parity`, and the backend default, `max-throughput`.
The Atari, Mario, classic-control, and discrete Gymnasium base presets select
`sb3-parity` and `fp32`. CPU and MPS use eager execution. SB3 backends remain
available for explicit comparisons.

To initialize native PPO from public tensor-only weights, set
`train.backend.config.initial_policy` with a Hugging Face `repo_id`, full
commit `revision` for `policy.pt`, `sha256`, `size_bytes`, and boolean
`reset_value_head`. GradLab validates the weights and starts a fresh optimizer
and transition counter. Recovery resumes the saved checkpoint.

Set `critic_only: true` to fit an independent critic while freezing the actor.
It requires `train.policy_model.share_features_extractor: false` and a positive
value-loss coefficient. Set `bootstrap_truncations: false` for zero continuation
at episode truncation. That choice is recorded in the critic value contract.

Saved-state curricula practise from recorded states when
`state_archive.curriculum.restore_entries: true`. See the
[archive guide](../docs/occupancy.md#enable-coverage-starts) and
[curriculum metrics](../METRICS.md) for archive generations and practice metrics.

## Screening recipes

`goals/Breakout-Atari2600-v0/TwoWalls/recipes/ppo-screening.yaml` uses the same
training settings, budget, and checkpoint cadence as `ppo-two-walls.yaml` while
disabling automatic Checkpoint evaluation and Checkpoint Monitoring. Both recipes
share one training preset; the existing evaluated recipe keeps its default.

Other screening recipes can import `recipes/_presets/screening.yaml` after their
training preset in `defaults`. The resolved mode and all source-file hashes remain
in the immutable recipe provenance. No new Goal or evaluation method is needed.

Inspect training metrics as proxies and select retained Checkpoints for explicit
evaluation through the existing Player workflow. An Evaluated Goal keeps its
frozen Acceptance criteria, evaluation seeds, and full-episode evidence checks;
screening itself cannot establish Acceptance or Promotion. A Training-Only Goal
remains ineligible for Acceptance. Monitoring remains a separate opt-in campaign.

## Mario route discovery

The [historical Level1-3 diagnostic](https://github.com/tsilva/gradlab/blob/7e643f4f820ecc0a82b6c7f90a787810c64c5ee8/experiments/goals/SuperMarioBros-Nes-v0/Level1-3/recipes/go-explore-route-diagnostic.yaml)
defined Go-Explore route search bounded to one million environment transitions,
with automatic Acceptance evaluation disabled. It remains available at that
source revision. The current Level1-3 directory keeps the
[qualified PPO recipe with SIL](goals/SuperMarioBros-Nes-v0/Level1-3/recipes/ppo.yaml).

Go-Explore checkpoint exports support portable route graphs and emulator snapshots
for replay and PPO diagnostics. Verify a candidate route from the original level
start before using it as a demonstration; search results do not establish the
learned PPO policy's 100-episode Acceptance.

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

Every checked-in `env-vizdoom-turbo` goal can opt into the same policy head without changing
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

## License

[MIT License](../LICENSE).
