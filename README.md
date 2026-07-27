<div align="center">
  <img src="./logo.png" alt="rlab" width="256" />

  **Reinforcement-learning workbench for training game agents**
</div>

rlab turns checked-in goals and recipes into portable, reproducible training
runs. Research code sees one container and does not contain provider-specific
scheduling logic. dstack places that container on a local GPU, spot instance,
or explicitly authorized on-demand machine; Modal evaluates immutable
checkpoints independently.

## Architecture

| Concern | Authority |
| --- | --- |
| Provisioning, placement, logs, cancellation, interruption retries | dstack |
| Run, attempt, lease, promotion, and terminal state | private control R2 |
| Evaluation intents, results, evidence, and videos | private eval R2 |
| Training and evaluation metrics | one W&B run |
| Downloadable checkpoints and public run index | public model R2 |
| Temporary event buffering | SQLite WAL in the training container |

One supervisor inside the training container is the only W&B process. The
learner performs no network I/O: it emits local metric and checkpoint events
and responds to a cooperative stop signal. The supervisor uploads and verifies
checkpoints, dispatches Modal evaluations, observes accepted results, signals
the learner at a safe boundary, and drains all frozen work before the task can
succeed.

dstack exit status alone never establishes scientific success. The
authoritative terminal receipt must prove complete checkpoint/evaluation
inventories, promotion, W&B delivery, and drain.

## Install

```bash
git clone git@github.com:tsilva/rlab.git
cd rlab
./install.sh
rlab validate
```

The project uses `uv`, a committed `uv.lock`, and a seven-day package-age gate.
The exact Turbo runtime is `stable-retro-turbo==1.0.1.post36`,
`supermariobrosnes-turbo==0.5.0`, `breakout-turbo-env==0.5.1`, and
`vizdoom-turbo==1.3.0.post2`. Their explicit package-age exceptions are
recorded in `pyproject.toml` and `uv-tool.toml`.

The four Turbo providers must implement Turbo Vector API v1. rlab validates
their immutable capability and signal declarations, resolved action semantics,
observation ownership and buffer depth, read-only active state indices, and
per-lane rendering surface at construction time. It requires canonical
`state_catalog`/`state_indices` reset selection, consumes the declared action
contract directly, and uses `render_lane()`, `get_images()`, and lane-zero
`render()` without provider-specific probing or fallbacks. Providers that do
not advertise exactly v1 are rejected.

Register a local ROM without uploading it to source control:

```bash
rlab rom sync --game SuperMarioBros-Nes-v0
rlab rom status --json
```

Local dstack hosts use the hash-verified read-only ROM cache. Each Modal-backed
run stages its exact ROM bytes and manifest to eval-private R2. ROMs, R2
credentials, W&B credentials, and Modal credentials are never embedded in the
image.

## Train and play a recipe with uvx

`uvx` can run a built-in recipe without cloning rlab. Select this distribution
explicitly because the `rlab` project name on PyPI is owned by an unrelated
package. Downstream repositories should replace `<immutable-rlab-ref>` with a
release tag or full commit:

```bash
RLAB_SOURCE='git+https://github.com/tsilva/rlab.git@<immutable-rlab-ref>'

uvx --from "$RLAB_SOURCE" rlab train Breakout-Atari2600-v0/ppo
uvx --from "$RLAB_SOURCE" rlab play --recipe Breakout-Atari2600-v0/ppo

uvx --from "$RLAB_SOURCE" rlab train SuperMarioBros-Nes-v0/Level1-1/ppo
uvx --from "$RLAB_SOURCE" rlab play --recipe SuperMarioBros-Nes-v0/Level1-1/ppo

uvx --from "$RLAB_SOURCE" rlab train VizdoomBasic-v1/ppo
uvx --from "$RLAB_SOURCE" rlab play --recipe VizdoomBasic-v1/ppo
```

Training writes a unique run below `./runs`; recipe playback selects the newest
completed matching model. Local training disables W&B and checkpoint evaluation
by default, so it needs no orchestration credentials and cannot establish goal
acceptance or checkpoint promotion. Use repeatable `--set KEY=VALUE` overrides,
`--seed`, `--runs-dir`, or `--wandb` when needed. A recipe YAML in another
repository also works when it lives at
`experiments/goals/<goal>/recipes/<recipe>.yaml` beside its owning `_goal.yaml`.
Mario additionally requires a lawful local ROM registered once with
`rlab rom sync --game SuperMarioBros-Nes-v0`; local training verifies and binds
that cache without sending the ROM anywhere.

Reward transforms belong to the common task contract for every provider.
`task.reward.reward_scale` is a positive divisor, followed by
`task.reward.reward_clip`, which is `false`, `true` for `[-1, 1]`, or an
explicit `[low, high]` pair. rlab disables provider-native reward transforms so
the same ordered transform is used in training, evaluation, and playback.

## Launch and observe

Keep non-sensitive operator metadata and Keychain references in the private
user config, using the checked-in example as the schema:

```bash
mkdir -p ~/.config/rlab
install -m 600 ops/operator.example.toml ~/.config/rlab/operator.toml
rlab experiment operator-preflight --json
```

Replace the example placeholders and create the referenced generic-password
items in macOS Keychain Access. Secret values never belong in
`operator.toml` or the repository `.env`; explicit process environment values
remain supported for CI and non-macOS operators. Modal tokens remain in
Modal's active `~/.modal.toml` profile, which must use mode `0600`.

```bash
rlab experiment launch \
  --goal-file experiments/goals/SuperMarioBros-Nes-v0/Level1-1/_goal.yaml \
  --recipe-file experiments/goals/SuperMarioBros-Nes-v0/Level1-1/recipes/ppo.yaml \
  --seed 123 \
  --run-description "Mario Level1-1 PPO seed 123" \
  --compute local \
  --target b3 \
  --max-duration 48h \
  --json
```

`launch` runs the same read-only operator preflight before runtime readiness or
R2 mutation, then requires a clean, pushed source revision and waits for its
verified immutable training image and source-specific Modal deployment. It
returns the rlab run ID, attempt ID, dstack task, selected compute offer,
source/image digest, W&B URL, and public run-index URL.

W&B projects remain organized by canonical game family. New orchestrated runs
keep their immutable `rlab-…` run ID while using a readable
`<goal>__<recipe>__s<seed>__<short-run-id>` display name. Runs from a declared
campaign share a campaign group; otherwise runs sharing a goal, recipe, and
launch-time recipe variant share a cohort group.

Compute policy:

- `auto` uses an idle compatible local host first.
- `spot` requires finite `--max-price` and `--max-cost-usd`.
- without a cloud budget, `auto` waits for local capacity.
- `on-demand` also requires `--allow-on-demand`.
- every task requires a finite maximum duration.

Observe or control one logical run:

```bash
rlab experiment status --run <rlab-run-id> --json
rlab experiment follow --run <rlab-run-id>
rlab experiment wait --run <rlab-run-id> --until terminal --timeout 48h
rlab experiment logs --run <rlab-run-id> --tail 200
rlab experiment cancel --run <rlab-run-id>
rlab experiment retry --run <rlab-run-id>
```

Retry preserves the logical run ID and creates a new attempt ID. It requires a
terminal prior dstack attempt, an expired writer lease, and a 30-second
quiescence interval. A run whose learner already finished resumes in
drain-only mode and cannot retrain.

For a short B3 integration smoke, use the checked-in
`experiments/goals/SuperMarioBros-Nes-v0/Level1-1/recipes/dstack-smoke.yaml`
recipe. Repeatable `--set KEY=VALUE` ablations are composed, validated, and
included in the immutable portable recipe hash; use a checked-in leaf recipe
for durable or shared variants.

## Checkpoints and playback

Periodic and final checkpoints are immutable:

```text
runs/<run-id>/checkpoints/<step>-<sha256>/model.zip
runs/<run-id>/checkpoints/<step>-<sha256>/manifest.json
runs/<run-id>/index.json
```

The public index is mutable through ETag compare-and-swap and served with
`Cache-Control: no-store`; checkpoint objects are immutable and cacheable.
Start the web player without a source to browse repository-declared projects
and goals, then their W&B runs and public checkpoints. A W&B project or run URL
preselects that level; an exact source still opens directly:

```bash
rlab play
rlab play "https://wandb.ai/<entity>/<project>"
rlab play "https://wandb.ai/<entity>/<project>/runs/<rlab-run-id>"
rlab play --run <rlab-run-id>
rlab play --model <local-checkpoint>
rlab play hf://<owner>/<repository>
```

The player uses shareable hierarchical routes: `/` lists projects,
`/projects/<project-id>` lists goals,
`/projects/<project-id>/goals/<goal-id>` lists runs, and the nested
`/runs/<run-id>/checkpoints/<checkpoint-id>` route identifies a checkpoint.
Checkpoint rows show goal-required acceptance results from W&B when available.
Browser Back and Forward navigation follow the same resource hierarchy.

The playback dashboard is a GridStack workspace. Policy, reward, action, signal,
and reward-component views are instances of one configurable telemetry panel:
use **Panels → Add** to combine stats, compatible line series, histograms,
policy distributions, and dynamic metric explorers. Panels can be edited,
duplicated, hidden, resized, rearranged, or moved to a synchronized window.
The reward summary, value estimate, step reward, and episode return each have
their own built-in panel so history graphs never force the summary panel to
scroll. Workspace v4 intentionally starts clean instead of migrating older
saved layouts.

The checked-in goal catalog supplies projects and goals. W&B supplies run
metadata and available goal-required acceptance results only after a goal is
selected. Playback downloads model bytes from the public checkpoint store;
videos, episode evidence, ROMs, and recovery journals remain in R2.

## Evaluation and early stop

Every ready periodic checkpoint and the natural final checkpoint is evaluated
against the immutable goal-owned episode manifest. Modal validates the
checkpoint, goal, recipe, environment, protocol, and episode-manifest hashes.
Acceptance fails fast on the first valid failed episode and requires all 100
episodes to pass.

The supervisor polls results every two seconds. An accepted result requests
learner stop within ten seconds; the learner stops cooperatively at a safe
on-policy boundary and saves a final checkpoint. The ready set is then frozen,
every member reaches a terminal eval state, and the lowest-step accepted
checkpoint is promoted exactly once.

## Goals, recipes, metrics, and reports

- Active goals: `experiments/goals/`
- Goal-local launchable recipes: each goal’s `recipes/`
- Reusable presets: `experiments/recipes/_presets/`
- Metric contract: `METRICS.md`
- Hardware and dstack operations: `INSTANCES.md`
- Control-plane units and templates: `ops/dstack/`

Useful commands:

```bash
rlab validate
rlab env list
rlab env inspect supermariobrosnes-turbo:SuperMarioBros-Nes-v0
rlab env preflight \
  --goal-file experiments/goals/SuperMarioBros-Nes-v0/Level1-1/_goal.yaml \
  --recipe-file experiments/goals/SuperMarioBros-Nes-v0/Level1-1/recipes/ppo.yaml
rlab leaders runs --goal SuperMarioBros-Nes-v0/Level1-1 --min-seeds 3
rlab leaders checkpoints --goal SuperMarioBros-Nes-v0/Level1-1 --limit 1 --json
rlab reports plan --goal SuperMarioBros-Nes-v0/Level1-1
rlab reports sync --goal SuperMarioBros-Nes-v0/Level1-1
rlab reports verify --goal SuperMarioBros-Nes-v0/Level1-1
rlab benchmark list
```

## Datasets and model release

`rlab dataset` records and verifies Gymrec v3 gameplay data with provider-native
actions, rewards, boundaries, seeds, environment contracts, and approved policy
provenance:

```bash
rlab dataset record mario-level1-1 \
  --env-id SuperMarioBros-Nes-v0 \
  --provider supermariobrosnes-turbo \
  --agent human
rlab dataset verify mario-level1-1
rlab dataset play mario-level1-1 --episode 1
rlab dataset upload mario-level1-1 <owner/repository>
```

External SB3 checkpoints are Python-executable content. rlab stages and hashes
the complete model closure and requires approval before deserialization unless
the exact source matches `RLAB_MODEL_SOURCE_ALLOWLIST`.

Published model releases use Hugging Face model cards and include a
representative `replay.mp4` when the policy has visual behavior. Generated
artifacts belong under ignored `runs/`, `logs/`, and `models/` directories,
never source control.
