<div align="center">
  <img src="./logo.png" alt="gradlab" width="256" />

  **Reinforcement-learning workbench for training game agents**
</div>

gradlab turns checked-in goals and recipes into portable, reproducible training
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
the learner at a safe boundary, closes automatic evaluation admission, and
drains every evaluation submitted before acceptance before the task can succeed.

dstack exit status alone never establishes scientific success. The
authoritative terminal receipt must prove the complete checkpoint inventory,
the terminal inventory of automatically submitted evaluations, promotion,
W&B delivery, and drain.

## Install

```bash
git clone git@github.com:tsilva/gradlab.git
cd gradlab
./install.sh
gradlab validate
```

The project uses `uv`, a committed `uv.lock`, and a seven-day package-age gate.
The exact Turbo runtime is `stable-retro-turbo==1.0.1.post37`,
`supermariobrosnes-turbo==0.6.2`, `breakout-turbo-env==0.5.2`, and
`vizdoom-turbo==1.3.0.post19`. Their explicit package-age exceptions are
recorded in `pyproject.toml` and `uv-tool.toml`.

The four Turbo providers must implement Turbo Vector API v1. gradlab validates
their immutable capability and signal declarations, resolved action semantics,
observation ownership and buffer depth, read-only active state indices, and
per-lane rendering surface at construction time. It requires canonical
`state_catalog`/`state_indices` reset selection, consumes the declared action
contract directly, and uses `render_lane()`, `get_images()`, and lane-zero
`render()` without provider-specific probing or fallbacks. Providers that do
not advertise exactly v1 are rejected.

Register a local ROM without uploading it to source control:

```bash
gradlab rom sync --game SuperMarioBros-Nes-v0
gradlab rom status --json
```

Local dstack hosts use the hash-verified read-only ROM cache. Each Modal-backed
run stages its exact ROM bytes and manifest to eval-private R2. ROMs, R2
credentials, W&B credentials, and Modal credentials are never embedded in the
image.

## Train and play a recipe with uvx

`uvx` can run a built-in recipe without cloning GradLab. Pin the distribution
version in downstream repositories when exact reproducibility matters:

```bash
GRADLAB_VERSION='0.1.1'

uvx --from "gradlab==$GRADLAB_VERSION" gradlab train Breakout-Atari2600-v0/ppo
uvx --from "gradlab==$GRADLAB_VERSION" gradlab play --recipe Breakout-Atari2600-v0/ppo

uvx "gradlab@$GRADLAB_VERSION" train SuperMarioBros-Nes-v0/Level1-1/turbo-demo \
  --rom-path /absolute/path/to/SuperMarioBros.nes

uvx --from "gradlab==$GRADLAB_VERSION" gradlab train VizdoomBasic-v1/ppo
uvx --from "gradlab==$GRADLAB_VERSION" gradlab play --recipe VizdoomBasic-v1/ppo
```

From a source checkout, the YAML-defined Breakout Go-Explore search runs locally with:

```bash
uv run gradlab train Breakout-Atari2600-v0/go-explore-20m
```

Training writes a unique run below `./runs`; recipe playback selects the newest
completed matching model. Local training disables W&B and checkpoint evaluation
by default, so it needs no orchestration credentials and cannot establish goal
acceptance or checkpoint promotion. Use repeatable `--set KEY=VALUE` overrides,
`--seed`, `--runs-dir`, or `--wandb` when needed. A recipe YAML in another
repository also works when it lives at
`experiments/goals/<goal>/recipes/<recipe>.yaml` beside its owning `_goal.yaml`.
On an interactive terminal, local training opens a Textual dashboard with
in-place progress, rate and ETA, bounded events and histories, and the common
plus algorithm-specific telemetry declared by the backend. Press `q`, `Ctrl+C`,
or `Ctrl+Q` to request a graceful stop at the backend's safe boundary; the
dashboard remains open until the learner finishes writing its terminal model
and result. Pass `--no-tui`, redirect output, or use a dumb terminal for stable
plain progress logs. The dashboard is presentation-only and does not enter
recipe hashes, training metrics, or run receipts.
For Mario, `--rom-path` verifies and uses a lawful raw `.nes` file in place without
copying it or modifying GradLab's ROM registry or cache. The completed run prints
a version-pinned `uvx ... play` command using the same ROM. The demo targets
macOS arm64 and Linux x86_64; its 98,304 training steps take about two minutes on
the calibrated M1 Pro, while timing varies by hardware. A first invocation may
also need time to download GradLab, Torch, and the environment wheels. The
existing `gradlab rom sync` workflow remains available for queued staging and
repeated registered-cache use.

Reward transforms belong to the common task contract for every provider.
`task.reward.reward_scale` is a positive divisor, followed by
`task.reward.reward_clip`, which is `false`, `true` for `[-1, 1]`, or an
explicit `[low, high]` pair. gradlab disables provider-native reward transforms so
the same ordered transform is used in training, evaluation, and playback.

## Launch and observe

Keep non-sensitive operator metadata and Keychain references in the private
user config, using the checked-in example as the schema:

```bash
mkdir -p ~/.config/gradlab
install -m 600 ops/operator.example.toml ~/.config/gradlab/operator.toml
gradlab experiment operator-preflight --json
```

Replace the example placeholders and create the referenced generic-password
items in macOS Keychain Access. Secret values never belong in
`operator.toml` or the repository `.env`; explicit process environment values
remain supported for CI and non-macOS operators. Modal tokens remain in
Modal's active `~/.modal.toml` profile, which must use mode `0600`.

```bash
gradlab experiment launch \
  --goal-file experiments/goals/SuperMarioBros-Nes-v0/Level1-1/_goal.yaml \
  --recipe-file experiments/goals/SuperMarioBros-Nes-v0/Level1-1/recipes/ppo.yaml \
  --seed 123 \
  --run-description "Mario Level1-1 PPO seed 123" \
  --compute local \
  --max-duration 48h \
  --json
```

`launch` runs the same read-only operator preflight before runtime readiness or
R2 mutation, then requires a clean, pushed source revision and waits for its
verified immutable training image and source-specific Modal deployment. It
returns the gradlab run ID, attempt ID, dstack task, selected compute offer,
source/image digest, W&B URL, and public run-index URL.

W&B projects remain organized by canonical game family. New orchestrated runs
keep their immutable `gradlab-…` run ID while using a readable
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
gradlab experiment status --run <gradlab-run-id> --json
gradlab experiment follow --run <gradlab-run-id>
gradlab experiment wait --run <gradlab-run-id> --until terminal --timeout 48h
gradlab experiment logs --run <gradlab-run-id> --tail 200
gradlab experiment cancel --run <gradlab-run-id>
gradlab experiment retry --run <gradlab-run-id>
```

Retry preserves the logical run ID and creates a new attempt ID. It requires a
terminal prior dstack attempt, an expired writer lease, and a 30-second
quiescence interval. A run with a published final checkpoint resumes in
drain-only mode and cannot retrain. If acceptance or a training plateau was
already recorded but the final checkpoint was not published, recovery resumes
the latest checkpoint only long enough to request a safe finalization boundary.

For a short local-fleet integration smoke, use the checked-in
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
Start the web player without a source to browse repository-declared
environments and goals, automatically indexed goal-contract variants, then
their control-plane runs and public checkpoints. An exact W&B run URL opens that
run directly:

```bash
gradlab play
gradlab play "https://wandb.ai/<entity>/<project>/runs/<gradlab-run-id>"
gradlab play --run <gradlab-run-id>
gradlab play --model <local-checkpoint>
gradlab play hf://<owner>/<repository>
```

The player uses shareable hierarchical routes: `/` lists environments,
`/environments/<environment-id>` lists goals,
`/environments/<environment-id>/goals/<goal-id>` lists goal variants, and the
nested `/variants/<goal-variant-id>/runs/<run-id>/checkpoints/<checkpoint-id>`
route identifies a checkpoint inside the selected run. Checkpoint rows show
goal-required acceptance results from W&B when available. Browser Back and
Forward navigation follow the same resource hierarchy.

The playback dashboard is a GridStack workspace. Policy, reward, action, signal,
and reward-component views are instances of one configurable telemetry panel:
use **Panels → Add** to combine stats, compatible line series, histograms,
policy distributions, and dynamic metric explorers. Panels can be edited,
duplicated, hidden, resized, rearranged, or moved to a synchronized window.
The reward summary, value estimate, step reward, and episode return each have
their own built-in panel so history graphs never force the summary panel to
scroll. The player reads and writes only the current workspace v5 schema;
noncurrent saved layouts are ignored.

Policy diagnostics are capability-driven. PPO and A2C expose their actor
distribution and state-value critic; action programs expose their fixed program
cursor and fallback behavior. Go-Explore checkpoints retain safe search and
archive summary provenance while playing the resulting action program.
Unsupported diagnostics remain visibly unavailable instead of being replaced
with synthetic probabilities, entropy, log-probabilities, or values.

The required `experiments/goals/_catalog.yaml` namespace index supplies
environments and goals. Launch and supervisor paths register each versioned
goal-variant descriptor in a rebuildable private-control-R2 per-goal index, so
the variant panel needs one bounded object read rather than a run or artifact
scan; `gradlab experiment catalog-rebuild` replaces those indexes from immutable
current run manifests. The private index supplies run metadata; W&B supplies
available goal-required acceptance results only after a checkpoint-bearing run
is selected. Playback downloads model bytes from the public checkpoint store;
videos, episode evidence, ROMs, and recovery journals remain in R2.

## Evaluation and early stop

Ready periodic checkpoints are evaluated against the immutable goal-owned
episode manifest until an accepted result closes automatic evaluation
admission. Modal validates the checkpoint, goal, recipe, environment, protocol,
and episode-manifest hashes. Acceptance fails fast on the first valid failed
episode and requires all 100 episodes to pass.

The supervisor polls results every two seconds. An accepted result requests
learner stop within ten seconds; the learner stops cooperatively at a safe
on-policy boundary and saves a final checkpoint. Evaluations submitted before
acceptance finish without retries, while later periodic and final checkpoints
remain published but unevaluated for future explicit user action. The
lowest-step accepted checkpoint is then promoted exactly once.

Before submitting any newly ready evaluation, the supervisor reconciles
durable results for existing attempts. This prevents a result that finished
concurrently with a training plateau or final checkpoint from being missed.
An intent that never crossed the submission boundary is recorded as deferred
after acceptance; it is not sent to Modal.

For an evaluated goal, a training plateau is provisional until evaluation
drain settles. Acceptance wins even if the learner reached the plateau first.
Without acceptance, the plateau becomes a scientific failure only when every
published checkpoint has a valid rejected evaluation. Failed, expired, or
otherwise incomplete evaluation evidence produces a resumable attempt instead.
Evaluation attempts retain their own expiry windows; the 300-second terminal
delivery deadline starts only after evaluations have settled.

## Goals, recipes, metrics, and reports

- Active goals: `experiments/goals/`
- Goal-local launchable recipes: each goal’s `recipes/`
- Reusable presets: `experiments/recipes/_presets/`
- Metric contract: `METRICS.md`
- Portable compute and dstack policy: `COMPUTE.md`
- Operator-local fleet inventory: `~/.config/gradlab/instances.md`
- Control-plane units and templates: `ops/dstack/`

PPO and A2C recipes declare their actor–critic architecture under
`train.policy_model`. Each role owns an independent hidden stack; an empty list
connects encoded features directly to SB3's final output projection:

```yaml
train:
  policy_model:
    schema_version: 1
    topology:
      kind: shared_encoder
      encoder: {kind: nature_cnn, features_dim: 512}
    fusion: post_encoder_concat
    context_encoders: {}
    routes: {}
    heads:
      action: {hidden_sizes: [], activation: tanh}
      state_value: {hidden_sizes: [256], activation: tanh}
    normalize_images: true
    orthogonal_init: true
```

Context is concatenated after image encoding and only into the roles named in
`routes`. The supported head activations are `tanh` and `relu`; every hidden
width must be positive. Backend fields `policy_net_arch` and `value_net_arch`
are unsupported.

Useful commands:

```bash
gradlab validate
gradlab env list
gradlab env inspect supermariobrosnes-turbo:SuperMarioBros-Nes-v0
gradlab env preflight \
  --goal-file experiments/goals/SuperMarioBros-Nes-v0/Level1-1/_goal.yaml \
  --recipe-file experiments/goals/SuperMarioBros-Nes-v0/Level1-1/recipes/ppo.yaml
gradlab leaders runs --goal SuperMarioBros-Nes-v0/Level1-1 --min-seeds 3
gradlab leaders checkpoints --goal SuperMarioBros-Nes-v0/Level1-1 --limit 1 --json
gradlab reports plan --goal SuperMarioBros-Nes-v0/Level1-1
gradlab reports sync --goal SuperMarioBros-Nes-v0/Level1-1
gradlab reports verify --goal SuperMarioBros-Nes-v0/Level1-1
gradlab benchmark list
```

## Datasets and model release

`gradlab dataset` records and verifies Gymrec v3 gameplay data with provider-native
actions, rewards, boundaries, seeds, environment contracts, and approved policy
provenance:

```bash
gradlab dataset record mario-level1-1 \
  --env-id SuperMarioBros-Nes-v0 \
  --provider supermariobrosnes-turbo \
  --agent human
gradlab dataset verify mario-level1-1
gradlab dataset play mario-level1-1 --episode 1
gradlab dataset upload mario-level1-1 <owner/repository>
```

External SB3 checkpoints are Python-executable content. gradlab stages, hashes,
and re-verifies the complete model closure before deserialization. Playback
performs these integrity checks automatically without a model pre-approval step.

Published model releases use Hugging Face model cards and include a
representative `replay.mp4` when the policy has visual behavior. Local run
artifacts and downloaded model caches default to `~/.config/gradlab/runs/`;
other generated artifacts belong under ignored `logs/` and `models/`
directories, never source control.
