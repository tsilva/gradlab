# Workflow Ownership

Use the owning source when a rule changes. Skills provide operating guidance;
they do not create a second scientific contract or replace command validation.

| Concern | Authoritative source |
| --- | --- |
| Project-wide stakeholder requirements | Root [`SPECS.md`](../../SPECS.md) |
| Capability-specific stakeholder requirements | [`docs/specs/README.md`](../specs/README.md) and its indexed specifications |
| Domain vocabulary and architectural decisions | [`CONTEXT.md`](../../CONTEXT.md) and [`docs/adr/`](../adr/) |
| Compute policy and operator-local fleet/access facts | [`COMPUTE.md`](../../COMPUTE.md) and `~/.config/gradlab/instances.md` |
| Metric names and scientific meaning | [`METRICS.md`](../../METRICS.md) |
| Runtime versions and dependency resolution | [`pyproject.toml`](../../pyproject.toml) and `uv.lock` |
| Workflow selection, authorization boundaries, and evidence interpretation | The matching `.codex/skills/<name>/SKILL.md` |
| Detailed command procedures and unusual recovery cases | The selected skill's `references/` files |
| Autoresearch search contract | [`search-policy.md`](../../.codex/skills/gradlab_autoresearch/references/search-policy.md) |
| Model-preview editorial and provider policy | [`release-metadata.md`](../../.codex/skills/gradlab_upload-youtube-video/references/release-metadata.md) |
| Reservation, admission, identity, and integrity enforcement | Maintained controller/service code and its behavior tests |

Apply root specification review at the task boundary; skill composition does not
start another task or require rereading unchanged documents. Review stakeholder
intent again before finishing as required by the specification policy.

## Training conventions

- Read the owning goal's `_goal.yaml` before selecting recipes, caps, metrics, or Promotion criteria. Current Mario Level1-1 uses `experiments/goals/SuperMarioBros-Nes-v0/Level1-1/_goal.yaml`. Seed ranges belong to `gradlab.seeds`.
- Launchable recipes live in their active goal's `recipes/`; reusable defaults live in `experiments/recipes/_presets/`.
- Every Run needs a specific description. `gradlab experiment launch` derives one from the goal, recipe, and seed when omitted.
- Training tasks are profileless and locked to exact-source immutable runtime images. V1 assigns one single-GPU host per task; explicitly configured CPU-only execution follows its scoped contract.
- Logical Run IDs are `gradlab-<32 lowercase hex>`; Attempt IDs are `attempt-<16 lowercase hex>`. Retries retain the Run and create an Attempt. Numeric database or batch IDs do not replace them.
- Route W&B projects by canonical game family; retain provider identity and `environment_hash` in configuration. The W&B Run ID and group equal the immutable GradLab Run ID.
- The lease-holding supervisor is the sole selected-service writer. Modal workers receive no selected-service or control-private credentials. Checkpoints, evidence, replays, ROMs, and recovery bytes stay in their designated R2 buckets.
- Store generated Runs under `~/.config/gradlab/runs/`; use ignored `logs/` and `models/` for other generated outputs. Keep machine-specific access facts in the private inventory.

## Routing examples

| Request | Workflow |
| --- | --- |
| Launch or follow this recipe/Run | `gradlab_launch-experiment`, launch/monitor path |
| Fix or finish this failed workflow | `gradlab_launch-experiment`, repair mode within the request's bounds |
| Evaluate these selected published Checkpoints | `gradlab_launch-experiment`, manual-evaluation path and requested list/result completion gate |
| Tune this recipe's numeric PPO/A2C hyperparameters | `gradlab_autoresearch`, fixed training-only protocol |
| Optimize architecture, rollout size, rewards, or another backend | Explicit experiment plan covering the requested axes |
| Publish this unevaluated Checkpoint | `gradlab_upload-checkpoint`, Checkpoint Publication path |
| Publish this accepted Policy to HF/YouTube | `gradlab_upload-checkpoint`, accepted-release path |
| Update this existing YouTube description | `gradlab_upload-youtube-video`, existing-video path |
| Publish the next Python package release | `gradlab_build-release`, publication path |
| Build a release candidate locally | `gradlab_build-release`, local-candidate path |
| Certify or debug the orchestration lifecycle | `gradlab_certify-lifecycle` |
