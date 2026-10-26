# Project Rules

## Product Specifications

At the start of substantive repository work, use `$specs-author` to read the entire root `SPECS.md` once per ongoing task. Reuse unchanged contents across follow-ups and composed skills. Reread when the file or repository association changes, or new stakeholder intent requires a fresh comparison. Before finishing, always check for new or changed stakeholder intent; reread if the file or intent changed. Purely conversational follow-ups do not require another full read.

- Treat `SPECS.md` as the persistent source of stakeholder requirements that cannot be inferred reliably from code or remembered conversations.
- Apply the scope test to proposed and existing requirements: root `SPECS.md` contains only project-wide intent; scoped intent belongs in its nearest authoritative specification and must not be broadened to fit the root.
- If the task, repository, or user request contradicts, omits, or ambiguously interprets the specification, tell the user. Continue safe exploration and work that does not depend on resolving the issue, but never silently choose an interpretation.
- Never edit `SPECS.md` from inference. Propose the exact change, explain why it reflects stakeholder intent, and edit the file only after the user explicitly approves that exact change.
- Keep `SPECS.md` complete, concise, and compacted. It must contain stakeholder intent rather than implementation, architecture, operations, or transient project detail.

## Scoped Specifications

Before changing publication, playback, queued execution, or Go-Explore behavior, read the matching authoritative scoped specification indexed by `docs/specs/README.md`. Treat each scoped requirement as user product intent for that capability, preserve its original presumption of immutability until the user explicitly evolves it, and keep it consistent with root `SPECS.md` and `CONTEXT.md`.

## GPU Instances

Before choosing hardware, launching training, changing concurrency, or recommending compute targets, read `COMPUTE.md`. For local compute, also read the operator-local `~/.config/gradlab/instances.md`; it is the source of truth for that operator's fleet names, access commands, capacity, enrollment, cleanup, and host-specific gotchas. If the local inventory is absent, do not invent a target or capacity.

When changing dstack host behavior, preserve the root-owned runtime-image cleanup contract: prune only unused gradlab-managed images while preserving active containers and every immutable digest demanded by pending or active dstack tasks.

## Stable Retro

- Use PyPI `env-stableretro-turbo`; import it as `env_stableretro_turbo`.
- Native-vector code should use `env_stableretro_turbo.RetroVecEnv`, whose constructor follows the original `RetroEnv` positional signature plus vector-only keyword arguments.
- Runtime version source of truth: the exact pins in `pyproject.toml` and the resolved versions in `uv.lock`. Use `uv sync --frozen`; make overrides explicit in recipes, compute policy, run descriptions, and selected-service tags.
- Every Turbo provider must declare and pass the strict Turbo Vector API v2 contract before gradlab consumes it. Do not add provider probing or legacy fallbacks.
- Native-vector observations are channel-first `(n_envs, channels, height, width)`. Reject other layouts instead of transposing or guessing.
- Keep portable runtime history and benchmark conclusions in experiment reports. Keep machine-specific access and benchmark facts in the operator-local inventory.

## Training Runs

- For orchestration certification, lifecycle validation, or debugging, use `$gradlab_certify-lifecycle` in `.codex/skills/gradlab_certify-lifecycle`.
- For dstack-backed launch, monitoring, or selected-checkpoint evaluation, use `$gradlab_launch-experiment` in `.codex/skills/gradlab_launch-experiment`; its recovery matrix governs observe and authorized repair modes. Delegate active-run potential-bug diagnosis to the read-only `training_run_investigator`.
- For numeric SB3 PPO/A2C tuning within the fixed training-only search contract, use `$gradlab_autoresearch` in `.codex/skills/gradlab_autoresearch`. Requests involving architecture, rollout settings, rewards, categorical choices, or other backends need an explicit experiment plan rather than silently narrowing their scope.
- Before operating training or evaluation, read the shared conventions in [`docs/agents/workflows.md`](docs/agents/workflows.md). Scientific requirements remain authoritative in `SPECS.md` and the matching scoped specification; commands enforce their executable checks.

## Metrics

- `METRICS.md` is the source of truth for registered metric names and semantics across selected services.
- When adding, removing, renaming, or changing the meaning of a logged metric, update `METRICS.md` in the same change.
- When touching metric logging, dashboards, reports, eval summaries, or answering metric semantics questions, audit the relevant emitted metric names/templates against `METRICS.md` and patch any missing or stale entries before finishing.
- When the user asks a metric question and the answer is not already clear from `METRICS.md`, improve `METRICS.md` with that clarification before finishing.

## Publication and Package Releases

- For Checkpoint Publication or accepted Policy Releases, use `$gradlab_upload-checkpoint` in `.codex/skills/gradlab_upload-checkpoint`; use `$model-card-author` for generated-card quality.
- For standalone YouTube maintenance, use `$gradlab_upload-youtube-video` in `.codex/skills/gradlab_upload-youtube-video`. Its `references/release-metadata.md` owns preview policy; checkpoint-release previews use the durable publisher.
- For GradLab Python releases, use `$gradlab_build-release` in `.codex/skills/gradlab_build-release`, which composes the shared `$release-workflow`. The project launcher prepares metadata and pushes the release tag; GitHub Actions owns source validation, all release builds, artifact audits, and trusted publication. Normal release or validation requests do not build locally.

## Dependencies

Use `uv` for dependency resolution and keep `uv.lock` committed. Preserve Python supply-chain hardening in `pyproject.toml`.
The intentional exceptions to the seven-day `exclude-newer` window are `env-breakoutatari2600-turbo-native`, `env-gradoom-turbo-torch`, `env-stableretro-turbo`, `env-supermariobrosnes-turbo-emu`, and `env-vizdoom-turbo`, because this project tracks current forward native Breakout, Torch-native Doom, Stable Retro, Mario, and ViZDoom runtimes while keeping the rest of the dependency graph age-gated. Keep the per-package cutoffs in `[tool.uv.exclude-newer-package]`, `uv-tool.toml`, and the user-level uv config in sync so `uv tool install . --editable` remains installable without extra flags.

## Instruction Ownership

[`docs/agents/workflows.md`](docs/agents/workflows.md) maps authoritative requirements, domain vocabulary, operating procedures, and executable checks. Keep `AGENTS.md` focused on repository-wide rules and workflow triggers. Link to owning policies rather than copying their detailed procedures here.

## Agent skills

### Issue tracker

Issues are tracked in this repository's GitHub Issues. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles use their default label names. See `docs/agents/triage-labels.md`.

### Domain docs

This is a single-context repository with a root `CONTEXT.md` and ADRs under `docs/adr/`. See `docs/agents/domain.md`.
