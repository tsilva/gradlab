# Project Rules

## Product Specifications

When specifications exist, use `$private-skills_specs-author` for substantive work and
specification changes. Follow its shared read/reuse cadence, provisional-scenario
guidance, proportional detail, decision-based approval, and closing intent check.
The maintained procedure is `/Users/tsilva/.codex/skills/private-skills/private-skills_specs-author/SKILL.md`.

## Scoped Specifications

Before changing a capability with an authoritative scoped specification, consult the matching file indexed by `docs/specs/README.md` under the shared task-level read/reuse cadence. Preserve existing commitments and keep scoped requirements consistent with root `SPECS.md` and `CONTEXT.md`; apply the same `$private-skills_specs-author` decision-based approval policy to root and scoped changes.

## Agent Service Access

For agent-operated training on the reviewed local coordinator, use
`python3 -I -S -B scripts/training_vault.py check` and its
`experiment operator-preflight|launch|status|follow|logs|cancel|retry|resume-submit`
operations. Native Infisical 0.45.6 uses separate
`local-gradlab-training-operator` and `local-gradlab-training-worker` bundles.
The operator profile pins source/runtime, instance, CA and coordinator bindings;
changed pins require trusted review and native fake-first tests.
The reviewed sources are the Gradlab project's `prod`, `/` references for
the six `GRADLAB_{CONTROL,EVAL,MODELS}_R2_{ACCESS_KEY_ID,SECRET_ACCESS_KEY}`
values and `DSTACK_B3_TOKEN` in private `operator.toml`. Trusted administration
copies those exact values into dedicated Vault services. Ordinary tools never
read them. The three R2 services target their fixed loopback signing gateways
and original buckets; the coordinator service targets only its pinned API.
The proxy bootstrap source is Gradlab `dev`, `/agents/training-proxy`.
See [training Vault access](docs/agents/training-vault.md) for setup, checks,
recovery and limitations. Do not use raw injection or the older launchers as
fallbacks. Modal, W&B, YouTube and durable Publication access are separate.

For agent-operated standalone Hugging Face maintenance, use
`python3 -I -S -B scripts/huggingface_vault.py check`, or its `hf` mode for the
reviewed Hub file commands. It uses native Infisical 0.45.6, the dedicated
`local-gradlab-huggingface` bundle, a public `HF_TOKEN` placeholder, a fresh
scoped session and an OS sandbox. It validates protected source/runtime/SDK
pins and rejects raw-token, endpoint, cache, scheduler and login overrides.
Changed pins require trusted review and fake-first testing. The publisher's
credential resolver fails closed when Vault mode is selected; it cannot fall
back to a cached token. Do not substitute the older unsandboxed global launcher.
The Gradlab Infisical `dev` folder `/agents/huggingface` remains the source for
trusted administrative rotation. See [agent service access](docs/agents/workflows.md#agent-service-access)
for checks and transfer limitations.
This standalone route does not replace durable Checkpoint/Release Publication
or the separate training/operator route. It does not migrate YouTube or Modal. Do not start their
ordinary credential launchers as a fallback. Publication content and operations
still require their existing review and authorization.

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

- For Checkpoint Publication or accepted Policy Releases, use `$gradlab_upload-checkpoint` in `.codex/skills/gradlab_upload-checkpoint`; use `$private-skills_model-card-author` for generated-card quality.
- For standalone YouTube maintenance, use `$gradlab_upload-youtube-video` in `.codex/skills/gradlab_upload-youtube-video`. Its `references/release-metadata.md` owns preview policy; checkpoint-release previews use the durable publisher.
- For GradLab Python releases, use `$gradlab_build-release` in `.codex/skills/gradlab_build-release`, which composes the shared `$private-skills_release-workflow`. The project launcher prepares metadata and pushes the release tag; GitHub Actions owns source validation, all release builds, artifact audits, and trusted publication. Normal release or validation requests do not build locally.

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
