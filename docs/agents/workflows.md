# Workflow Ownership

Use the owning source when a rule changes. Skills provide operating guidance;
they do not create a second scientific contract or replace command validation.

| Concern | Authoritative source |
| --- | --- |
| Essential product commitments | Root [`SPECS.md`](../../SPECS.md) |
| Detailed capability commitments | [`docs/specs/README.md`](../specs/README.md) and its indexed specifications |
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
| Evaluate a completed local source-overlay Run on its own host | `gradlab evaluate-local` with the recorded baseline and overlay receipts, exact checkpoint digest, and a private queue directory |
| Tune this recipe's numeric PPO/A2C hyperparameters | `gradlab_autoresearch`, fixed training-only protocol |
| Optimize architecture, rollout size, rewards, or another backend | Explicit experiment plan covering the requested axes |
| Publish this unevaluated Checkpoint | `gradlab_upload-checkpoint`, Checkpoint Publication path |
| Publish this accepted Policy to HF/YouTube | `gradlab_upload-checkpoint`, accepted-release path |
| Update this existing YouTube description | `gradlab_upload-youtube-video`, existing-video path |
| Publish the next Python package release | `gradlab_build-release`, publication path |
| Build a release candidate locally | `gradlab_build-release`, local-candidate path |
| Certify or debug the orchestration lifecycle | `gradlab_certify-lifecycle` |

## Agent service access

For the reviewed local training coordinator and its R2 stores, use the dedicated
[training Agent Vault launcher](training-vault.md). Its operator and supervisor
bundles are separate; supervisors own finite sessions that survive client exit
and never reach the offline learner. Other service modes require their own
reviewed access route.

The opt-in `scripts/huggingface_vault.py` launcher selects the dedicated Agent Vault bundle
`local-gradlab-huggingface` for `huggingface.co`. The child sees the exact
`HF_TOKEN=hf_gradlab_proxy_placeholder`, a scoped proxy session, and process-local
CA trust. The launcher pins the Infisical instance and proxy CA, clears the
inherited environment, and requests a 30-minute session with Infisical 0.45.6.
The child runs inside a macOS sandbox that denies direct upstream networking,
private credential files, Keychain access, credential helpers and writes outside
its disposable workspace. No global CLI, app or model default is changed.

```sh
python3 -I -S -B scripts/huggingface_vault.py check
python3 -I -S -B scripts/huggingface_vault.py hf auth whoami
```

The original Gradlab Secret Manager `dev` folder `/agents/huggingface` holds the
fine-grained source key. After rotation, update the Agent Vault service through
trusted administration and repeat the identity and scope checks. Keep the
credential scoped to the reviewed repository operations. Collection permissions
apply across the account and require separate review. The publishing workflow
must still honor its publication approvals.

Trusted setup stores a metadata-only profile at
`~/.config/infisical/huggingface-clients/gradlab.json`, with owner-only directory
and file permissions. It pins the native source binding, exact repository files,
runtime, SDK sources, public CA and local proxy. Ordinary commands cannot approve
their own source changes or retrieve the service key. A changed binding or failed
authentication stops without raw-key or cache fallback.

The `check` mode runs the actual publication credential resolver and installed
Hub SDK identity call. The `hf` mode invokes the installed CLI for `auth whoami`,
`repo-files ls` and explicit small-file/folder `upload` operations. It returns only
a sanitized success/failure summary. Upload needs an explicit repository and
local payload; review the entire content and operation before use. Payloads must
be inside the checkout; private working notes, credentials and repository internals
are excluded. An explicitly prepared private payload may live under
`.internal/huggingface-vault/payloads/`. The tool does not authorize uploading a
Policy or Checkpoint outside the durable Publication workflow.

Native fake tests exercised the actual SDK and CLI upload, verified the received
bytes and substituted header, tested a rejected upstream that reflected the fake
key, and proved direct-network/private-file/helper denial and no cached-token
fallback. A real read-only identity check passed; no live upload was performed.
Run the offline guard tests with `pytest tests/test_huggingface_vault.py
tests/test_publication_credentials.py` in the locked development environment.

Large-file LFS and Xet transfers remain unsupported and fail before their upload
or token-exchange requests. They need separate compatibility work; do not enable
direct networking or raw credentials. The older global launcher remains installed
without OS sandboxing and is not an agent fallback. Complete durable releases
also require separately reviewed YouTube and R2 routes. Operator, training and
Modal credential launchers remain separate and must not be used as a shortcut.
