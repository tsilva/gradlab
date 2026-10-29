# Launch Experiment

Use `gradlab experiment` for dstack-backed Runs. dstack owns compute placement
and task state; private R2 receipts own scientific state and terminal evidence.
A successful task exit alone does not establish completion or Acceptance.

## Operating prerequisites

The [entry skill](../SKILL.md) owns workflow selection, prerequisite reads,
authorization, and observer reconnection. Use its recovery matrix; authorization
persists through the task. The procedures below supply command arguments and
receipt interpretation for the chosen path.

## Launch

Resolve one launchable recipe in the selected goal's `recipes/` directory;
reusable defaults may come from `experiments/recipes/_presets/`. Separate
ablations are separate Runs. Record each launch's complete `--set KEY=VALUE`
list; overrides are composed, validated, and hash-bound by the CLI.

Run preflight with the same recipe, overrides, explicit evaluation mode, and
local target as the intended launch:

```bash
gradlab experiment operator-preflight --recipe-file <recipe-file> --json
```

Require readiness before launch. The recipe's frozen `tracking.backend` and
`tracking.delivery` determine whether MLflow/W&B access or local-only delivery
is required. Do not require W&B or Modal credentials for modes that do not use
them. Use the private operator loader; never reconstruct or inline secrets.

```bash
gradlab experiment launch \
  --recipe-file <recipe-file> --seed <seed> \
  --run-description "<specific purpose>" \
  --compute auto --max-duration <duration> --json
```

Use an explicit `--target <fleet>` only for a requested override. Paid spot
compute needs finite hourly and total-cost bounds; on-demand additionally needs
`--allow-on-demand`. Preserve a finite whole-task duration. Use
`--checkpoint-eval-backend modal|training-container|none` only when overriding
the resolved recipe mode. Defaults come from recipes or execution configuration;
Modal remains the current fallback. Record the resolved mode in the Run
Configuration. `none` admits no Acceptance evaluations or Promotion. Opt-in
Checkpoint Monitoring is a separate non-Acceptance dataset campaign and keeps
its declared limits.

Launch requires clean, pushed exact source and an immutable runtime receipt.
Modal deployment and ROM staging apply to the selected provider/evaluation
mode. `--existing-runtime-only` prohibits building or deploying missing runtime
assets. Report readiness failures instead of silently using older images.

Immediately report Run/Attempt IDs, dstack task, compute and bounds, source/image
identity, resolved tracking/evaluation modes, `metrics_url` when available, and
the public Run index. A missing URL is expected for local-only or pending service
binding; it is not evidence of a failed launch.

## Follow and verify completion

Start one yielded long-lived monitor per Run and retain its session handle:

```bash
gradlab experiment follow --run <run-id>
```

Resume the same process with empty polls; do not send a newline or interrupt it.
Each JSON line combines dstack and authoritative R2 state. Inspect `completed`,
`attempt_terminal`, `semantic.terminal`, and the receipt's frozen tracking mode:

- **Training-only execution:** successful online completion uses `succeeded`;
  successful local-only completion uses `complete_local`. Require task success,
  the complete Checkpoint inventory, retained journal, drain and quiescence.
  No canonical Acceptance terminal is expected; no solved-goal claim follows.
- **Acceptance-backed success:** additionally require the canonical terminal,
  exact evidence satisfying the Goal's declared Acceptance rule, cooperative
  learner stop, and the correct Promotion. Do not impose a fixed episode count
  or a universal success-rate threshold. Fixed-episode means finish the full
  manifest; other rules retain their declared fail-fast behavior.
- **Evaluation drain:** require terminal results for all automatically submitted
  Acceptance evaluations. Acceptance closes new admission; already submitted
  work finishes without retries. Later Checkpoints may remain unevaluated for
  future explicit action. Never require every saved Checkpoint to be evaluated.
- **Evidence delivery:** the validated journal is authoritative in every mode.
  Online success proves the selected service's acknowledged high-water mark;
  local-only success proves journal retention without claiming remote delivery.
  Enabled Monitoring also needs its own terminal inventory and quiescence.

Failed, canceled, interrupted, or resumable terminal Attempts are distinct
outcomes. Do not wait for an Acceptance receipt a failed Attempt cannot produce.
Use receipt evidence rather than inferring scientific success from dstack state.
Use `gradlab.r2_store.public_object_request` for anonymous artifact checks; Cloudflare can reject Python's default user agent with error 1010 and produce a misleading HTTP 403.

If the follow process fails or shows an anomaly:

```bash
gradlab experiment status --run <run-id> --json
gradlab experiment logs --run <run-id> --tail 200
```

Dispatch active-run potential bugs to the read-only `training_run_investigator`
with the repository, Run ID, snapshot, and logs. Continue monitoring; reuse the
investigator for the same fingerprint. Follow the conversation's progress-update
cadence, reporting meaningful changes and required attention.

## Evaluate selected published Checkpoints

An evaluation request is separate from a training-monitor request. Freeze the
selected Checkpoint IDs and catalog selection fence, then admit once through
player evaluation controls or `ManualEvaluationQueue`. Dispatch and verify the
Modal jobs even while training remains active. Preserve one durable intent and
idempotent dispatch per Checkpoint.

The lease-holding supervisor alone writes the selected service. While training
owns the lease, defer service projection, Promotion, and the manual terminal
receipt; do not defer dispatch, result verification, or list visibility. If the
user's completion gate is the Checkpoint list, finish when each selected result
has raw/verified evidence and renders there. `awaiting_projection` may remain
queued for later lease acquisition without turning this into training monitoring.

Use the immutable Run's exact-source runtime. Do not add legacy reader probing
to accommodate an older Run; any backport needs an evidence-proven repair.

## Recovery and final report

In repair mode, wait for the failed task to become terminal, its writer lease
to expire, and the CLI's quiescence interval. Retry only a supported resumable
failure with `gradlab experiment retry --run <run-id>`. A finished learner uses
drain-only recovery. `--repair-runtime` selects a committed repaired exact-source
runtime when appropriate; immutable Run scientific settings remain unchanged.
For a manifest-only pre-submit failure, use supported `resume-submit` recovery
rather than admitting a duplicate Run. Runtime publication and pushes still
require their corresponding authorization.

Report identities, source/runtime, resolved modes, stop reason and final step,
training outcome separately from Acceptance, Promotion/evaluation counts,
Checkpoint/Monitoring inventories, journal retention, selected-service delivery
or `complete_local`, host release, available service/public URLs, and the exact
`gradlab play --run <run-id>` command. Verify public artifact access and faithful
playback when the request's completion gate requires it. Do not expose private
presigned URLs or credentials.
