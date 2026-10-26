---
name: gradlab_launch-experiment
description: Launch and follow dstack-backed GradLab training, or evaluate selected published checkpoints through the durable queue. Verify scientific receipts and resource release; repair when authorized.
---

# Launch Experiment

## When to use

Use for a dstack-backed launch, monitoring an existing Run, or manual evaluation
of selected published Checkpoints. A checkpoint-list evaluation request has its
own completion gate and does not imply following training to its end.

## Prepare

Apply the repository specification review once for this task; reuse already-read
unchanged documents. Read `COMPUTE.md`, `docs/specs/queued-execution.md`, and the
selected goal/recipe. Local compute requires `~/.config/gradlab/instances.md`.
Read `docs/specs/training-datasets.md` when Monitoring is enabled.
Read the requested path in [operations.md](references/operations.md).

The CLI owns validation and exact-source runtime readiness. Private R2 receipts
own scientific state. Use existing operator credential loading.

## Act

For a launch, run `gradlab experiment operator-preflight` with the intended
recipe, overrides, evaluation mode, and target, then `gradlab experiment launch`.
Require readiness, clean pushed source, immutable runtime, and finite duration/cost
bounds. Immediately report identities, compute/bounds, source/runtime, modes,
public Run index, and `metrics_url` when available.

Keep one `gradlab experiment follow --run <id>` process per Run; resume its
session with empty polls. For manual evaluation, freeze the selection and admit
once through the player or `ManualEvaluationQueue`, following the reference.
Training's writer lease defers projection, not dispatch or verification.

## Recover

Authorization persists through the task. A request to fix, harden, or finish the
workflow enables evidence-preserving repair mode; recognize it without repeated
approval questions. A launch/monitor request alone uses observe mode.

| Action | Observe mode | Authorized repair mode |
| --- | --- | --- |
| Repeat transient reads or reconnect an exited observer | Continue within the existing monitoring deadline; retain Run identity and snapshots | Same |
| Diagnose an active-run potential bug | Delegate read-only to `training_run_investigator`; keep monitoring | Same; use findings to guide repair |
| Repair source | Report findings | Preserve evidence, reproduce the cause, fix, and add a meaningful regression |
| Retry or resume submission | Report the supported recovery action | Use supported CLI recovery after terminal/lease/quiescence checks, within authorized attempt, duration, and cost bounds |
| Change credentials, replace infrastructure, cancel unrelated work, commit/push, or perform destructive cleanup | Requires corresponding authorization | Requires corresponding authorization |

Reuse a healthy observer. Reconnecting an exited observer creates no Attempt and
never resubmits work. Resolve an uncertain launch against durable manifests.
Preserve scientific contracts, capacity, and resource bounds. Read recovery details
in the reference before retrying or repairing.

## Finish

Use the reference's receipt checklist. Distinguish training outcome, Acceptance,
Promotion, evaluation/Monitoring inventories, retained journal, selected-service
delivery or `complete_local`, and resource release. Failed Attempts have their own
terminal evidence. Include public/service links and `gradlab play --run <id>`.
