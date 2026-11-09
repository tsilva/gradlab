---
name: gradlab_certify-lifecycle
description: Run GradLab's credential-free deterministic orchestration lifecycle certification, interpret its invariants, and replay preserved failures. Use for lifecycle certification, regression validation, or debugging.
---

# Certify Lifecycle

## When to use

Use to certify, validate, regression-test, or debug orchestration semantics:
authority, journals/delivery, evaluation, stopping, recovery, cancellation, and drain.
Tier 1 exercises the real supervisor and local persistence with scripted boundaries.

## Prepare

Apply the repository specification review once for this task and read
`docs/specs/queued-execution.md`. Discover scenarios with `--list`; the registry
in `src/gradlab/lifecycle_certification.py` owns the list. Read the requested path
in [operations.md](references/operations.md) for targeted runs or evidence replay.

## Act

```bash
gradlab experiment certify --tier simulated --json
```

Require every registered scenario and independent invariant verifier to pass.
Do not substitute isolated unit tests. For certification, compare two unchanged
runs for byte determinism; the gate normally completes within 60 seconds.

## Recover

Preserve and replay a failure bundle before changing invariants. Repair the
production path exercised by a targeted scenario, then rerun the complete gate.
Keep the original replay and scientific invariants intact.

## Finish

Report status, `report_sha256`, failed scenarios/invariants, preserved evidence,
and the exact replay command when applicable. Tier 1 proves deterministic local
orchestration; live scheduling, service access, deployed runtimes, and GPU training
require their respective integration or hardware checks.
