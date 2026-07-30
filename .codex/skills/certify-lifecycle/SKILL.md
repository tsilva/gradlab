---
name: certify-lifecycle
description: Run and interpret gradlab's credential-free deterministic Tier 1 orchestration lifecycle certification. Use when the user asks to certify, regression-test, validate, or debug fleet/dstack orchestration; verify checkpoint-to-Modal evaluation, W&B delivery, eval-driven early stopping, lease fencing, cancellation, or drain-only recovery; or replay a preserved lifecycle failure.
---

# Certify Lifecycle

Use the checked-in `gradlab experiment certify` gate. It executes the real
supervisor authority, SQLite outbox/ledger, file-backed R2 protocol, checkpoint
publication, evaluation dispatch/validation, sole-writer W&B projection, early
stop, promotion, recovery, cancellation, and terminal-receipt logic against
deterministic local boundaries. It denies network access and requires no
credentials.

Read `SPECS.md` before running or changing the gate.

## Run the complete Tier 1 gate

```bash
gradlab experiment certify --tier simulated --json
```

Do not substitute isolated unit tests for this command. A pass requires every
registered scenario and independent invariant verifier to pass. Report:

- overall status and `report_sha256`;
- each failed scenario and invariant, if any;
- whether a failure bundle was preserved;
- the exact replay command when a bundle exists.

The complete gate must remain byte-deterministic across two unchanged runs and
normally finish in under 60 seconds.

## Run targeted scenarios

List available scenarios:

```bash
gradlab experiment certify --list
```

Select one or more:

```bash
gradlab experiment certify \
  --scenario full-lifecycle \
  --scenario parallel-run-isolation \
  --json
```

Use a targeted scenario while repairing a failure, then rerun the complete gate
before declaring the lifecycle certified.

## Preserve or replay evidence

Keep raw file-backed buckets, SQLite ledgers, supervisor transcript, simulated
W&B events, report, and replay manifest:

```bash
gradlab experiment certify \
  --tier simulated \
  --artifacts-dir ~/.config/gradlab/runs/certification/manual-check \
  --json
```

The destination must be empty. Without an explicit destination, passing
artifacts are temporary; failing artifacts are preserved under
`~/.config/gradlab/runs/certification/failure-<report-prefix>/`.

Replay the exact scenario set:

```bash
gradlab experiment certify \
  --replay <failure-bundle>/replay.json \
  --json
```

Compare the failing invariant, raw R2 objects, SQLite rows, transcript event
order, and W&B event IDs/high-water marks. Repair the production path exercised
by the scenario rather than weakening the invariant or changing the replay.

## Scope boundary

Tier 1 proves deterministic orchestration semantics with real local persistence
and scripted service boundaries. It does not prove live dstack scheduling, real
R2 permissions/consistency, W&B backend visibility, Modal deployment, ROM
provisioning, or GPU training. Those belong to service-integration and
hardware-specific acceptance tiers.
