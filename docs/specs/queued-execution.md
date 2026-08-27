# Queued Execution

This specification applies to GradLab's current local queue and dstack-backed execution architecture.

## Requirements

- Operator-initiated background work must use one durable and extensible local queue.
- One on-demand local worker must outlive requesting clients, recover safely after interruption, expose evidence-backed status, and exit when no work remains.
- Each current queued training Run must execute in one training container on one single-GPU host.
- Checkpoint Acceptance evaluation must use separately scheduled Modal compute and state isolated from active training.
- The execution architecture must minimize separately operated services and must not require a project-operated relational database service, as recorded in [`ADR-0001`](../adr/0001-avoid-project-operated-database-service.md).
- Embedded file-backed state such as SQLite is allowed.
- The lifecycle certification gate must be deterministic, require no credentials, and preserve replayable evidence.
- Certification evidence must cover authority, delivery, evaluation-driven stopping, recovery, cancellation, and terminal correctness.
- A terminal drain must prove the complete Checkpoint inventory, the terminal status of every automatically submitted evaluation, Promotion state, metrics delivery, and quiescence.
- Checkpoints not admitted for evaluation before Acceptance may remain unevaluated for later explicit action.
