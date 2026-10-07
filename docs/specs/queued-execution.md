# Queued Execution

This specification applies to GradLab's current local queue and dstack-backed execution architecture.

## Requirements

- Operator-initiated background work must use one durable and extensible local queue.
- One on-demand local worker must outlive requesting clients, recover safely after interruption, expose evidence-backed status, and exit when no work remains.
- Each current queued training Run must execute in one training container on one single-GPU host or one explicitly configured CPU-only local worker.
- CPU-only local workers must use the same exact-source immutable training image and orchestration lifecycle as GPU workers; target selection must not silently change the training recipe.
- Enabled Checkpoint evaluation defaults to separately scheduled Modal compute; an explicit training-container backend uses bounded CPU work alongside training. Either backend uses the same frozen Acceptance contract and supervisor authority. An explicit disabled mode submits no Acceptance evaluations and cannot promote a Checkpoint.
- The execution architecture must minimize separately operated services and must not require a project-operated relational database service, as recorded in [`ADR-0001`](../adr/0001-avoid-project-operated-database-service.md).
- Embedded file-backed state such as SQLite is allowed.
- The lifecycle certification gate must be deterministic, require no credentials, and preserve replayable evidence.
- Certification evidence must cover authority, delivery, evaluation-driven stopping, recovery, cancellation, and terminal correctness.
- A terminal drain must prove the complete Checkpoint inventory, the terminal status of every automatically submitted Acceptance evaluation, Promotion state, retained metric-journal inventory, selected-service delivery when online, and quiescence. A complete local-only Run has a distinct `complete_local` outcome and no remote-delivery claim.
- Checkpoints not admitted for Acceptance evaluation before Acceptance may remain unevaluated for later explicit action.
- Every enabled evaluation of a visual environment records one declared evaluation episode as a playable video by default, without adding an Acceptance episode. Nonvisual environments may omit video. Workers store evidence privately; the lease-holding supervisor delivers video references and metrics to the selected service when online.
- Opt-in Checkpoint Monitoring must evaluate every unique saved Checkpoint, including final and distinct interrupted artifacts, on the same frozen episode manifest without closing monitoring admission on Acceptance or replacing older queued work.
- Monitoring must use a bounded same-host CPU process during training and the task's full allocated CPU capacity after learner exit, including concurrent Checkpoint evaluation under shared memory, disk and retained-data limits.
- Monitoring failures must not stop the learner; retry unfinished operational work once, reuse verified complete episodes and delivered artifacts, and preserve logical identities across recovery.
- Monitoring, dataset verification, video generation, R2 delivery and selected-service delivery must be recoverable phases; online success requires all required inventories, remote acknowledgements and worker quiescence within the finite whole-task deadline. Local-only completion requires the same scientific and durable-journal evidence without service acknowledgements.
- Cancellation and exhausted budgets must stop monitoring and release resources with truthful partial evidence; training and monitoring outcomes remain separate.
- The lease-holding supervisor remains the sole selected-service writer; workers must not receive metrics-service credentials. The Run freezes one backend and delivery mode across Attempts. The validated metric journal remains authoritative and durable for the Run's lifetime, subject to an explicit retention or deletion policy. Checkpoints and other asset bytes must remain in their designated R2 stores; MLflow and W&B must retain asset links rather than artifact copies, with durable references, acknowledgements and replay without duplicate scientific points.
- A `complete_local` Run may later be explicitly synchronized to its frozen backend under the writer lease. Synchronization appends auditable service-delivery evidence and must not rewrite the original terminal receipt.
