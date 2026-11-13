# Sequential experiment batches

`python -m gradlab.experiment_batch enqueue --file BATCH.json` schedules finite
training intents in the existing durable job queue. This optional operator
workflow uses approved `training` and `training-follow` Runbox profiles; ordinary
GradLab commands retain their interfaces.

The data-only batch names its exact clean pushed source, local target, admission
deadline, optional predecessor Run and ordered intents. Each intent records a
stable submission key, recipe path, seed, overrides, purpose and whole-task bound.
The worker performs recipe-aware preflight and exact-source admission only when
the preceding Run is terminal. It creates no worker capability while waiting.

Inspect the resulting job with `gradlab jobs show JOB_ID`. Its subjects retain
admission identities and terminal receipts. Each Run has one lock-protected
long-lived observer. Observers request `experiment follow --pretty` so large
immutable snapshots pass line-bounded output redaction intact without changing
its limits. An interrupted observer can reconnect without resubmission. Failed
access, source drift, operational failures and uncertain admissions stop the
batch for review. Never retry an uncertain admission without reconciling its
immutable manifests.

Before expanding beyond the first GPU Run, preserve its container assignment,
CUDA tensor operation and actual learner CUDA log. Record that reviewed evidence
in the job work directory's `startup-verified.json` with `run_id`, `image_digest`,
`gpu_assigned`, `cuda_tensor_verified` and `learner_cuda_verified`. All three checks
must be true. This acknowledgement never replaces the underlying evidence.

Batch cancellation stops future admissions; already admitted Runs keep their
finite bounds and observers. A completed queue proves terminal processing, not
Acceptance or recipe qualification. Review the scientific receipts separately.

Batch enqueue starts a worker restricted to `experiment-batch` jobs. It preserves
other queued job types without claiming or recovering them. For explicit scoped
recovery use `gradlab jobs flush --job-type experiment-batch`; the default flush
still processes all job types. A worker with a different active selection must
finish before a new selection can acquire the shared queue lock.
