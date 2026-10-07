# Selectable metrics backends

Status: implementation plan supporting [issue #58](https://github.com/tsilva/gradlab/issues/58), updated 2026-09-29. No metrics backend, host deployment, or specification has been changed.

## Outcome

Let a Run select exactly one of W&B or MLflow through an inherited `tracking.backend` setting. W&B remains the default. A project-wide setting applies to every Goal; a composed Goal, shared recipe preset, leaf recipe, or launch override can replace it. Online delivery is the default; the bundled no-credential example explicitly selects `local_only`. The resolved backend and delivery mode are frozen for the Run and every retry. Adding a third backend should require an adapter and its contract tests, without changes to the learner, supervisor lifecycle, goal semantics, or Playback presentation.

Both backends must preserve the same metric registry, scientific step axes, training and evaluation evidence, videos, delivery proof, public result identity, and recovery behavior. Selection changes where GradLab projects a Run; it does not change Acceptance or Promotion authority. New common telemetry and comparison views start with Runs created under the new contract. Existing published Releases remain usable, but historical W&B Runs need no migration or legacy reader.

For new Runs, make the validated GradLab metric journal the durable scientific record and treat the selected backend as a remotely verified projection and research UI. Retain the complete journal for the Run's lifetime, subject only to explicit deletion or retention policy, and maintain a bounded, indexed read model. This gives Playback, comparisons, public telemetry, and a future third backend one provider-neutral source.

## Existing constraints and seams

- The learner already emits validated structured frames to a SQLite outbox. The lease-holding supervisor alone publishes them. Private R2 metric segments restore that outbox after interruption. Keep this ownership and the current R2 checkpoint and evidence locations. New-schema metric segments become durably retained, hash-verified journal objects with a compact index; a backend switch must never depend on that backend's retention or query API.
- `wandb_publisher.py` converts six frame kinds, binds three scientific axes, publishes promotion and terminal summaries, and resumes a logical Run. `run_supervisor.py` and `local_wandb.py` use W&B-specific remote high-water checks before claiming delivery.
- `RunManifest`, `TerminalReceipt`, local publication receipts, the playback catalog, leader queries, reports, workspaces, CLI output, credentials, and runtime image currently name W&B. A writer-only wrapper would leave these paths inconsistent.
- `goal_contract_sha256()` currently hashes the entire composed Goal. A `tracking` field in `_goal.yaml` would otherwise create a new Goal Revision or Variant for an operational choice. Backend selection must be excluded from the semantic Goal projection while remaining visible in resolved Run Configuration and provenance. Existing Goal hashes with no `tracking` field must remain byte-for-byte unchanged.
- The queued-execution specification requires no operated relational database service. [MLflow's file-backed SQLite metadata store](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/) satisfies that constraint; a required PostgreSQL deployment would not. SQLite capacity and backup behavior need validation at GradLab's actual event rate.
- The publication specification requires public telemetry links for a public result. A private MLflow URL cannot be the only public telemetry surface. The public GradLab catalog must provide backend-independent, durable result summaries and playable media; an authorized backend deep link can supplement them.

## Configuration and identity

Use the same optional setting at each authored layer:

```yaml
tracking:
  backend: wandb  # or mlflow
```

Add `experiments/_settings.yaml` as the checked-in project-wide layer, initially declaring `wandb`. Resolve in this order, with the later explicit value winning:

1. Built-in `wandb` default, so old recipes continue to resolve identically.
2. `experiments/_settings.yaml`.
3. Goal composition, including its shared defaults and leaf `_goal.yaml`.
4. Recipe composition, including shared presets and the leaf recipe.
5. Launch `--override tracking.backend=...` (and a matching direct-local CLI option where no Goal recipe is used).

Reject unknown backends and, after migrating checked-in recipes, reject legacy `logging.wandb_mode` with a clear validation error. For online delivery, also reject missing selected-backend operator profiles or unreachable selected-backend endpoints before external mutation. `tracking.backend` changes only the operational Run Configuration: Goal Revision, Goal Variant, environment hash, Acceptance, and ranking stay fixed. Recipe/launch override provenance still records the selection. Show the resolved value and its winning source in config inspection and launch output.

Keep connection details and secrets out of Goal and recipe YAML. Private operator configuration maps a logical MLflow profile to its endpoint and credentials. Freeze the backend and logical profile ID in the Run manifest, and store the provider's run ID and URL in a create-only R2 binding receipt. Keep GradLab's `gradlab-<hex>` Run ID primary in every catalog, receipt, tag, and public link. A changed root setting affects new Runs only; attempts and `gradlab sync` use the original binding. A remote-create crash must reconcile by the unique GradLab Run ID tag, reject ambiguous matches, and never silently create a second scientific Run.

Use `tracking.delivery: online | local_only` in new configuration, defaulting to `online`. Online Runs require remote delivery proof and cannot be relabeled local-only after a delivery failure. A `local_only` Run reaches a distinct `complete_local` outcome after its journal and required scientific evidence are durable; an explicit later `gradlab sync` projects to the Run's frozen backend without rewriting its terminal receipt. Direct local Runs may keep the journal on local disk without external credentials; queued Runs seal it to private R2. A direct local Run must upload the required journal and artifacts before public Publication. Publication and Release can precede tracker sync if their own evidence, public telemetry, and durability requirements are met.

## Deep module at the outbox seam

Introduce a `MetricDelivery` module between the existing supervisor and provider adapters. Its small interface accepts the resolved immutable Run binding and canonical journal frames, publishes promotion/terminal projections, reports a verified contiguous remote high-water mark, and closes a Run. It owns common validation, ordering, retry policy, local acknowledgements, lag/health measures, and receipt production. The supervisor retains the writer lease, scheduling, checkpoints, evaluation, and terminal decisions; the learner remains unaware of the backend. Journal sealing precedes external projection, and the terminal receipt separately records journal and selected-backend high-water marks.

Each `MetricsBackend` adapter must implement replay-safe `open_or_resume`, `publish_frame`, `publish_summary`, `observe_delivery`, and `close` operations. The interface must state ordering, idempotency/reconciliation, timeout, media, and remote-visibility guarantees, not merely method signatures. A common journal reader supplies validated config, bounded metric history, and summaries to Playback and comparison code for new Runs. Reuse the common metric registry for names, units, placement, reducers, and scientific axes. Keep provider-native workspace/report management outside the core delivery interface.

W&B becomes one adapter. MLflow becomes the second. Neither adapter may determine Training Success, Acceptance, or Promotion. Provider-specific identifiers and SDK objects stay inside adapters. The supervisor and receipts use neutral `metrics_*` names in new schema versions; no compatibility reader for historical `wandb_*` Run receipts is required. Do not reinterpret immutable historical receipts or break existing published Releases.

For MLflow, map train metrics to training step, evaluation metrics and media to checkpoint step, and operational metrics to event sequence. Store full nested Run Configuration as an artifact and expose small searchable dimensions as tags; [MLflow's parameter and tag limits](https://mlflow.org/docs/latest/api_reference/rest-api.html) make a direct W&B-config copy unsuitable. Render tables and video references from the canonical evidence in the GradLab catalog, with useful MLflow artifacts or links where supported. Keep checkpoint bytes and authoritative evaluation evidence in their existing R2 buckets. Compute reducers and leader projections from the journal once, then give both adapters the same values.

**Replay proof is a hard gate.** [MLflow accepts repeated metric writes](https://mlflow.org/docs/latest/api_reference/rest-api.html) at the same step, unlike GradLab's current W&B replay assumption. A spike must exercise ambiguous acknowledgements, server restart, retry after local acknowledgement loss, and media upload interruption against a real local MLflow server. Candidate implementation: publish a frame and a monotone delivery marker together, read the marker after uncertain calls, and reconcile a frame by its durable event ID/sequence before replay. Use content-addressed media paths. Adopt that design only if the selected MLflow version proves its transaction and read-after-write behavior. If direct MLflow writes cannot keep its own charts free of duplicate points, design and prove another ingest protocol before enabling MLflow Runs; the canonical journal protects scientific truth but does not by itself make the selected UI seamless. Never claim terminal success solely from an SDK return.

## Beast-3 deployment

Start with an isolated, version-pinned MLflow service on Beast-3, host-owned persistent SQLite state, an explicit backup/restore test, and a private authenticated endpoint. Keep MLflow artifacts in a separate R2 prefix or bucket through [MLflow's S3-compatible artifact support](https://mlflow.org/docs/latest/self-hosting/architecture/artifact-store/) with server-side credentials; do not give its credentials to learners or Modal. Record the endpoint only in operator-local configuration. Probe accessibility from each selected execution target before launch. A Beast-3-only route is sufficient for the first pilot; Beast-2, the Mac worker, and cloud compute must fail preflight until they have a verified secure route. Do not silently fall back to W&B. If database restoration is impossible, replay the GradLab journal into a replacement MLflow Run under an append-only audited binding while retaining the original binding record.

Measure event throughput, query latency, database growth, video handling, CPU/memory/disk use, and training interference while a representative Beast-3 run is active. Keep the MLflow process outside the one-container dstack task and outside GradLab runtime-image cleanup. A service outage must leave the outbox and R2 recovery evidence intact and produce a truthful incomplete delivery state.

## Implementation sequence and gates

1. **Contract and config.** Align authoritative specifications with the agreed backend and local-only outcomes, including any required exact approval of root `SPECS.md` text. Add the project/Goal/preset/recipe/CLI precedence resolver, semantic Goal projection, validation, provenance, and frozen manifest selection. Migrate checked-in recipes and make the bundled no-credential example explicitly local-only. Prove that a tracking-only change does not alter a Goal Revision or Variant and that each layer overrides exactly its predecessor.
2. **Neutral scientific record and delivery.** Retain new-schema R2 metric journals durably with bounded indexes and reads. Extract common outbox, receipt, health, and terminal logic behind `MetricDelivery`. Keep W&B as default and pass the existing deterministic lifecycle certification and W&B regressions before adding MLflow. Verify new Runs remain readable after the selected backend is unavailable and after local state is removed.
3. **MLflow spike and adapter.** Pin MLflow through `uv.lock` under the seven-day package gate, build a credential-free SQLite-server fixture, prove or reject the replay design, then implement all frame kinds, summaries, binding recovery, and remote visibility. Force failures before send, after remote commit, before local acknowledgement, during media transfer, and during final drain.
4. **Readers and public surfaces.** Make Playback enrichment, run comparison, leaders, and terminal links read new Runs through the common journal and catalog. Preserve contract mismatch suppression and exact scientific axes. Publish an immutable, validated per-Checkpoint metric snapshot into the public R2 catalog so public Playback does not need backend credentials. Provide backend-independent public telemetry and GradLab-owned comparison/report views before declaring presentation parity. Keep W&B-native workspace/report commands available for W&B Runs. Existing published Releases remain usable without a historical W&B Run reader in the new views.
5. **Operations and rollout.** Deploy the isolated Beast-3 trial, test backup restoration and network paths, run local and dstack pilot Runs with both backends, compare complete metric histories and receipts, then opt one Goal or recipe into MLflow. W&B remains the global default until repeated full lifecycle Runs pass. Update `METRICS.md`, `COMPUTE.md`, operator templates, experiment docs, CLI help, runtime lock/image, and validation in the same implementation change.

Acceptance requires identical normalized scientific histories and checkpoint evidence for matched synthetic Runs; no duplicate scientific points in the journal or selected UI after replay; complete journal and remote high-water plus Promotion visibility before successful online terminal receipts; a distinct durable `complete_local` outcome without remote acknowledgement; working public result links and videos; usable existing published Releases; no metrics credentials in learners or Modal; and deterministic credential-free lifecycle certification for both adapters. A new adapter should pass the same contract suite without edits to core supervisor, scientific contracts, or Playback UI.

## Change map

| Area | Main files and responsibility |
| --- | --- |
| Configuration | `recipe_documents.py`, `goal_schema.py`, `recipe_schema.py`, `train_config.py`, `config_validation.py`: compose and validate the hierarchy; keep `tracking` out of semantic Goal hashes. |
| Frozen contracts | `run_contracts.py`, `experiment_cli.py`, `run_authority.py`, `local_publication.py`: backend identity, create-only binding, versioned neutral receipts, and public links. |
| Delivery | New `metric_delivery.py` and `metrics_backends/` package; `wandb_publisher.py`, `supervisor_runtime.py`, `run_supervisor.py`, `local_wandb.py`: common journal/outbox lifecycle and two concrete adapters. |
| Reading and presentation | `metric_store.py`, R2 journal/index, `play_catalog.py`, `wandb_leaders.py`, `wandb_reports.py`, `wandb_workspaces.py`, public catalog projection: provider-neutral reads and public views. |
| Operations | `operator_credentials.py`, `ops/operator.example.toml`, training image/dependency lock, `COMPUTE.md`: private endpoint/credentials, server and fleet preflight. |

The listed files are implementation targets, not permission to rewrite unrelated concurrent edits. Reinspect the current tree before each phase.

## Effort and decision points

For one developer, budget roughly six to nine engineering weeks: about one week for config/contracts, one to two for the durable journal and neutral delivery, one to two for the MLflow replay spike and adapter, one to two for backend-independent readers/reports, and about one for host deployment and full-run validation. These are planning ranges, not elapsed-time commitments. The MLflow replay spike is a go/no-go point; a failed proof changes the ingest design and estimate before production work continues. Beast-3 hosting alone is a much smaller task, but it does not establish swappable GradLab Runs.

## Specification changes to approve before implementation

Root `SPECS.md`, under `### Environments and Execution`, proposed additions:

> - Every Run must freeze a metrics backend selected through project-wide, Research Goal, recipe, and launch configuration; W&B is the default.
> - Either metrics backend must preserve the same scientific measures, evidence, Run identity, and declared delivery guarantee.
> - An explicitly local-only Run must have a distinct outcome and support later sync to its frozen backend.

`docs/specs/queued-execution.md` currently assumes every Run delivers to W&B. Replace its terminal-drain requirement and three W&B-specific requirements, and add the explicit local-only rule:

> - A terminal drain must prove the complete Checkpoint inventory, the terminal status of every automatically submitted Acceptance evaluation, Promotion state, mode-required metrics delivery, and quiescence.
> - Every enabled evaluation of a visual environment records one declared evaluation episode as a playable video by default, without adding an Acceptance episode. Nonvisual environments may omit video. Workers store evidence privately; the lease-holding supervisor delivers video and metrics to the selected metrics backend for online Runs and seals them durably for local-only Runs.
> - Monitoring, dataset verification, video generation, R2 delivery and mode-required metrics delivery must be recoverable phases; success requires all required inventories, acknowledgements and worker quiescence within the finite whole-task deadline.
> - The lease-holding supervisor remains the sole writer to the Run's selected metrics backend for online Runs; workers must not receive its credentials. Actual representative video media is permitted, with durable references, acknowledgements and at-least-once transport semantics.
> - An explicitly local-only Run may finish with a distinct locally complete outcome when its journal and required evidence are durable; a failed online Run must recover selected-backend delivery rather than change delivery mode.

`docs/specs/training-datasets.md` also names W&B in five requirements. Replace only those requirements with:

> - When at least one episode is recorded, generate one full-episode video from the selected episode's original committed RGB frames at contracted cadence, without environment replay; store canonical bytes/hashes in R2 and deliver actual playable video plus full-set metrics through the sole selected-backend writer for online Runs. Local-only Runs retain the canonical video and metrics for public GradLab telemetry and optional later tracker sync.
> - Track episode execution, dataset verification, video generation, R2 media and mode-required metrics delivery separately so late failures do not restart scientific work unnecessarily.
> - Terminal receipts must prove required Checkpoint/evaluation/dataset/video inventories, mode-required metrics delivery and worker quiescence. Online success requires selected-backend acknowledgement; locally complete Runs must prove durable local-only evidence. Cancellation/deadline/budget exhaustion must preserve honest partial evidence and release resources; training and monitoring outcomes remain separate.
> - Measure representative early/intermediate/stronger compatible Checkpoints and long episodes, including inference, capture, encoding, R2, journal and selected-backend delivery when online, memory/spool/retained bytes and concurrent learner throughput; unavailable representative inputs mean incomplete calibration.
> - A separately authorized bounded live campaign must cover at least three unique Checkpoints including final, full configured evaluation/recording sets, active contention and parallel final drain; inspect actual selected-backend charts for online Runs or GradLab telemetry for local-only Runs, playable video in the in-app browser, remote integrity, cleanup and resource release. Live HF Publication requires an explicitly authorized target.

The existing no-operated-relational-database requirement remains unchanged. `METRICS.md` is implementation authority rather than stakeholder intent; revise its W&B-authority and journal-retention rules alongside code after these specification changes are approved. The exact root `SPECS.md` addition needs separate explicit approval before editing that file.
