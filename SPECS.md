## PROJECT PURPOSE

GradLab is a reproducible reinforcement-learning workbench that carries game-agent research goals through training, evaluation, comparison, playback, publication, and local or queued execution with traceable, comparable results.

## PROJECT REQUIREMENTS

### Goals and Run Contracts

- Every research goal must declare whether it is evaluated or training-only.
- Evaluated goals must define their environment, success criteria, ranking, evaluation, and release rules.
- Training-only goals must remain ineligible for acceptance, completion, checkpoint promotion, or release.
- Every launchable training configuration must belong to one goal, declare a finite resource limit and meaningful description, and resolve all execution choices.
- Mario and ViZDoom recipes must configure an unsuccessful early-stop condition for sustained lack of improvement in a task-aligned training metric by default; an explicit launch-time override may disable it for a bounded run, and neither the condition nor its absence may establish goal acceptance or checkpoint promotion.
- Invalid or inconsistent goals, run configurations, benchmarks, capacity rules, and execution settings must be rejected before execution or external mutation.
- Each run must declare one complete observation, action, reward, discount, event, start-state, and episode-boundary contract and preserve its policy-facing semantics.
- Training-only curricula and phase-specific behavior must be declared, and evidence outside the final evaluation contract must not establish acceptance or promotion.

### Evaluation and Evidence

- Acceptance and promotion must use goal-defined checkpoint evaluation except in an explicitly declared deterministic-search workflow.
- ViZDoom Basic, Deadly Corridor, and Defend the Line checkpoint evaluations must run all 100 episodes without outcome-based early termination and determine acceptance only from the complete 100-episode mean return.
- Deterministic search may accept only a policy that produces the goal’s success event within its resource limit; the accepted policy must be published and playable.
- Recording, playback, integrity verification, reexecution, and their datasets or results must not establish acceptance or promotion.
- Evaluation and playback must preserve declared action-selection semantics by default; playback counterfactuals must be visible and ineligible for evaluation or promotion evidence.

### Provenance and Security

- Every run, cohort, campaign, result, and artifact must have a stable identity and provenance covering its goal, configuration, overrides, seed, source, runtime, environment, target, and launch context.
- Programmatic policy identities must describe execution semantics independently of their producing algorithm while preserving producer provenance.
- Active references must remain readable; purged legacy W&B or R2 data and retired PostgreSQL or Fleet state require no compatibility.
- Generated outputs and secrets must remain untracked, and normal operation must not expose credentials.
- Externally supplied executable models must be integrity-checked in full before execution; playback must not require model pre-approval.
- Installation must be reproducible, supply-chain hardened, resistant to known-bad releases, and compatible with supported workflows.

### Environment Compatibility

- A first-time user must be able to run a bundled local demonstration from the published package in one uvx command, without cloning, persistent installation, credentials, or preregistration of environment-specific inputs.
- Supported providers must offer isolated parallel execution with deterministic, nonduplicated episode streams and no cross-lane reset or boundary effects.
- Equivalent providers must preserve declared observation, action, reward, event, and episode semantics; switching providers may change only traceable provider identity.
- GradLab must apply reward scaling before clipping and disable equivalent provider-side transforms.
- Provider-specific requirements must not leak into common workflows, and every supported environment must remain trainable, evaluable, and playable through them.

### Training Results and Publication

- Every backend must honor the common local and queued lifecycle for progress, outcome metrics, checkpointing, graceful stopping, and final artifacts.
- Algorithm-specific telemetry may extend but not replace the common lifecycle.
- Training must preserve authoritative metrics and checkpoints by default, separately from job state and diagnostics.
- Observability must neither throttle training nor determine scientific outcomes.
- One writer must keep a logical run’s training and evaluation metrics together without races.
- Checkpoints, evaluation evidence, replays, and recovery data must reside in object storage rather than the metrics service.
- Evaluation must run on separately scheduled compute.
- Valid acceptance must stop training at the next safe learner boundary, close new automatic evaluation admission, allow submitted evaluations to finish, and preserve unevaluated checkpoints.
- Publication-enabled queued runs must publish every ready periodic and final checkpoint independently of evaluation.
- Published checkpoints must remain downloadable and playable without private infrastructure credentials.
- Published policies must include the artifact, portable provenance, reproducibility metadata, verified evaluation evidence, and a representative browser-safe replay.

### Playback and Human Control

- Playback must support local and remote artifacts, default to training-time policy semantics, and distinguish evaluation reproduction from counterfactual departures.
- Concurrent viewers must share one trajectory, and mutable references must refresh when their content changes.
- Bare playback must provide a fast, searchable Environment → Goal → derived Goal Variant → Run → public Checkpoint flow with hierarchical browser-history routes.
- Goal variants must derive from fully materialized contracts and be labeled by the goal plus a normalized, proven difference from its canonical contract.
- Playback discovery must use a rebuildable precomputed catalog, avoid synchronous storage scans, display only available acceptance evidence, and progressively resolve CLI references.
- Run selection must distinguish checked-in recipes from searchable launch-time overrides without requiring new recipes.
- Playback must expose resolved goal and recipe YAML; derived goal and recipe variants must show their proven differences from the corresponding canonical or checked-in base contract.
- Interactive playback must provide independently arranged, synchronized views of frames, policy inputs and decisions, transition facts, and bounded histories.
- Inspection must not alter the trajectory or policy randomness.
- Playback must expose only semantically applicable actor, critic, action-value, program, attribution, and calibration diagnostics.
- Unsupported, unobserved, and contract-incomparable diagnostics must remain visibly distinct without fabricated values.
- Critic calibration may compare values with realized returns only when environment, reward, discount, action sampling, and episode-boundary semantics match training.
- Human control must preserve declared action semantics and fail safely when focus or control is lost.
- Human-intervened results must remain ineligible for acceptance or promotion.

### Queued Operation and Benchmarks

- Queued execution must be fail-closed, isolated, and reproducible.
- Operator-initiated background work must use one durable, extensible local queue whose single on-demand worker outlives requesting clients, recovers safely after interruption, exposes evidence-backed status, and exits when no work remains.
- Retries must preserve exact provenance and durable results.
- Reported states must be evidence-backed, and cleanup must not affect active work.
- Each v1 training run must execute in one container.
- Compute placement must support local machines and cost-bounded cloud capacity without provider-specific research code.
- The orchestration stack must minimize independently operated services and require no project-owned relational database service; embedded file-backed state such as SQLite is allowed.
- Orchestration must provide a credential-free deterministic certification gate with replayable evidence for authority, delivery, evaluation-driven stopping, recovery, cancellation, and terminal correctness.
- Benchmark claims must compare equivalent environments, semantics, workloads, concurrency, and host-load conditions reproducibly.
