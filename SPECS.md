## PROJECT PURPOSE

rlab is a reproducible reinforcement-learning workbench for game-agent researchers. It carries explicit research goals through validated training, trustworthy evaluation and ranking, playback, publication, and local or queued execution while keeping results traceable and comparable.

## PROJECT REQUIREMENTS

### Goals and Run Contracts

- Each research goal must declare whether it is evaluated or training-only. Evaluated goals must define their environment, success criteria, ranking, evaluation, and release rules. Training-only goals must remain ineligible for goal acceptance, goal completion, checkpoint promotion, or release.
- Every launchable training configuration must belong to one goal, declare a finite resource limit and meaningful description, and resolve every choice needed for execution.
- Invalid or internally inconsistent goals, run configurations, benchmarks, capacity rules, or execution settings must be rejected before execution or external mutation.
- A run must resolve one complete declared observation, action, reward, discount, event, start-state, and episode-boundary contract and preserve its policy-facing semantics across training, evaluation, and evidence-bearing playback.
- Mario Level1-1 must provide a full reward system with game-score progress and a speedrun reward system without it; both must include a per-step deduction negligible relative to forward-progress reward.
- Every launchable Mario recipe must configure an unsuccessful early-stop condition for sustained lack of improvement in a task-aligned training metric; that condition must not establish goal acceptance or checkpoint promotion.
- Training-only curricula or phase-specific behavior must be explicitly declared, and evidence produced outside the finalized evaluation contract must not establish acceptance or promotion.

### Evaluation and Evidence

- Except for an explicitly declared deterministic-search workflow, acceptance and promotion must use goal-defined checkpoint evaluation rather than training metrics or playback behavior.
- A deterministic-search workflow may accept only a policy that produces the goal’s success event within its resource limit, and the accepted policy must be published and playable.
- Recorded datasets and results from recording, playback, integrity verification, or reexecution must never establish goal acceptance or checkpoint promotion.
- Policy evaluation and playback must default to stochastic action sampling. Playback may explicitly select deterministic sampling, but must visibly preserve that choice and never use deterministic results as evaluation or promotion evidence.

### Provenance and Security

- Every run, cohort, campaign, result, and artifact must have a stable identity and enough provenance to reconstruct its goal, configuration, overrides, seed, source, runtime, environment, execution target, and launch context.
- References produced by the active workflow must remain readable; no compatibility is required for purged legacy W&B or R2 data or retired PostgreSQL and Fleet state and identifiers.
- Generated outputs and secrets must remain outside tracked source, and normal operation must not expose credentials.
- Externally supplied executable models must not run until their complete content is integrity-checked and the user has trusted their source or approved that invocation after an explicit authority-and-credential warning.
- Installation must be reproducible, supply-chain hardened, resistant to known-bad releases, and compatible with supported workflows.

### Environment Compatibility

- Supported environment providers must provide correct, isolated parallel execution with deterministic, nonduplicated episode streams. Resetting, completing, or forcing a boundary in one lane must not disturb any other lane.
- Equivalent providers must preserve the same declared observations, actions, rewards, events, and episode semantics across training, evaluation, and playback. Switching providers must change only provider identity and remain traceable.
- Rlab must own reward scaling and clipping uniformly across all environments; scaling is applied before clipping, and equivalent environment-provider reward transforms must be disabled and ignored.
- Provider-specific requirements must not leak into common workflows, and every supported environment must remain trainable, evaluable, and playable through those workflows.

### Training Results and Publication

- Training must durably preserve authoritative, unambiguous metrics and checkpoints by default, keep scientific evidence separate from job state and diagnostics, and prevent observability systems from throttling training or determining scientific outcomes.
- Training and evaluation metrics for one logical run must be available together, with one writer responsible for the metrics run so evaluation cannot race training.
- Heavy checkpoints, evaluation evidence, replays, and recovery data must live in object storage rather than the metrics service.
- Evaluation must run independently on separately scheduled compute, and a goal-valid acceptance result must stop training at the next safe learner boundary.
- For queued runs with publication enabled, every ready periodic and final checkpoint must be published independently of evaluation and remain downloadable and playable without private infrastructure credentials.
- Published policies must include the policy artifact, portable provenance and reproducibility metadata, verified evaluation evidence, and a representative browser-safe replay for visual behavior.

### Playback and Human Control

- Playback must support local and remote artifacts, default to the checkpoint’s training-time policy-facing semantics, make evaluation-contract reproduction and counterfactual departures explicit, keep concurrent viewers on one trajectory, and refresh mutable references when their content changes.
- Bare playback must open a searchable repository-backed project-to-goal selection flow followed by W&B-backed run and public-checkpoint selection; project, goal, run, and checkpoint selections must have hierarchical resource routes with browser-history navigation; checkpoint lists must show available goal-required acceptance results without fabricating unavailable partial evidence; a less-specific CLI W&B reference must preselect its matching level, while an exact CLI checkpoint source may enter playback directly.
- Run-selection views must visibly distinguish checked-in recipes from launch-time configuration overrides and make exact override values searchable without requiring a new checked-in recipe.
- Interactive playback must provide independently arrangeable, synchronized views of game frames, policy inputs and decisions, transition facts, and bounded histories without inspection changing the trajectory or policy randomness.
- Critic calibration diagnostics must compare value estimates with realized returns only when environment, reward, discount, action-sampling, and episode-boundary/bootstrap semantics match training; otherwise the comparison must be visibly unavailable.
- Human control must preserve declared action semantics, fail safe when focus or control is lost, and keep all human-intervened results ineligible for acceptance or promotion.

### Queued Operation and Benchmarks

- Queued execution must be explicit, fail closed, isolated, and reproducible. Attempts must preserve exact provenance and durable results across interruption and retries, report only evidence-backed states, and clean unused resources without affecting active work.
- Each v1 training run must execute in one container while compute placement supports local machines and cost-bounded cloud capacity without provider-specific research code.
- The orchestration stack must minimize independently operated services and must not require a project-owned relational database.
- The orchestration lifecycle must provide a credential-free deterministic certification gate with replayable failure evidence for authority, delivery, evaluation-driven stopping, recovery, cancellation, and terminal-state correctness.
- The clean-slate orchestration refactor is accepted only after a B3 Mario Level1-1 run reports its training and evaluation metrics, durably publishes every checkpoint, completes the goal-owned 100-episode acceptance evaluation, and early-stops because all 100 episodes succeed.
- Benchmark claims must be reproducible and compare equivalent environments, semantics, workloads, concurrency, and host-load conditions.
