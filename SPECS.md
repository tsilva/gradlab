## PROJECT PURPOSE

GradLab is a public, reproducible, general-purpose reinforcement-learning research workbench for researchers, with games as its flagship domain. It carries Research Goals through training, evaluation, comparison, Publication, Playback, and local or queued execution while supporting compute operators and consumers of published Policies with traceable, comparable results.

## PROJECT REQUIREMENTS

### Research Goals and Runs

- Every Research Goal must have a stable identity and declare its environment, goal type, and Training Success criteria.
- Every Evaluated Goal must define its Acceptance criteria, ranking rules, evaluation method, and Release requirements.
- A Training-Only Goal may publish Checkpoints but must remain ineligible for Acceptance, Promotion, or Release.
- Every change to a Research Goal contract must produce an immutable Goal Revision, and one Goal Revision must be the current default.
- Every Goal Variant must identify a normalized change to the scientific meaning of its Goal Revision.
- Every Run must identify its exact Research Goal, Goal Revision, applicable Goal Variant, and resolved Run Configuration.
- Every Run must state its purpose, resolve all execution choices, and declare a finite resource limit.
- Every Run must preserve a complete contract for observations, actions, rewards, discounts, events, start states, and episode boundaries.
- Every Run must declare its curricula, training phases, and phase changes.
- GradLab must reject invalid or inconsistent goals, runs, benchmarks, compute rules, and execution settings before work or external mutation begins.

### Scientific Evidence

- Training Success must remain a training-time proxy and must not confirm that a Research Goal is solved.
- An early-stop rule must be declared when used and must not cause Acceptance or Promotion.
- Acceptance must use an immutable Policy, the Evaluated Goal’s declared evaluation method, evaluation seeds different from training seeds, and state isolated from active training.
- Acceptance based on a fixed-episode mean must use every declared episode before it makes its decision.
- Deterministic Search must grant Acceptance only when a Policy produces the declared success event within the declared resource limit.
- Training measures, recordings, playback, integrity checks, reexecution, and human-controlled results must not cause Acceptance or Promotion.
- Evaluation and Faithful Playback must preserve the recorded action-selection rules.
- Counterfactual Playback must remain visible and ineligible as Acceptance or Promotion evidence.
- Acceptance must stop training at the next safe update boundary and close admission of new automatic evaluations.
- Evaluations submitted before Acceptance must be allowed to finish, and unevaluated Checkpoints must be preserved.

### Provenance, Security, and Compatibility

- Every Run, Attempt, Policy, Checkpoint, result, published artifact, and Release must have a stable identity.
- Provenance must record the Research Goal, Goal Revision, Goal Variant, Run Configuration, overrides, training seeds, evaluation seeds, source, runtime, environment, provider, compute target, and launch context.
- A Policy identity must describe its execution rules independently of its trainer and algorithm while preserving producer provenance.
- Current GradLab must support only current GradLab-owned contracts and workflows.
- Every Release must remain usable through its recorded execution requirements when current GradLab contracts change.
- Generated outputs and secrets must remain untracked, and normal operation must not expose credentials.
- Machine-specific inventory, fleet names, hostnames, SSH identities, and operator endpoints must remain outside tracked source.
- External Policy artifacts must use known data-only formats and pass integrity and compatibility checks before use.
- GradLab must reject externally supplied executable Policy artifacts.
- Supported installation methods must be reproducible, supply-chain hardened, and resistant to newly published or known-bad dependencies.
- A new user must be able to run a bundled local example from the published package with one command and without cloning, persistent installation, credentials, or private environment assets.

### Environments and Execution

- GradLab must call a capability supported only when an authoritative project registry lists it and it passes its required contract checks.
- Supported providers must isolate parallel lanes and provide deterministic, nonduplicated episode streams without cross-lane reset or boundary effects.
- Equivalent Providers must preserve observations, actions, rewards, discounts, events, start states, and episode boundaries.
- Every Run must record its provider identity.
- Every Run must declare its reward transforms and their order, and providers must not apply undeclared or duplicate transforms.
- Every supported environment must work through common training, evaluation, Publication, and Playback workflows.
- Equivalent Run contracts must work on local and cost-limited cloud compute without provider-specific research code.
- Every supported execution system must provide the common lifecycle for each execution mode that it declares.
- The common lifecycle must cover progress, outcome measures, Checkpoints, graceful stopping, and final artifacts.
- Algorithm-specific measures may extend but must not replace the common lifecycle and scientific record.
- Each Run must have one consistent record that joins its training measures, evaluation measures, Checkpoints, and final result without loss or races.
- Scientific measures and Checkpoints must remain distinct from job state and diagnostics.
- Diagnostics must not determine Training Success, Acceptance, or Promotion or change Policy behavior.
- Queued work must continue after its requesting client exits, recover safely after interruption, report evidence-backed status, and stop consuming compute when no work remains.
- Every retry must create a new Attempt under the same Run while preserving exact provenance and durable results.
- Cleanup must not damage active work.
- Queued execution must be fail-closed, isolated, and reproducible.

### Publication

- Public surfaces for one result must use one stable identity, link to each other, and distinguish the Policy, Checkpoint, evaluation evidence, and representative replay.
- Published work must be organized by Research Goal, immutable Release, explicit Policy Lineage, and curated environment and goal indexes.
- Every publication-enabled Run must declare a Publication Policy that selects which Checkpoints to publish independently of evaluation status.
- Every published Checkpoint must remain downloadable and playable without private infrastructure credentials.
- Published artifacts must remain durably available independently of transient job and diagnostics state.
- Every Release must contain its promoted Policy, provenance, reproducibility information, verified evaluation evidence, representative replay, and recorded execution requirements.

### Playback and Comparison

- Playback must support local and public remote artifacts.
- Playback must distinguish Faithful Playback, Evaluation Reproduction, and Counterfactual Playback.
- Playback and Evaluation Reproduction must not create Acceptance evidence.
- Viewers in one Playback Session must share one trajectory, and separate Playback Sessions must remain independent.
- New content behind a mutable reference must not change an active Playback Session without an explicit reload.
- Users must be able to find public Checkpoints by environment, Research Goal, Goal Revision, Goal Variant, Run, and Checkpoint.
- Checkpoint views must distinguish Training Success measures, evaluation-acceptance evidence, ranking measures, and missing evidence for the exact Goal Revision and Goal Variant.
- Playback must expose the resolved Research Goal and Run Configuration and explain every Goal Variant’s scientific difference from its Goal Revision.
- Users must be able to find and compare Runs created from checked-in recipes and launch-time overrides without creating a new recipe.
- Interactive inspection must expose synchronized frames, Policy inputs and decisions, transition facts, and bounded histories without altering the trajectory or Policy randomness.
- Playback must show only diagnostics that apply to the selected Policy and recorded data.
- Unsupported, missing, and scientifically incomparable diagnostics must remain distinct without fabricated values.
- Value calibration must compare predictions with realized returns only when environment, reward, discount, action-selection, and episode-boundary rules match.
- Human control must preserve the declared action format and stop safely when focus or control is lost.
- Human control must create Counterfactual Playback and must not cause Training Success, Acceptance, or Promotion.
- Benchmark claims must compare Equivalent Providers, policy-facing contracts, workloads, concurrency, and host-load conditions.
