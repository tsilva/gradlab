# GradLab

GradLab is a public reinforcement-learning research workbench. Researchers are its primary users; compute operators and consumers of published policies support the research lifecycle.

## Research Contracts

**Research Goal**:
A durable contract that defines the result a research effort seeks and whether policies can be evaluated.
_Avoid_: Task, experiment

**Goal Revision**:
An immutable version of a research goal's contract. One goal revision is the current default for new runs, and every run records the exact revision that it uses.
_Avoid_: Goal variant, run configuration

**Goal Variant**:
A normalized goal contract created when an override changes a goal revision's scientific meaning, such as environment behavior, policy-facing behavior, training success, acceptance, ranking, or release rules.
_Avoid_: Goal revision, run configuration

**Supported Capability**:
An environment, provider, backend, or workflow that an authoritative project registry lists and that passes its required contract checks.
_Avoid_: Available capability, detected capability

**Equivalent Providers**:
Providers that preserve the same observations, actions, rewards, discounts, events, start states, and episode boundaries for a policy-facing contract. Provider identity remains part of run provenance.
_Avoid_: Compatible providers, interchangeable providers

**Evaluated Goal**:
A research goal that defines both training-success criteria and stricter evaluation-acceptance criteria. It can accept and promote a policy through an authorized evaluation method.
_Avoid_: Evaluation, evaluated run

**Training-Only Goal**:
A research goal that defines training-success criteria and can produce and publish checkpoints but cannot accept, promote, or release a policy.
_Avoid_: Unevaluated goal, incomplete goal

**Deterministic Search**:
An evaluation method for an evaluated goal that accepts a policy through a declared deterministic success event and resource limit.
_Avoid_: Goal type

## Execution

**Run**:
One logical execution of one resolved research goal, configuration, seed, source, and runtime. A run can finish successfully without producing an accepted policy.
_Avoid_: Attempt, job

**Run Configuration**:
The resolved execution choices for a run that do not change the research goal's scientific meaning, such as trainer, algorithm, hyperparameters, resource limit, seed, or equivalent provider.
_Avoid_: Goal variant, research goal

**Attempt**:
One execution try within a run. A retry creates a new attempt without creating a new run.
_Avoid_: Run, retry run

## Playback

**Faithful Playback**:
Playback that uses the policy and action-selection rules recorded by the run.
_Avoid_: Evaluation reproduction, counterfactual playback

**Evaluation Reproduction**:
Faithful playback that also uses the recorded evaluation contract and seed. Evaluation reproduction can inspect evidence but cannot create acceptance evidence.
_Avoid_: Evaluation, acceptance

**Counterfactual Playback**:
Playback that changes policy rules, action selection, environment behavior, or human input. Its results cannot establish training success, acceptance, or promotion.
_Avoid_: Faithful playback, evaluation reproduction

**Playback Session**:
One trajectory shared by its viewers and isolated from other playback sessions. New content behind a mutable reference does not change its active trajectory without an explicit reload.
_Avoid_: Viewer, playback process

## Results

**Policy**:
An agent's decision rule together with the rules needed to execute it. Its identity comes from those execution rules, while provenance records the trainer and algorithm that produced it.
_Avoid_: Model, checkpoint

**Data-Only Policy Artifact**:
A policy artifact that contains state in a known non-executable format for trusted GradLab code to interpret. External data-only policy artifacts require integrity and compatibility checks before use.
_Avoid_: Executable policy artifact, policy code

**Executable Policy Artifact**:
A policy artifact that contains or invokes supplied executable code. GradLab does not accept executable policy artifacts from external sources.
_Avoid_: Data-only policy artifact

**Checkpoint**:
A saved policy state from a run. A checkpoint can be unevaluated.
_Avoid_: Policy, release

**Training Success**:
The provisional classification that a run meets goal-defined criteria calculated during training. Training success is a cheap proxy used to classify and compare runs; it is not acceptance and does not confirm that the research goal is solved.
_Avoid_: Acceptance, solved goal

**Acceptance**:
The determination, from a separate evaluation on seeds different from the training seeds, that a policy meets an evaluated goal's stricter criteria.
_Avoid_: Completion, publication

**Promotion**:
The selection of an accepted policy as an evaluated goal's official result.
_Avoid_: Acceptance, publication

**Publication**:
The act of making an artifact public without implying acceptance.
_Avoid_: Release, promotion

**Publication Policy**:
A run-owned contract that selects which checkpoints GradLab publishes. It can select checkpoints independently of evaluation status.
_Avoid_: Release policy, evaluation policy

**Policy Lineage**:
The explicit derivation relationship among policies published for one research goal.
_Avoid_: Research goal, release

**Release**:
An immutable public package of a promoted policy with its provenance, evaluation evidence, representative replay, and recorded execution requirements. A release remains usable through those recorded requirements when current GradLab contracts change.
_Avoid_: Publication, checkpoint
