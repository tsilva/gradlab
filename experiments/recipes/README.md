# Experiment Recipes

Machine-readable recipes define reusable training ingredients: environment
semantics, PPO hyperparameters, reward shaping, and logging defaults.

Goal specs under `experiments/goals/<goal>/specs/` remain the launchable queue
documents. A YAML spec may `extends` one or more recipe files, set goal-specific
fields such as `state`, `seeds`, `run_target`, and W&B naming, then use
`overrides` for the small delta that defines the candidate.

At enqueue time, `rlab train --spec-file <path>.yaml` resolves every `extends`
entry, materializes the final `train_config`, validates the existing expanded
train-spec contract, and stores composition source hashes in `spec_payload_json`.
The queue still executes a fully expanded immutable payload.

Use `profile` only for queue/eval execution lanes such as profile-locked eval
workers. Do not use it as a synonym for hyperparameter recipe; use `recipe`,
`spec`, or `lane` instead.
