# Experiments

This directory holds active goal contracts, training recipes, benchmark
profiles, report declarations, and experiment utilities.
Keep broad repo rules in the top-level runbooks:

- `../AGENTS.md` for repo rules and stable-retro runtime cautions.
- `../COMPUTE.md` for portable compute policy.
- `~/.config/gradlab/instances.md` for operator-local hardware inventory.

Use `goals/<env-id>/` for durable goal-family contracts and optional provider-specific
environment fragments named `_env-<provider>.yaml`. Goal-family report declarations
live beside those contracts as `_reports.yaml`. Active training recipes live under
`recipes/`, while benchmark profiles live under `benchmarks/`. Generated local run
outputs belong under `~/.config/gradlab/runs/`; other generated logs and models
belong under ignored `logs/` and `models/` paths.

Current research state:

- `goals/`: active goal contracts, optional environment fragments, and report declarations.
- `recipes/`: active checked-in training recipes and presets.
- `benchmarks/`: reproducible benchmark profiles and supporting documentation.
- `scripts/`: active experiment utilities used by benchmarks and tooling.
