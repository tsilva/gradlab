# Experiments

This directory holds experiment evidence and queue notes. Keep active operational
instructions in the top-level runbooks:

- `../AGENTS.md` for repo rules and stable-retro runtime cautions.
- `../INSTANCES.md` for known GPU targets and benchmark-backed concurrency.
- `../GOAL.md` for the current screening goal.

Use `specs/`, `fleet.json`, and `policies/` for durable queue inputs.
Generated local run logs and outputs belong under ignored `runs/` or `logs/`.

Current machine-readable research state:

- `goals/`: active goal contracts, including metric, runtime, seed, and
  promotion policy.
- `specs/`: checked-in experiment hypotheses and queue payloads.
- `policies/`: capacity and scheduling policies for keeping compute busy without
  mixing incomparable runtime envelopes.
