---
name: gradlab_autoresearch
description: Tune numeric hyperparameters of one checked-in GradLab SB3 PPO/A2C recipe with a bounded training-only fixed-rung search and five untouched confirmation seeds. Broader optimization needs an explicit experiment plan.
---

# Autoresearch

## When to use

Use for numeric hyperparameter tuning of one SB3 PPO/A2C leaf recipe with fixed
architecture, rollout settings, provider, and scientific contract. Requests that
include architecture, rollout size, rewards, categorical choices, or another
backend need an explicit experiment plan for those axes. Do not silently drop
requested axes or launch a broader search under this skill.

## Prepare

Apply the repository specification review once for this task. Read `COMPUTE.md`,
the private local inventory, the selected goal/recipe and composed sources, and
[search-policy.md](references/search-policy.md), which owns the frozen search
contract. [controller.md](references/controller.md) owns command procedures.

A direct invocation authorizes at most 48 reserved local-fleet runs under that
contract. It authorizes no infrastructure changes, runtime builds/deployments,
retries, cancellation, commits/pushes, checkpoint evaluation, or Promotion.

## Act

Use the maintained controller for reservations, ranking, and winner application:

```bash
gradlab autoresearch init --goal <goal-file> --recipe <recipe-file>
gradlab autoresearch reconcile --study <study>
gradlab autoresearch next --study <study>
```

Reconcile before submission and after meaningful status changes. It discovers
existing submissions, records terminal observations, and collects verified
journal evidence without launching or retrying. Execute only the controller's
next action; propose candidates when requested and reserve within observed capacity.
Use `$gradlab_launch-experiment` in observe mode for launches and persistent monitors.

## Recover

`waiting` identifies pending work or retryable reads; repeat reconciliation within
the existing deadline. `blocked` preserves a paused study and its reason; diagnose
before supported recovery. Never hand-edit state or restart a trial ad hoc.
Source/runtime drift, ambiguous submissions, and invalid evidence stop progression.
Preserve confirmation holdout isolation and all barriers in the search contract.

## Finish

Apply only the confirmed numeric delta through prepare/apply/verify to the pointed
leaf recipe. Preserve its normal evaluation, tracking, seeds, budget, and Publication
settings. Generated studies belong under `~/.config/gradlab/runs/autoresearch/`.
Report budget/rungs, ranking, confirmation count, delta, source/runtime, recipe,
journal fence, and service links. Call the result `training-signal-confirmed`;
it confers no Acceptance or Promotion. Leave edits uncommitted unless requested.
