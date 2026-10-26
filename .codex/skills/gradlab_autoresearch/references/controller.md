# Autoresearch Controller Operations

Run these commands from the repository root. The maintained controller is
`gradlab.autoresearch`; do not create a second study engine in the skill.

## Initialize or resume

```bash
gradlab autoresearch init \
  --goal <goal-file> \
  --recipe <recipe-file> \
  [--target <local-fleet>] \
  [--strong-threshold 0.90]
```

The helper resumes the sole matching current study. A new study resolves the
schema-v3 operator `dstack.default_fleet` (or the explicit `--target` override)
once, then pins that fleet and its coordinator, committed `HEAD`, the resolved tracking mode, all composed
source hashes, the runtime triplet, rung
caps, seeds, and threshold in `~/.config/gradlab/runs/autoresearch/<study>/study.json`. Existing
studies retain their recorded fleet. Never hand-edit the state.

Ask for exactly one deterministic action at a time:

```bash
gradlab autoresearch next --study <study>
```

Obey the returned action. Never infer or skip a rung, evidence read, or barrier.

## Read capacity and reserve

Before every reservation, read dstack without mutating capacity:

```bash
dstack ps --json
```

Read the pinned fleet's effective capacity from the private inventory. Pass it
as `effective_capacity` and count active training tasks targeting that same
fleet as `active_reservations`. Wait when no slot is available.

Baseline screen and pair:

```bash
gradlab autoresearch reserve-wave \
  --study <study> --phase baseline-screen \
  --effective-capacity <capacity> --active-reservations <active>

gradlab autoresearch reserve-wave \
  --study <study> --phase baseline-pair \
  --effective-capacity <capacity> --active-reservations <active>
```

For `propose_search`, compute `available_slots = effective_capacity - active_reservations` and
propose exactly `min(next.max_candidates, available_slots)` new deltas. Change one coherent numeric
group per candidate, then reserve the one-seed screens:

```bash
gradlab autoresearch reserve-wave \
  --study <study> --phase search-screen \
  --candidates-json '[{"delta":{"learning_rate":0.0002}}]' \
  --effective-capacity <capacity> --active-reservations <active>
```

Reserve a passed candidate's two fresh 50% seeds only when requested:

```bash
gradlab autoresearch reserve-wave \
  --study <study> --phase search-pair --candidate-id <candidate-id> \
  --effective-capacity <capacity> --active-reservations <active>
```

Reserve the five full-cap confirmation seeds only when requested:

```bash
gradlab autoresearch reserve-wave \
  --study <study> --phase confirmation \
  --effective-capacity <capacity> --active-reservations <active>
```

Reservations are durable. Search always retains five of the 48 jobs for at least one confirmation.

## Reconcile, launch, and monitor

Run one bounded observation pass before submission and after meaningful status changes:

```bash
gradlab autoresearch reconcile --study <study>
```

This command owns submission discovery, terminal recording, and journal collection.
It reads authoritative private R2 manifests and hash-validated recipe provenance,
uses the existing experiment status path for task/receipt observations, and writes
only local study bookkeeping. It never submits, retries, cancels, repairs, changes
capacity, closes scientific barriers, or applies a winner.

The JSON result has `state`, `next`, and `observations`:

- `ready`: perform the returned controller action. A zero-match reserved wave
  includes generated launch commands; confirm there is no launch still in flight
  and use those commands once within currently observed capacity.
- `waiting`: keep the existing monitors, or repeat a retryable read within the
  current deadline. Read the observation's `action` and `error_type`.
- `blocked`: the study is paused with its reason in `next`; diagnose and use
  supported recovery evidence before advancing.
- `done`: report the controller's result.

Discovery is bounded by `--max-scan-keys` (default 10000 control-object keys).
`ScanLimitReached` cannot establish absence and supplies no launch commands;
repeat with an explicitly chosen sufficient read bound or inspect the inventory.
A read limit changes neither training budget nor compute capacity.

An exact complete cohort recovers even if the launch response was lost. Partial,
duplicate, or mismatched cohorts pause instead of creating replacements. A
manifest without a submitted task remains unresolved; inspect it and use supported
`resume-submit` only with corresponding authorization. Current schema-v6 studies
and exact runtime/source identities are required; do not probe old readers.

Use `$gradlab_launch-experiment` in observe mode for every generated launch and keep one
persistent `gradlab experiment follow --run <id>` process per Run. Report available
selected-service URLs and stay attached through terminal, resource release, and
active-run investigator results. `potential_bug` or `attention_required` pauses
later waves through `record-attention`.

The frozen search contract owns terminal classifications and negative-evidence
censoring. Reconciliation reuses the same gates as the lower-level commands;
it never treats task exit alone as scientific evidence. Online trials require
selected-service delivery; local-only trials require journal retention.

For diagnosis or replaying a saved observation, the lower-level commands remain:
`record-launch --payload-file`, `record-terminal --event-file`, and
`collect-training-evidence --run-id`. Do not hand-assemble their payloads during
normal operation. Transient evidence reads are repeatable; invalid hashes/fences
pause the study. Do not use failed confirmation per-seed values for new proposals.

Persist attention and resume only from supported new evidence:

```bash
gradlab autoresearch record-attention --study <study> --event-file <event.json>
gradlab autoresearch resume --study <study> --reason '<resolved evidence>'
```

## Close barriers and confirm

After every screen has evidence and every passing candidate's paired rung has evidence:

```bash
gradlab autoresearch close-round \
  --study <study> --round <round>
```

Success mode ranks two-seed evidence by strong-seed count, median and worst censored first-strong
step, then worst peak window-100 minimum. Return mode pairs only the best robust screen per round
and ranks paired evidence by worst and median final-10% mean, worst and median final-10% p05, then
median peak. Stop after three stale rounds or when budget cannot support another complete
candidate plus confirmation.

After exactly five full-cap confirmation runs have evidence:

```bash
gradlab autoresearch close-confirmation --study <study>
```

Success mode requires at least four seeds to reach the frozen strong threshold. Return mode
requires at least four seeds to meet the incumbent's frozen worst paired final-10% mean. A failed
holdout is a binary exclusion; do not inspect its per-seed details to choose later candidates. If
no candidate can be confirmed, run the requested `finish-no-winner` action and leave the recipe
unchanged.

## Apply the winner

Patch only the pointed leaf recipe with the winning `train.backend.config` numeric delta. Never
write rung caps, seeds, study identity, `checkpoint_eval_backend=none`, or descriptions into the
recipe.

Create the exact planned postimage under the generated study directory with `apply_patch`, then:

```bash
gradlab autoresearch prepare-apply \
  --study <study> --postimage-file <planned-recipe-file>
```

Apply that exact patch to the recipe with `apply_patch`, then:

```bash
gradlab autoresearch complete-apply --study <study>
```

The helper recomposes the recipe and proves that only the confirmed backend delta changed. A
baseline winner is a valid no-op. Run targeted tests and leave the recipe patch uncommitted.

Report the study, job budget, rung caps, ranking evidence, confirmation count, winning delta,
runtime triplet, recipe path, selected-service links when available, and verification commands. Call the result only
`training-signal-confirmed`. State explicitly that autoresearch ran no checkpoint evaluation,
promoted no checkpoint, and established no goal acceptance.
