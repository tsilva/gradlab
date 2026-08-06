---
name: launch-experiment
description: Launch, continuously monitor, and when explicitly authorized repair dstack-backed gradlab training, or run operator-requested Modal evaluations of published checkpoints. Use when the user asks to launch, run, start, execute, follow, watch, monitor, diagnose, fix, or harden a gradlab training recipe, run, research goal, or checkpoint evaluation. Keep live manual evaluation separate from training monitoring, defer its W&B projection and promotion until the run-writer lease is available, and use the request's evidence-backed completion gate.
---

# Launch Experiment

Use `gradlab experiment` for dstack-backed runs. dstack owns compute placement,
logs, cancellation, retries for genuine interruption/no-capacity, and resource
release. Private R2 receipts own run, checkpoint, evaluation, promotion, and
terminal semantics. Never infer scientific success from dstack exit status.

Read `SPECS.md` and `COMPUTE.md` before either launch or existing-run monitoring.
Before a new local launch also read `~/.config/gradlab/instances.md`, plus the
selected goal and recipe. If the local inventory is absent, do not invent a
fleet name or capacity.

Before a new launch, run the credential-safe read-only operator gate:

```bash
gradlab experiment operator-preflight --json
```

It must report ready before launch. Never reconstruct credentials by parsing
human-formatted `dstack secret` output, write protected values into the
repository `.env`, or inline them in a command. `launch` repeats this preflight
before runtime readiness or external mutation.

## Safety mode

- **Observe mode (default):** launch and monitor only. Diagnose potential bugs
  read-only. Do not edit, retry, cancel, restart, or mutate infrastructure.
- **Repair mode:** only when the user explicitly asks to fix, harden, or keep
  working until the complete workflow passes. Preserve failed attempts, repair
  root causes, add regression coverage, and launch a fresh attempt. Never weaken
  the goal, episode manifest, acceptance threshold, or cost policy.

Repair mode does not authorize destructive cleanup, credential changes,
commits/pushes, dstack server replacement, unrelated-run cancellation, or
unbounded cloud cost.

## Launch

Resolve exactly one checked-in goal and one launchable recipe from the selected goal's `recipes/` directory
per requested run; reusable defaults may come from
`experiments/recipes/_presets/`. Independent ablations are separate runs unless the user
explicitly asks to combine them. Repeatable `--set KEY=VALUE` overrides are
allowed when each launch row records its complete isolated override list; they
are composed, validated, and hash-bound in the immutable recipe contract.

Default to `--compute auto`, which uses `GRADLAB_LOCAL_FLEET` from the private
operator configuration. Use `--target <fleet>` only for an explicit per-launch
override. Spot requires finite `--max-price` and `--max-cost-usd`. On-demand
additionally requires `--allow-on-demand`. Always use a finite `--max-duration`.

```bash
gradlab experiment launch \
  --goal-file <goal-file> \
  --recipe-file <recipe-file> \
  --seed <seed> \
  --run-description "<specific description>" \
  [--set <key=value> ...] \
  --compute <auto|local|spot|on-demand> \
  [--target <fleet-or-instance>] \
  [--max-price <hourly-price>] \
  [--max-cost-usd <total-bound>] \
  [--allow-on-demand] \
  --max-duration <duration> \
  --json
```

The command requires a clean, pushed source revision, resolves the verified
exact-source immutable training image and Modal deployment, stages the
hash-verified ROM to eval-private R2, creates the run manifest, and then submits
the dstack task. It never falls back to an older runtime.

Immediately report the returned run ID, attempt ID, dstack task, selected
compute/offer and maximum cost, source/image digest, W&B URL, and public R2 run
index.

## Evaluate selected checkpoints

Treat a request to evaluate already-published checkpoints as an evaluation task,
not an implicit request to monitor training. Do not start the training monitor
solely because the selected run is active.

1. Read the public checkpoint catalog and freeze the exact requested checkpoint
   IDs with its selection fence. Never substitute checkpoints published later.
2. Admit the selection once through the player checkpoint-evaluation endpoint or
   the equivalent `ManualEvaluationQueue` path. Preserve one durable intent and
   one idempotent Modal dispatch per checkpoint.
3. Dispatch and verify the Modal evaluations immediately on their separately
   scheduled compute, even while training is active. Inspect durable queue state
   and eval-private raw plus verified evidence when diagnosing stalled work.
4. Preserve the single-writer contract: while training holds the run-writer
   lease, defer only W&B projection, promotion, and the manual-evaluation terminal
   receipt. Do not defer Modal dispatch, result verification, or checkpoint-list
   visibility.
5. When the active run's immutable source uses an older current contract than the
   checkout, execute through that exact-source runtime. Backport only an
   evidence-proven operational repair; never weaken current-contract readers with
   legacy probing or fallback parsing.

For a request whose completion gate is the checkpoint list, finish only after
every selected checkpoint has raw and verified evaluation evidence and its
evaluation result renders in the actual list. `awaiting_projection` is expected
while an active training run owns W&B; keep the durable worker queued to reconcile
later, but do not turn that deferred projection into training monitoring.

## Monitor

For a training launch, follow, or monitoring request, start one yielded
long-lived monitor per run and retain its session handle:

```bash
gradlab experiment follow --run <run-id>
```

Resume the same process with empty polls. Do not send a newline or interrupt it.
Each JSON line is a combined snapshot of dstack state and authoritative R2
semantic state. Send compact progress updates at most every two minutes unless
state requires attention.

The monitor is complete only when `semantic.terminal` exists. A successful
dstack exit without that receipt is an operational failure, not a scientific
success. Keep monitoring while any checkpoint evaluation or W&B drain remains
pending.

Use these read-only commands only when the follow process fails or a snapshot
shows an operational anomaly:

```bash
gradlab experiment status --run <run-id> --json
gradlab experiment logs --run <run-id> --tail 200
```

If an active run reveals a potential bug, dispatch the project
`training_run_investigator` with the repo path, run ID, latest combined
snapshot, and relevant log excerpt. Keep monitoring while it investigates.
Reuse the same investigator for the same fingerprint.

## Retry and repair

Never blindly retry. A manual retry is permitted only after:

1. the previous dstack attempt is terminal;
2. its R2 writer lease expired;
3. the CLI-enforced 30-second quiescence interval; and
4. the failure is supported as resumable.

```bash
gradlab experiment retry --run <run-id>
```

The logical run ID remains stable and a new attempt ID is created. If a final
checkpoint already exists or acceptance was recorded, retry enters drain-only
recovery and never retrains.

In repair mode, wait for the failed attempt to become terminal and for its
investigator to return. Reproduce narrowly, patch the root cause, add a
deterministic regression, run the affected tests, publish the exact-source
runtime, then retry or launch fresh as the durable state requires.

## Training completion

Launching, seeing W&B, observing an accepted eval, or seeing dstack exit is not
completion. A successful accepted run requires all of:

- authoritative terminal receipt with accepted stop reason;
- accepted 100/100 evidence for the immutable episode manifest;
- eval-driven cooperative stop at a safe learner boundary;
- every frozen periodic/final checkpoint publicly downloadable and terminally
  evaluated;
- the lowest-step accepted checkpoint promoted exactly once;
- W&B through its recorded high-water mark, written by the supervisor only;
- dstack successful and the host released;
- credential-free `gradlab play --run <run-id>` from the public index.

Report the run and attempt IDs, dstack task/compute, source and image digest,
terminal stop reason and step, promoted checkpoint/evidence counts, W&B URL,
public index/checkpoint URLs, drain result, idle-GPU tail, and exact play
command. Do not expose credentials or presigned private URLs.
