# Standalone policy dataset collector

Status: design agreed through the grilling interview. This document is the implementation plan; the collector has not been implemented or launched.

Published specification: [GitHub issue #47](https://github.com/tsilva/gradlab/issues/47), labeled ready-for-agent.

## Purpose and isolation

Collect trajectories from a trained policy for later neural-emulator training. The initial target is the pinned native Breakout Turbo implementation, with interactive RGB, reward, and episode-boundary prediction as the eventual model objective. Neural-model training is outside this feature.

Support Breakout first without needlessly hardcoding game-specific dimensions, frame skip, action formats, or policy preprocessing. Read those from the checkpoint and runtime contract. Supporting arbitrary environments or arbitrary Gymnasium policy adapters is not an initial requirement.

Keep all new executable code, debugging, tests, and usage notes in one isolated directory:

```text
experiments/scripts/policy_dataset/
  collect.py
  test_collect.py
  README.md
```

The script owns collection, persistence, validation, live debugging, and offline inspection. Keep its internal components small and share its dataset reader and renderer between modes. Do not create a plugin framework, general-purpose service, new package CLI command, or Player integration. Do not modify GradLab core or provider code for this prototype. Reuse existing imports rather than copying checkpoint loading or policy execution implementations. No dependency changes are currently needed.

Root SPECS.md remains unchanged: these decisions are scoped to the experimental collector.

## Execution contract

- Accept one local immutable checkpoint bundle per collection session, including model and recipe sidecars. Use GradLab's existing verified, data-only loading path.
- Reconstruct the checkpoint's contracted environment, policy inputs, action mapping, conditional overrides, preprocessing, decision cadence, and episode boundaries by default.
- Capture the initial RGB frame and one successor RGB frame per environment step. The user's intended Breakout policy uses frame skip two, but do not encode two as a collector constant or assume every checkpoint matches it.
- Record configured action repeat and actual elapsed native frames when the provider supplies them. Do not generate, infer, or interpolate intermediate images.
- Preserve the recorded stochastic/deterministic selection mode by default. Temperature one leaves that distribution unchanged; a fidelity mode must not force deterministic actions.
- Use one environment lane initially. Existing capture diagnostics provide terminal RGB before automatic reset for that path.
- Support an explicit full-game collection override that removes training-specific success/life-loss cutoffs, while preserving policy inputs, actions, and cadence. Record original and effective boundary contracts. The pilot will use this override, ending at native game-over or an explicit episode-duration cap.
- Treat dataset resource cutoffs, episode-duration truncation, task termination, and native game-over as distinct facts. Never invent a terminal transition when a recording stops.
- Record checkpoint hashes, provider/runtime versions, effective environment contract, source/script hash, collection settings, policy-selection mode, seed streams, and all overrides. Temperature or boundary interventions are declared collection variants, not acceptance evidence.

Exact pilot checkpoint selection and episode-duration cap are launch parameters to resolve from the chosen checkpoint and collection configuration before running, not reasons to embed current run details in the script.

## RGB and transition data

Updated user intent (2026-09-12): support HUD masking during collection, before image hashing and storage, so HUD-only differences can share stored frame identities. For native Breakout, fill source rows 0–16 with black while preserving the full 210×160 RGB canvas and all pixels from row 17 onward. Record the mask region, fill and capture stage in the dataset contract and reject appends that mix masked and unmasked captures. Apply the transform only to copied captures; preserve policy inputs, actions, rewards, ordered transition occurrences and episode boundaries. Full unmodified RGB remains available when masking is disabled. Validate uint8 HWC RGB and consistent dimensions; do not crop, resize, grayscale or lossily encode stored images.

Copy provider buffers before advancing or queuing writes. Hash a canonical header containing shape, dtype, channel order, and format version plus raw RGB bytes with SHA-256. Verify byte equality on matching digests. Deduplicate exact images globally within the dataset, across episodes and resumed sessions.

Retain every transition occurrence even when both images already exist. RGB equality is not simulator-state equality. Frame stacks and histories can be assembled from ordered frame references without repeating image bytes or crossing episode boundaries.

Store:

- Episode/sequence IDs, policy-decision index, source and successor frame IDs.
- Policy-selected, effective, and native actions where available; action-override facts.
- Effective sampling mode and temperature.
- Native and policy-facing rewards separately.
- Native/task termination, truncation, cutoff, and terminal-image availability.
- Available compact simulator labels, such as ball velocity, remaining lives, and brick configuration, with their schema and missingness.
- Episode/session provenance, checkpoint identity, seeds, boundaries, split membership, and completion status.

Simulator labels are separate supervision/debugging data, not part of image identity or required inputs to the future neural emulator. Omit processed policy-observation tensors and full stacks in version one. Preserve their configuration as metadata, without promising that metadata alone can reconstruct every original policy input.

Capture final RGB before reset and keep the following reset image separate. If required terminal RGB is unavailable, report the failure explicitly and leave an incomplete prefix; never substitute a reset image.

## Temperature exploration

Default collection leaves action selection unchanged. Enable temperature variation explicitly and independently of live/headless mode and boundary overrides.

Agreed initial exploratory schedule:

- Choose a temperature at the beginning of each block of 256 policy decisions.
- Sample temperatures 0.75, 1.0, and 1.25 with probabilities 0.20, 0.60, and 0.20.
- Reset block position at each episode start.
- Use a seeded schedule RNG independent of environment and policy RNGs.
- Record the effective temperature on each transition.
- Make values, probabilities, and block length configurable; validate finite positive temperatures and supported policy capabilities before stepping.

These are provisional pilot settings, not a claim of optimal coverage. Do not add random-action replacement or game-specific phase detection in version one.

## Storage and resume

Use one dataset directory beneath an ignored generated-output location, defaulting under ~/.config/gradlab/runs/. One writer holds an exclusive dataset lock. A dataset owns one compatible effective environment/image/cadence contract; compatible checkpoints can contribute through later collection sessions, with distinct provenance. Reject incompatible appends.

Proposed physical layout:

```text
manifest.json                 immutable dataset contract and format version
index.sqlite                  frame locations, allocated episodes, committed batches
frames/part-*.bin              packed losslessly compressed RGB records
transitions/part-*.parquet     ordered transitions with compact frame IDs
episodes/part-*.parquet        session/episode metadata and completion records
progress.json                 replaceable committed-progress view
```

Use standard-library zlib initially and the existing PyArrow dependency. The frame index maps compact integer IDs and SHA-256 digests to shard offsets, lengths, and decoded dimensions. Avoid one file per frame, unbounded in-memory indexes, and repeated checkpoint/configuration payloads per transition.

Finalize and sync referenced frame bytes and Parquet batches before publishing their index entries and counters in a SQLite transaction. Readers consume committed batches only. Bound queues and apply backpressure rather than dropping transitions. Resource accounting includes indexes, temporary files, and pending batches so configured disk limits are respected.

On interruption, preserve committed data and mark the interrupted episode incomplete. Resume validates the contract, cleans up only uncommitted output under the dataset lock, and allocates a fresh episode without reusing identities or seed assignments. It continues using the existing global frame store. Exact mid-episode simulator/policy-RNG restoration is deferred.

Pilot formats are versioned and disposable: migration is not guaranteed. Preserve old datasets unless the user explicitly requests deletion. Establish a stable format or migration policy before expensive collection.

Partition whole episodes into training and held-out data using existing seed rules. Record assignments durably. Sharing physical RGB records does not authorize using held-out labels or histories during training. Future saved-state branches must remain grouped by their ancestor; branching itself is deferred.

## Progress

Report locally; no W&B integration is part of this prototype.

- Captured frame occurrences, counting each initial frame and successor once rather than counting both references in every transition.
- Exact committed unique captured-RGB count. With HUD masking enabled, HUD-only differences share image identity; otherwise they count as RGB novelty.
- New unique images per second and new-image fraction in a stated recent window.
- Cumulative reuse fraction: 1 - unique_images / captured_frame_occurrences, unavailable at zero occurrences.
- Committed transitions, complete/incomplete episodes, elapsed time, throughput, actual disk use, and projected growth.
- Compression savings separately from duplicate reuse, using explicit baselines.
- Session/checkpoint contributions and available gameplay-stage labels.

Derive authoritative totals from committed records. Distinguish pending data from committed data. Discovery attribution is order-dependent; novelty saturation is not evidence of complete game-state or action coverage.

## Self-contained debugging

Use an optional local pygame window, backed by the installed pygame-ce dependency. Headless collection and visible collection share the same collector and storage code.

Live mode provides pause, resume, and single-step. Show:

- Copied source RGB and RGB decoded from the actual committed frame store, aligned to the same episode and transition.
- Source/recorded IDs, exact pixel-equality result, selected/executed action, reward, boundaries, temperature, and new/reused frame status.
- Progress counters and any persistence lag, displayed outside image pixels.

Read back actual on-disk bytes, not a second in-memory encode/decode. Debug mode may flush per step for immediate inspection. If it displays a committed batch asynchronously, retain only bounded source copies and clearly identify the compared transition. Offline inspection cannot independently recover the original live source frame; its checks are stored-data integrity checks.

Redrawing, pausing, and navigating recordings must not call the policy, consume policy RNG, or advance the environment. Single-step advances exactly one contracted environment step. Pacing affects throughput and wall-time limits, so compare fidelity over equal transition prefixes rather than equal elapsed time.

Provide an offline inspection mode in the same script. It opens committed data, selects an episode, navigates previous/next transitions, plays/pauses recorded frames, and shows metadata. It uses the shared dataset reader/renderer without loading a checkpoint or environment. Add a headless validation mode for integrity and continuity checks.

No browser server, Player protocol adapter, human gameplay controls, or shared UI components are required.

## Existing code to reuse and pitfalls

- policy_bundle.py and trusted_inputs.py/policy_models.py: reconstruct and verify immutable checkpoint contracts.
- env.py: construct the contracted environment with one lane and step diagnostics.
- policy_runtime.py and existing observation preparation: preserve structured policy inputs, execution context, actions, and resets.
- take_step_diagnostics(): capture terminal_frame, provider labels, effective/native actions, rewards, and boundary facts. Ordinary nonterminal SB3 info dictionaries may be empty.
- play_trajectory.encode_tree/decode_tree: existing data-only serialization for structured actions and labels; do not stringify unsupported numerical values.
- seeds.py: preserve the existing seed allocation rules.
- pygame: standalone viewing; the provider's human-play loop is a reference, not a collector runner, because it has its own stepping/reset behavior.

For the pinned native Breakout provider, rendered RGB is available directly. Do not claim broader raw-RGB support merely because an adapter has get_images(); some adapters reconstruct those images from processed policy observations. Validate additional providers when support is actually requested.

## Implementation and validation milestones

1. Implement the isolated single-checkpoint rollout and RGB capture path. Resolve the chosen checkpoint's actual contract; verify action/frame alignment, initial/terminal frames, labels, and explicit full-game override.
2. Implement exact image deduplication, bounded batched storage, lock/commit rules, and fresh-episode resume.
3. Add optional temperature blocks and exact progress counters.
4. Add the script-local live viewer, offline inspector, and headless validator using the same reader and capture path.
5. Run meaningful adjacent tests for exact round trips, duplicate image reuse with preserved transitions, terminal-before-reset handling, structured actions, seeded temperature schedules, incompatible appends, and interruption at commit boundaries.
6. Compare collection against an uninstrumented seeded baseline under identical effective contracts with no temperature intervention. Separately compare visible and headless collection over equal transition prefixes. Check actions, rewards, boundaries, and RGB bytes; do not use fixed wall time for equality.
7. Run a bounded 10 GiB pilot with one chosen checkpoint. Use the full-game override and explicitly selected exploratory schedule after fidelity checks pass. Report integrity results, a reconstructed preview, raw-frame novelty, compression/reuse, index overhead, storage projection, throughput, and bounded memory behavior.

Hardware/concurrency selection requires COMPUTE.md and the operator-local inventory before launch. Pilot outputs remain untracked. No pilot is authorized by this planning document alone.

The first iteration is complete when the isolated collector can collect, debug, stop, resume, inspect, and validate its dataset and the pilot report is available. Training a neural emulator, broad environment support, native-frame trace capture, checkpoint mixtures within a session, saved-state branching, distributed writers, publication, and integration into GradLab remain later work.
