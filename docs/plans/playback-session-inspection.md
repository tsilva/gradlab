# Playback Session inspection

Status: published as [issue #46](https://github.com/tsilva/gradlab/issues/46) with `ready-for-agent` on 2026-09-12, following the user's request to synthesize and publish this plan. The issue is the implementation handoff; this document retains the source map and planning evidence. No implementation changes made.

## Outcome and scope

One deep Playback Session inspection module owns the selected transition, replay clock, recorded reads, retained presentation data, and the ordering that joins them. Local controls, recorded replay, and synchronized windows use its interface. Tests exercise the production implementation instead of reconstructing cursor changes and read invalidation in test setup.

Keep this a frontend refactor preserving the requirements in [Playback](../specs/playback.md). Chart history, reward reference, Checkpoint selection, Policy execution, diagnostic computation, and publication keep their existing ownership. Python production code, wire formats, dependencies, and scientific semantics are outside scope. The recorded Playback runner refactor from candidate 2 is separate.

The recommended design includes bounded snapshot/frame retention and frame eligibility inside inspection ownership. Bitmap decoding, canvases, panel mounting, and workspace layout remain in their existing modules. This makes validity local without absorbing the rendering implementation.

## Baseline and source map

Inspected current branch at `ff5dc28b`. Before implementation, record the actual starting revision and check overlapping edits without switching branches or overwriting other work. The existing untracked plans `breakout-dataset-collector.md` and `checkpoint-selection.md` belong to other work.

| Module or test | Current friction or responsibility | Planned treatment |
| --- | --- | --- |
| `src/gradlab/web_player/app.js` | `resetSession`, retained trace functions, `applySnapshot`, cursor/read functions, frame handling, and inspection messages share mutable state | Move inspection decisions and retention behind the new module; retain application routing and browser adapters |
| `src/gradlab/web_player/playback-transport.js` | Replay scheduling requires shared app state and caller-owned cursor/cancellation callbacks | Make replay an internal implementation of inspection; remove the shared-state callback interface |
| `src/gradlab/web_player/episode-timeline.js` | `RecordedStepReader` serializes demand and coalesces selections | Reuse internally; retain standalone behavior tests |
| `src/gradlab/web_player/recorded-step-prefetch.js` | Bounded sequential lookahead | Reuse internally with existing limits and cancellation behavior |
| `src/gradlab/web_player/synchronized-presentation.js` | Orders ready snapshots and frame preparation | Reuse internally; characterize reset during pending preparation |
| `src/gradlab/web_player/panels/runtime.js`, `panels/observation.js`, `panels/game.js` | Frame preparation, decoding, and display | Remain rendering adapters; preserve exact identity and late-decode guards |
| `src/gradlab/web_player/checkpoint-selection.js` | Source routing, loading, and validated presentation tickets | Remains authoritative; preserve ticket registration and completion ordering |
| `tests/web_player/playback-transport.test.mjs` | Test setup imitates selection, return-to-live, and read invalidation | Migrate scenarios to the production inspection interface |
| `tests/web_player/observation.test.mjs` | Some inspection assertions match app source strings | Replace those assertions with module or mounted-browser behavior tests |

Symbols take precedence over line numbers. Relevant starting points in `app.js` are `resetSession:234`, `handleMessage:404`, `handleFrame:699`, `pruneRetainedTrace:833`, `setInspectionCursor:1330`, `inspectStep:1425`, `returnToLive:1465`, and `bindWorkspaceSync:2201`.

## Proposed module and interface

Add `src/gradlab/web_player/playback-inspection.js`, exporting `PlaybackInspection`. It is one logical module and may use private implementation files when needed. Use the existing Playback Session domain term; no domain glossary or specification change is required.

Prefer state ownership over a reducer whose caller executes effects. An external effect runner would still need to coordinate reads, frame preparation, timers, and cancellation, recreating the current interface burden. Pure calculations can remain private implementation details.

The proposed interface has three parts:

- **Inputs.** A closed set of context and transport events: admitted live snapshot, history, decoded frame envelope, session reset, control/connection update, frame-demand/RGB update, command result, and peer inspection message. Document their payloads with JSDoc. Accept domain data, not the whole application state or raw DOM/WebSocket events.
- **Intents.** `seek` takes an explicit step or sequence target; `play`, `pause`, and `returnToLive` express user actions. The module decides imported/live behavior, replay interruption, read invalidation, and peer announcement. Callers do not pass `preserveReplay` or invoke internal invalidation methods.
- **Outputs and lifetime.** `read` and `subscribe` expose a read-only view containing selected/live snapshots, history for inspection, cursor, seeking/replay state, and replay capability. `dispose` invalidates owned work and releases retained references. Do not expose mutable maps or duplicate writable inspection state in `app.js`.

Inject adapters for authenticated recorded-step reads, command/frame-request transmission, panel frame preparation and display, peer messages, and clock operations. The browser implementation and controlled test implementation justify each seam. Adapters execute effects; inspection owns validity and ordering. Keep ordinary UI error presentation in the app, with inspection publishing only errors from current work.

The module owns retained snapshots, frame blobs and their eligibility, inspection history, current episode identity, read/presentation generations, replay scheduling, missing-frame request timers, and pending inspection-pause identity. Retain existing history normalization and event-overview calculations as internal mechanisms where inspection uses them. Chart computation and its independent cache remain in `ChartHistory`.

Keep the server's latest application state distinct from the selected Playback presentation. Source browsing and background snapshot decisions remain in `CheckpointSelection` and the app. Inspection must not turn a suppressed background snapshot into a foreground presentation.

## Ordering contracts

1. **Identity and replacement.** Validate session epoch, recorded episode identity or the existing nonrecorded episode identity, sequence, and diagnostic generation as applicable. Same-epoch episode replacement invalidates old inspection work. A generation change must precede acceptance of replacement presentation work.
2. **Current work wins.** Rapid seeks coalesce through the real reader. Late results, errors, frame preparation, and timer callbacks cannot change a superseding selection. Returning to live, peer cursor selection, RGB changes, replacement, and disposal invalidate the affected work. Preserve the latest seeking indicator when an older request settles.
3. **Cancellation semantics.** `RecordedStepReader` suppresses invalidated demand results but does not abort an ordinary uncached demand fetch. Prefetch does abort its own requests. Preserve this distinction; correctness must also hold when cancellation is ignored.
4. **Two clocks.** Pause stops recorded replay and requests an inference pause, including the final Pause in an unacknowledged Pause/Play/Pause sequence. Play behind the head resumes recorded replay and eligible unfinished Policy inference. Paused scrubbing advances neither. Preserve completed-episode, storage-error, observer, and human-control restrictions.
5. **Imported recordings.** Use existing server seek/play/replay commands. Never start the local live-recording replay clock or execute an imported Checkpoint attachment.
6. **Presentation.** Preserve exact sequence/generation matching and retain displayed frames while missing target frames are requested. Background live-frame preparation must not replace pending inspection decoding. Delayed decoding cannot repaint an obsolete cursor. Panel visibility and RGB changes update frame demand without moving the cursor.
7. **Checkpoint completion.** Retain `CheckpointSelection.presentationFor` ticket registration before asynchronous preparation and `presented` completion after the existing required presentation work. Carry the original ticket and snapshot identity through the rendering adapter. Preserve same-epoch cancellation and background browsing behavior. Command acknowledgement is not presentation readiness.
8. **Peer messages.** Preserve existing message types, targeting, and self-message suppression. Receive peer cursors without rebroadcast or a second inferred pause. Apply existing session/episode checks, select exact diagnostic generations, and keep missing-frame requests coalesced. Peer adapters must not mutate inspection state directly.
9. **Retention and pacing.** Preserve server-provided history limits, selected-transition retention, full recorded range despite cache eviction, four prefetch entries, the 8 MiB serialized UTF-8 budget, sequential lookahead, and demand retry after speculative failure. Preserve absolute replay scheduling, FPS changes, RGB-off/unlimited mode, and discarded timing debt after suspended tabs.
10. **Other owners.** Seeking preserves reward reference and chart range; episode replacement still resets chart range through its existing owner. Diagnostic availability, calibration, Policy randomness, Acceptance, and Promotion semantics stay unchanged.

Disconnect/control-loss behavior must first be characterized through current commands and UI. Do not add automatic reconnect or change observer replay policy as part of the refactor. Disposal on actual teardown must cancel timers and suppress all subsequent outputs.

## Implementation sequence

1. **Characterize the joined behavior.** Extend deterministic tests and a complete-player fixture with explicit gates for recorded reads, frame preparation, and peer delivery. Capture current control-loss, imported, missing-frame, and Checkpoint-completion behavior. Add the reset-during-prepare case below. Done when expected outcomes are grounded in the current implementation and authoritative Playback requirements, with any discrepancy identified explicitly.
2. **Implement the owning module.** Move cursor, replay, read orchestration, retained presentation data, and their reset rules together. Compose the real reader, prefetch, and presentation scheduler behind the proposed interface. Tests inject only external effects and time. Done when those tests need no imitation of production cursor changes or mutable application state.
3. **Integrate the application.** Route local controls, snapshot/frame/history events, command results, session changes, RGB/frame-demand changes, return-to-live restoration, and peer messages through the module. Feed the resulting view into existing panel and chart presentation. Preserve Checkpoint selection admission and presentation tickets. Done when every inspection input has one production owner and the app no longer writes inspection state or orders its invalidation.
4. **Remove superseded orchestration.** Remove obsolete app functions and the external transport callback interface. Keep useful reader, prefetch, scheduler, and renderer tests. Replace source-string inspection assertions only after their behavior is covered. Done when searches find no caller-owned inspection maps, replay timers, `preserveReplay` plumbing, or duplicated cursor validity rules.
5. **Verify complete Playback.** Run the focused and full browser test suites, parser checks on changed JavaScript, and the browser scenarios below. Record starting/tested revisions, commands, outcomes, and any independently established baseline failures in `docs/verification/playback-session-inspection.md`. Done when no new failures or unresolved requirement conflicts remain.

## Verification matrix

Use deferred promises, a controlled clock, and explicit fixture release gates. Elapsed time alone must not prove a race.

| Scenario | Required result |
| --- | --- |
| Seek A, B, C while A is pending | Only C can publish; intermediate demand coalesces; old settlement cannot clear C's loading state |
| Replay read pending, then Pause and Play | Exactly one replay clock; obsolete work cannot move the cursor or schedule another tick |
| Live head advances while inspecting | Selected presentation stays fixed; live context and eligible controls remain current |
| Return-to-live restoration followed by a new seek | Restored head cannot overwrite the new cursor; current live configuration remains authoritative |
| Same-epoch episode replacement or new session during read/decode | Old data, errors, and callbacks cannot publish; new presentation remains usable |
| Reset while presentation preparation is pending; immediately offer new ready snapshot | Old snapshot is suppressed; new work drains without requiring an unrelated later event |
| Correct sequence with stale attribution/CNN generation | Stale frame rejected; missing target frame follows existing retention/request behavior |
| Prefetch demand join, ignored abort, oversized entry, speculative failure | Existing bounds, coalescing, and retry semantics preserved |
| Imported playback, human control, observer, pending pause acknowledgements | Existing command eligibility and mode-specific behavior preserved |
| Peer seek/live/frame messages, self-message, wrong session/episode | Exact synchronization without feedback or obsolete presentation |
| Checkpoint replacement, failed replacement, cancellation, background browsing | Loading and foreground decisions remain owned by Checkpoint selection |
| RGB/visibility change, cache eviction, disposal during pending work | Correct frame demand/restoration; bounded retention; no outputs after disposal |
| Scrub with chart zoom and reward reference set | Both remain independent; episode replacement resets only the appropriate episode-scoped state |

The scheduler reset case is an identified coverage gap, not a reproduced bug. If characterization exposes a failure, record it and compare it to Playback requirements before prescribing a change; do not silently add broader scheduler behavior.

### Commands and browser fixtures

During planning, all 50 tests passed with:

```sh
node --test tests/web_player/playback-transport.test.mjs tests/web_player/recorded-step-prefetch.test.mjs tests/web_player/synchronized-presentation.test.mjs tests/web_player/observation.test.mjs tests/web_player/checkpoint-selection.test.mjs
```

During implementation, add the new inspection tests, preserve relevant episode-timeline tests, then run `npm run test:web`, `node --check` for changed JavaScript, and `git diff --check`. Historical failures in [Checkpoint selection verification](../verification/checkpoint-selection-44.md) are context only; verify failures against the actual starting revision before calling them baseline failures. No dependency installation is required for Node tests.

Use the native Codex in-app browser and the complete-player fixtures documented in [fixtures/README.md](../../tests/web_player/fixtures/README.md). Extend a fixture with focused read/frame/peer gates or add a sibling inspection fixture if those controls would otherwise complicate unrelated suites. Exercise two windows, paused scrubbing, Play/Pause behind the head, delayed frames, episode replacement, imported recordings, chart/reference independence, and Checkpoint loading completion. Check mounted frames and metadata, not merely internal state.

Fixture-only Python changes require focused fixture/host checks and Ruff on changed Python. Reuse existing dev servers and their printed URLs; keep generated browser evidence under ignored `logs/`. No training run or external credentials are needed.

## Tracker handoff

Recommended scope: one implementation change that owns inspection state, replay/read ordering, bounded retention, and frame eligibility, while preserving browser behavior and existing rendering/chart/selection ownership. Internal migration steps may be separate commits; do not ship a second writable owner as an intermediate architecture.

The user requested publication of the current design through to-spec. Issue #46 carries the ownership scope, user stories, testing decisions, and exclusions forward without another interview. Detailed method names can be refined during implementation without changing that scope. Root `SPECS.md`, scoped Playback requirements, `CONTEXT.md`, and ADR-0001 need no changes for this design.
