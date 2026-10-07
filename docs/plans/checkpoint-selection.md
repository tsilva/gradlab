# Checkpoint selection ownership

Status: proposed implementation plan for architecture-review candidate 1. The implementation specification is published as [GitHub issue #44](https://github.com/tsilva/gradlab/issues/44), labelled `ready-for-agent`. No production implementation has started.

## Outcome and scope

Give one browser module ownership of Checkpoint selection from user intent through failure or presentation completion. Source navigation, incoming messages, and frame presentation must no longer maintain separate copies of the pending selection or decide independently when it completes.

Preserve current user-visible behavior, navigation, command semantics, and the distinction between browsing and the background Playback Session. This is an in-process refactor: no new dependency, server protocol, automatic retry, loading timeout, cancellation policy, or concurrency policy. Checkpoint evidence assembly, catalog fetching/caching, chart-history ownership, diagnostic reads, and runner architecture remain separate work.

Reviewed on 2026-09-12 against clean `main` at `091403d7724a369035277bcacb629e97ab740c93`. Before implementation, record the actual starting revision and inspect overlapping edits. Work on the current branch unless the user authorizes another arrangement; preserve concurrent work.

Root `SPECS.md`, `docs/specs/playback.md`, `CONTEXT.md`, and ADR-0001 remain authoritative. This plan introduces no new domain concept or stakeholder requirement and requires no changes to those documents.

## Current ownership and friction

Paths below are repository-relative; symbols remain authoritative if lines move.

| Location | Current responsibility | Planned treatment |
| --- | --- | --- |
| `src/gradlab/web_player/sources/browser.js`: `selectCheckpoint`, `navigate`, `renderActiveBreadcrumbs` | Selection command, route/history, `activeCheckpointPendingId`, and app loading callback | Submit selection/navigation intent; render pending identity from the common owner; retain catalog data and rendering |
| `src/gradlab/web_player/app.js`: `beginCheckpointLoad`, `finishCheckpointLoad`, `snapshotCompletesCheckpointLoad` | Command/Checkpoint identity, blocking mask, completion predicate | Replace mutable selection state and lifecycle decisions with the owner's interface; retain DOM updates |
| `app.js`: `openSourceRoute`, `handleMessage` | Synthetic browsing snapshots, background Playback Session, activation and asynchronous failure decisions | Consume one browsing/selection decision; retain general message processing and scientific view state |
| `app.js`: `applySnapshot`, connection callbacks | Clear loading after presentation, command rejection, disconnect, or socket error | Report lifecycle facts to the owner and apply its resulting view |
| `src/gradlab/web_player/playback-transition.js` | Two predicates that leave orchestration in callers | Absorb selection rules and remove this module once all references migrate |
| `src/gradlab/web_player/synchronized-presentation.js` | Exact-frame readiness, preparation, coalescing, generation invalidation | Preserve its independent depth; use its existing presentation path to report completion |

The deletion test favors a complete selection owner: removing it would disperse identity, ordering, background-session selection, and cleanup rules back into callers. Merely moving predicates or introducing an object with setters for the existing fields would not achieve the outcome.

## Proposed design

Add `src/gradlab/web_player/checkpoint-selection.js`, named for the existing Checkpoint concept. Its interface expresses selection lifecycle facts rather than exposing mutable state fields. Use explicit operations for selection, local browsing, received selection-related messages, presentation completion, and connection termination, plus a read-only view of the resulting state. Exact exported names may follow existing module conventions during implementation.

The module owns:

- Pending command and Checkpoint identity, including the selection route and a local generation for asynchronous completion work.
- Whether incoming active snapshots belong to the foreground selection or the background Playback Session while browsing.
- The distinction between waiting for activation and waiting for presentation.
- Selection-specific pending navigation, mask visibility, and terminal cleanup decisions.
- A presentation acknowledgement tied to the selection generation and the actual Playback Session/snapshot being rendered.

Callers retain concrete work: command transport and control checks, URL/history operations, catalog loading, DOM rendering, frame storage/decoding, and scientific view updates. The selection owner decides when that work represents a transition; callers do not reimplement its predicates. Retain existing low-level transport and rendering functions instead of building a generic event bus or adapter framework.

Use a narrow injected command function so the owner can coordinate selection admission with the existing command ID or refusal result. Production uses the current command transport; tests use a recording function with controlled results. Do not inject the entire app state, SourceBrowser, window, or panel runtime.

Expose derived presentation state and bounded, one-time actions for history changes or errors where needed. Rendering the same state twice must not send a second command, append a second history entry, or repeat a selection-specific error. SourceBrowser's catalog route may remain local to its catalog implementation, but it must stop owning a separate pending-selection lifecycle.

Keep application snapshots and frame caches in their existing owners. Move only the decision and retained reference needed to distinguish browsing from background Playback; do not create another independently mutable copy of the entire application state.

### Ordering and identity

| Fact received | Selection behavior to preserve |
| --- | --- |
| Command function refuses selection, including observer control refusal | No new tracked load or committed selection navigation |
| Selection command is sent | Track its identity once and show the existing loading state |
| Successful command acknowledgement | Keep loading; acknowledgement does not prove preparation or presentation |
| Rejected matching command | End its tracked load; preserve current error handling |
| Resolving/loading snapshot | Keep the selected transition pending |
| Unrelated active snapshot while locally browsing | Update the background Playback Session without replacing the selected browse screen |
| Matching active snapshot | Admit the selected Playback Session for presentation and preserve active-breadcrumb/pending-navigation reconciliation; keep the blocking mask until the current presentation completion point |
| Matching presentation completes | End the tracked transition and clear its blocking mask |
| Authoritative preparation or activation error | End the current load and retain the existing distinction between initial failure and failure with a previous runner |
| Disconnect or socket error | Release the blocking load as today; preserve connection feedback |
| Imported trajectory becomes active | Preserve the existing loading-clear path without requiring an executable Policy or Checkpoint selection |

Navigation readiness and the blocking mask need not change at the same milestone: today `renderActiveBreadcrumbs` clears pending navigation before frame presentation clears the mask. Represent both as derived state of the same transition rather than forcing them to be a single boolean.

Capture the local selection generation before starting asynchronous frame presentation. A callback from an earlier selection must not clear a later selection, including a repeated selection of the same Checkpoint. Checkpoint identity alone cannot establish that a presentation acknowledgement is current. Retain epoch/sequence validation at existing presentation seams; do not move frame-ordering implementation into this module.

The server's source-generation counter is private, and error snapshots do not carry the initiating command ID. Local generation checks protect browser-owned asynchronous work; they must not be represented as new server-side correlation guarantees. Preserve authoritative error-snapshot handling and server ordering. Do not silently expand this refactor to add wire fields or discard errors based on an inferred association.

### Preservation constraints

1. **Paused activation.** Loading a Checkpoint must not issue Play or advance Policy inference. The server retains preparation, activation, and pause authority.
2. **Browsing versus replacement.** Local browsing can retain an existing Playback Session. `SourceBrowser.navigate` intentionally avoids `browse_sources` when a runner exists because that server command closes it. Selecting a replacement invokes the existing server path, which requests a pause of the previous runner.
3. **History.** Preserve push versus replace behavior, Back/Forward navigation, direct URLs, chronological Checkpoint navigation, and the existing Run-to-Checkpoint route without adding an intermediate screen.
4. **Presentation.** Preserve exact-frame preparation and coalescing, the existing inspection-mode completion path, and layouts with no required game frame, including RGB hidden and detached windows. Do not require every optional diagnostic image before completing a selection.
5. **Multiple windows.** Each window owns its local selection/rendering state; viewers still share the server Playback Session. A window with no locally initiated selection must continue to accept authoritative session changes.
6. **Resource separation.** Keep prefetch bounded and opportunistic; it must not create a foreground selection or a loading mask. Keep catalog request generations and caches separate from selection identity.
7. **Failure and cancellation.** Characterize existing retry/cancel buttons, failed replacement, lazy SourceBrowser initialization, and route changes during loading before migration. Preserve observed routes and error presentation rather than inventing rollback or automatic retry behavior.

In that characterization, distinguish preparation Retry (`retry_source`) from catalog Retry, and distinguish `SourceBrowser.stop()` request cleanup from `cancel_source`. Retry/cancel currently bypass `beginCheckpointLoad`; adjacent navigation guards pending selection, while `selectCheckpoint` itself has no equivalent guard. Check whether the full-screen mask makes cancel controls inaccessible and whether a command failure leaves pending navigation after the mask clears. These are source-derived risks, not confirmed defects. Document any reproduced defect separately from the compatibility baseline; do not silently fix it, canonize it as a product requirement, or widen server selection concurrency in this refactor.

## Implementation sequence

### 1. Establish an executable selection contract

Inventory every reader/writer of `checkpointLoad`, `backgroundPlaybackSnapshot`, `activeCheckpointPendingId`, the two transition predicates, and selection-specific source-mode decisions. Record the current entry paths for table selection, adjacent navigation, direct routes, retry, cancel, and observer windows.

Add a credential-free complete-player selection fixture using the real player entrypoint, `PlaybackWebServer`, and `PlaybackHost`, with a synthetic catalog and controlled loader. Provide two distinguishable synthetic Checkpoints, controllable preparation success/failure, and delayed frame availability. Reuse synthetic recordings and session helpers where practical. Keep controls in test-only code and bind loopback on an automatically assigned port.

The existing `chart_player` fixture starts an already-created runner and does not exercise source selection; it is useful for subsequent playback regression checks but cannot prove this transition. Do not claim full-path coverage by replacing the production message handler or presenting a synthetic page that bypasses SourceBrowser.

**Exit:** the production selection path is observable without credentials, external storage, a trained Policy, or real inference; baseline behaviors and any pre-existing defects are recorded.

### 2. Implement the deep selection module

Implement the ownership and ordering described above with direct interface tests. Use deferred promises and controlled command results to force event order; do not use timing sleeps as proof of correctness. Capture all terminal paths and one-time effects.

**Exit:** one interface exercises selection through activation, presentation, rejection, asynchronous failure, and disconnect; callers do not need its internal state representation.

### 3. Bind SourceBrowser and app lifecycle together

Migrate Checkpoint table and adjacent-selection actions to the new owner. Replace the loading callback and independent pending-navigation ID with its derived state. Preserve payload construction, route normalization, history semantics, and prefetch behavior.

Route local browsing, server snapshots, matching command results, socket events, and presentation acknowledgements through the same owner. Keep general snapshot rendering in `app.js` and frame preparation in `SynchronizedPresentation`/panel runtime. Prevent an old lazy import or presentation promise from applying obsolete selection effects.

**Exit:** the complete player uses the owner from user selection through visible completion; no mirrored mutable selection state or independent activation/completion predicate remains.

### 4. Replace implementation-coupled tests and remove obsolete code

Replace `checkpoint-load.test.mjs` source slicing and VM execution with tests importing the production selection module. Replace the selection-specific patched-prototype tests in `source-browser.test.mjs` with mounted integration checks that use the real owner. Keep unrelated catalog-rendering tests.

In `controls.test.mjs`, replace assertions matching the text of `beginCheckpointLoad`, command-result handling, and `showFramesForSequence(...).then(...)` with behavioral mask/ARIA checks. Keep useful static markup/style assertions when they verify an actual rendering contract. Preserve independent `SynchronizedPresentation` tests.

Delete `playback-transition.js`, obsolete app loading functions, and pending-selection fields only after migration and a repository-wide reference check. Do not delete unique behavior coverage merely to reduce the test count.

**Exit:** tests and production callers cross the same selection interface; source-text extraction is absent from the selection tests.

### 5. Verify production integration and document results

Run the matrix below through the selection interface and representative complete-player sequences in the native Codex in-app browser. Record tested revision, exact commands, fixture settings, results, and remaining limitations. Confirm the repository still preserves dependency hardening and unchanged public message formats.

## Verification matrix

| Scenario | Evidence required |
| --- | --- |
| Table or adjacent Checkpoint selection | Exactly one selection command, correct history behavior, one pending identity, consistent mask/ARIA state |
| Acknowledgement before loading completes | Successful acknowledgement does not remove the mask |
| Snapshot arrives before its frame | Matching Checkpoint remains loading until current presentation completes |
| Frame arrives before its snapshot | Existing exact-frame matching succeeds without duplicate completion |
| A prior presentation resolves after a newer selection | It cannot clear the newer selection; include the same Checkpoint selected twice |
| Initial prepare failure and failed replacement | Existing error display, mask release, and prior-session behavior remain visible |
| Immediate command rejection or observer refusal | No stranded mask or unauthorized command; existing navigation outcome preserved |
| Local browsing while a runner exists | Background snapshots do not force return to Playback and no destructive browse command is sent |
| Back/Forward, direct Checkpoint URL, and adjacent navigation | Correct route, no duplicate history entry, no unintended replay of selection |
| Retry/cancel and navigation while preparing | Match characterized server/client behavior; no invented cancellation or rollback policy |
| Session replacement, disconnect, or delayed SourceBrowser import | No obsolete asynchronous completion mutates the current selection view |
| RGB hidden, no applicable frame, inspection, and detached window | Completion follows existing eligible presentation requirements without waiting for unavailable diagnostics |
| Selection initiated in another window | Both viewers follow the authoritative Playback Session; local browsing behavior is preserved |
| Imported Playback | Activates recorded content without executing its Checkpoint attachment |
| Prefetch and chart demand | No foreground selection caused by prefetch; existing chart ownership and data isolation preserved |

Run the focused JavaScript suite, adding the new interface/integration tests:

```bash
node --test tests/web_player/checkpoint-load.test.mjs tests/web_player/source-browser.test.mjs tests/web_player/controls.test.mjs tests/web_player/synchronized-presentation.test.mjs tests/web_player/playback-transport.test.mjs
```

Then run the existing complete JavaScript suite:

```bash
node --test tests/web_player/*.test.mjs
```

Exercise host/transport preservation and any new Python fixture tests with the frozen environment:

```bash
uv run --frozen pytest -q tests/test_play_application.py tests/test_play_web.py
```

Run Ruff on any changed Python fixture files. No dependency update is planned. Browser verification must use the real application on the fixture's printed loopback URL, with visible route, mask, frame, error, and control-state assertions plus test-owned command counts. Do not inspect private application globals to claim end-to-end success. Stop only servers started for this verification.

### Planning baseline

On the reviewed clean main revision, the focused command without `controls.test.mjs` ran 99 tests: **97 passed, 2 failed**. Both failures are existing metric-label expectations in `source-browser.test.mjs`: “run metrics use compact labels and values” and “checkpoint metric headers preserve semantics in one short line.” A separate run of `controls.test.mjs` passed all **23 tests**.

These are pre-implementation baseline results. Record them separately during implementation and do not silently change metric semantics or weaken tests to make this refactor green. No Python suite or complete-player browser selection verification was run during planning; those remain implementation gates.

## Completion criteria

- One module owns pending selection, background-versus-foreground decisions, and transition completion.
- SourceBrowser and app rendering consume its view without independent lifecycle predicates or mirrored pending identity.
- Successful acknowledgement, active snapshot, and presentation completion remain distinct.
- Current route/history behavior, server commands, paused activation, frame synchronization, and shared Playback Session behavior survive.
- Tests import production behavior rather than extracting its source, and the complete-player fixture proves the assembled path.
- The final report distinguishes passing checks, pre-existing failures, and unverified cases; no implementation or scientific behavior change is hidden in a test rewrite.
