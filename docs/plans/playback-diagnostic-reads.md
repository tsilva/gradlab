# Playback Session diagnostic-read refactor

Status: implemented. See [verification and review outcomes](../playback-diagnostic-reads-verification.md). The baseline and implementation sequence below preserve the approved design context.

## Baseline and feasibility

Reviewed on 2026-09-11 against freshly fetched `origin/main` at `1f38990f3c97d22ca7054173284daa01ae276102`. Local `main` remains at `1155bdb2394f379e3eedefcf27e93c0980fd5f62` and has concurrent, uncommitted playback responsiveness changes.

The refactor still fits current remote main. PR #39 has already implemented candidate one, browser chart-history ownership. It changed no Python production modules. Candidate two remains a separate Python refactor and must preserve the new browser consumer and the authoritative Chart History requirements in `docs/specs/playback.md`.

Before implementation, compare the actual starting tree with these revisions and the local changes described below. Work on the current branch, preserve other work, and update this plan's source references when code moves. Fetching main for this review did not merge, switch, stash, or reset anything.

## Approved outcome

One deep diagnostic-read module coordinates chart, reward, and event history for live and imported Playback. Its interface hides admission, read reservations, isolated computation, validity checks, and reservation release. Callers and tests cross the same seam.

Recording production remains in the runner. Recording adapters retain the authority to delete files after every diagnostic, inspection, and archive-export reservation has been released. The new module releases only the reservations it owns.

Preserve browser request and response formats, error behavior, calculations, resource limits, process isolation, and shutdown behavior. Existing scientific calculations remain distinct. No new cancellation policy, retry policy, storage system, dependency, or browser-state refactor belongs in this change.

This provides locality for read-lifecycle fixes and leverage across all three diagnostic kinds. Existing revision encoding and incremental indexes already provide depth and remain in place.

## Current source map

Paths below are relative to the repository root. Symbols are authoritative when line numbers change.

| Module | Current responsibility | Planned treatment |
| --- | --- | --- |
| `src/gradlab/play_diagnostics.py` | `DiagnosticRead`, `DiagnosticReads`, `LiveRecordingSource`, `ImportedRecordingSource`, `DirectDiagnosticReader`, `DiagnosticQueries`, `RecordedPrefix` | Own reservation through final validation; adapters supply recording authority without private runner-state access |
| `src/gradlab/play_trajectory.py` | `EpisodeRecording` and `ImportedTrajectory` reservations and retirement | Reuse both real adapters and shared retention accounting; narrowly add reservation support only if required |
| `src/gradlab/play_web.py` | Live runner reads, recording replacement, HTTP handlers, `ChartResponses` ownership | Bind live diagnostic source; translate HTTP requests through one reader seam; retain encoding and response presentation |
| `src/gradlab/play_trajectory_runner.py` | Imported runner diagnostic forwarding and close ordering | Bind imported source to the same lifecycle module |
| `src/gradlab/play_application.py` | `PlaybackHost.read_diagnostics`: active runner selection, phase and session-epoch checks | One session-aware path computes outside the host lock |
| `src/gradlab/playback_worker.py` | Bounded `begin_read`/`poll_read` admission and `IsolatedPlaybackHost.read_diagnostics` | One diagnostic operation; synchronous per-kind history dispatch removed; inspection remains separate |
| `src/gradlab/play_chart_history.py`, `play_reward_history.py`, `play_event_history.py` | Scientific calculation and incremental indexes | Keep semantics and algorithms intact; adapt their private input representation only as necessary |
| `src/gradlab/play_chart_transport.py` | Revisioned `chart-columns-v1` encoding | Keep at the HTTP output seam with its existing server-scoped cache |

## Interface and ownership

Use one internal diagnostic request with an explicit closed kind set: chart, reward, event. Carry episode identity and the existing optional `first` and `last` values. The transport-facing reader also receives the session epoch. Keep session authority in `PlaybackHost`; recording authority belongs to the recording source. Do not make the calculation process own either authority.

The request is not a generic normalized range:

- Chart `first` and `last` describe the existing inclusive chart selection and sampling behavior.
- Reward `first` selects the reward reference; `last` is a forward page cursor. Preserve defaults and discount semantics.
- Event `first` is a lower bound; `last` is the upper cursor for descending pagination. Preserve ordering, page size, and `next_last`.

The source adapter atomically captures the requested recording reservation and the required calibration annotations under its short identity lock. It returns an owned read reservation, a serializable recorded-prefix descriptor, and copied annotations. The reservation provides validity checking and exactly-once release; its callbacks and locks remain in the owning process. Only the descriptor and calculation inputs cross the process seam.

The diagnostic module runs the read through a guaranteed release scope from the first successful reservation onward, including failures during preparation, admission, submission, calculation, and final validation. It must not receive the whole runner or inspect `_diagnostic_lock`, `history`, or `_diagnostics` indirectly through an untyped runner-shaped object. A private calculation input can retain the existing `recording` and `history` shape while callers use the new interface.

Consolidate session selection and before/after validation once in the host path. Direct-runner and worker adapters satisfy the HTTP reader's common interface; route handlers must not repeatedly branch on concrete runner classes to choose argument conventions. Preserve unsupported-source responses explicitly.

Keep the current per-runner diagnostic computation lifetime. A new session can activate before the previous runner finishes draining. HTTP encoding, worker polling, and scientific calculation remain separate implementation responsibilities behind their appropriate seams; one operation does not require flattening them into one class.

## Compatibility and concurrency constraints

1. **Browser behavior.** Preserve the three HTTP routes, authorization, query defaults, payloads, headers, chart compression, and error classification. The new browser retries HTTP 408, 429, and 5xx, but treats 400 as permanent. Do not turn current overload or replacement errors into 429 or 409. Preserve the existing direct-versus-worker cancellation-error behavior; characterize it before changing forwarding.
2. **Chart revisions.** Preserve `chart-columns-v1`, constants, fields, rows, removals, annotations, revision hashes, and full responses when a requested base is unavailable or incompatible. Keep `ChartResponses` outside individual read objects, with the existing eight-version cache shared by viewers of one server.
3. **Episode replacement.** Existing reads retain old files, finish safely, reject outdated results, and release their reservations. Replacement does not introduce forced cancellation of running calculations.
4. **Session replacement.** Select the runner under the host lock, perform work outside it, and validate the session again before exposing the result. Preserve epoch increments and activation-before-old-runner-drain ordering.
5. **Shutdown.** Close admission; preserve `shutdown(wait=True, cancel_futures=True)`, including draining running work and cancellation where futures remain cancellable. Preserve the worker's outer shutdown limits. Do not hold recording or host locks while waiting for calculation completion.
6. **Capacity.** Preserve eight outstanding worker read jobs, including completed results awaiting polling; four worker read threads; four admitted per-runner diagnostic queries; one spawned calculation process per runner; and two cached diagnostic summaries. These limits govern different resources and must not be collapsed into one counter.
7. **Shared retention.** Live recording pins also protect archive exports. Diagnostic release must not delete a recording still retained by export or inspection. Preserve pending records in pinned prefixes and cleanup after the last owner releases.
8. **Scientific isolation.** Diagnostics must not move the cursor, execute imported Policy attachments, alter Policy randomness, fabricate unavailable values, or change calibration comparability.

The review found no conflict with root `SPECS.md`, `CONTEXT.md`, or ADR-0001. These implementation decisions do not require changing those documents.

## Concurrent local work to preserve

The uncommitted tree inspected during planning contains changes absent from the fetched remote main:

- `WebPlaybackRunner` now uses a dedicated short diagnostic lock instead of aliasing the trajectory lock. Replacement acquires trajectory then diagnostic lock; reads avoid waiting behind Policy execution. Preserve this ordering if these changes are in the implementation tree.
- Recorded-step inspection pins a prefix and encodes frames outside the identity lock; the host validates the epoch after inspection outside its lock. Preserve these changes and their tests.
- `inspect_recorded_step` now shares worker `begin_read`/`poll_read` admission with histories and polls at 1 ms, while histories retain 10 ms polling. It remains a separate operation from the approved chart/reward/event request. Keep its shared worker capacity and responsiveness when consolidating history dispatch.
- Inspection imports `RecordedPrefix` from `play_diagnostics.py`. Preserve that use or update both callers together if the helper moves.
- Other local work changes pacing, prefetch, and rendering. Preserve it without expanding this refactor's ownership into those areas.

## Implementation sequence

1. **Establish compatibility tests.** Capture current direct, host, and worker behavior for all three diagnostic kinds and both recording adapters. Assert HTTP status and payload behavior, chart revision recovery, and kind-specific pagination. Inventory every per-kind caller and fixture. Done when the existing contract is executable and the actual starting revision and overlapping edits are recorded.
2. **Implement the owned recording-read lifecycle.** Add the explicit request, source adapter, and reservation lifecycle around the existing calculation executor. Bind live and imported sources. Cover every exit after reservation acquisition, including invalidation and submission failure. Done when diagnostic orchestration no longer reaches into private runner fields and both adapters pass the same lifecycle tests.
3. **Consolidate host and worker coordination.** Replace repeated per-kind host guards and worker dispatch with the single diagnostic operation. Keep short authority checks and asynchronous admission/polling. Preserve any concurrent asynchronous inspection path. Update process-spawn fixtures and mocks with callers. Done when all three history kinds use the shared path and controls still progress while reads are held open.
4. **Bind HTTP adapters.** Keep route names and public handler methods, including the chart fixture's override of `chart_history`. Adapt each route to the shared reader, retain chart encoding after successful reads, and preserve direct and worker error translation. Done when HTTP compatibility tests pass without production JavaScript changes.
5. **Remove obsolete internals and verify.** Delete unused per-kind runner/proxy forwarding and synchronous history dispatch after updating all references. Keep necessary transport adapters and calculation tests. Replace tests coupled to old orchestration internals with interface tests, without discarding unique behavior coverage. Done when the verification matrix below passes and no duplicate history lifecycle remains.

## Verification matrix

Use events, barriers, controlled executors, and real temporary recordings to force ordering. Avoid sleeps as the mechanism proving races. Keep representative tests using real spawned processes; a controlled adapter alone cannot prove process cleanup or control responsiveness.

| Scenario | Required result |
| --- | --- |
| Live and imported chart/reward/event requests | Existing values, annotations, sampling, ordering, pagination, and missing-data behavior |
| Episode replaced during computation | Old files remain readable until release; old result rejected; new episode reads succeed |
| Checkpoint or imported session replaced during computation | New session activates; old result rejected by session identity; old runner drains safely |
| Shutdown with admitted, queued, and running reads | Existing cancellation/error behavior; every acquired reservation and slot released; process exits |
| Preparation, admission, or executor failure | No reservation leak or double release; subsequent valid reads remain possible |
| Diagnostic and export retain the same recording | Releasing either first leaves files for the remaining owner; final release permits cleanup |
| Pending disk writes and recording retirement | Reserved prefix includes exact pending steps and survives retirement |
| Slow diagnostic work | Pause/control requests and Policy inference progress independently |
| Capacity exhaustion and polling | Existing limits and error mapping; completed unpolled jobs continue to count as today |
| Concurrent recorded-step inspection, where present | Existing short-lock behavior, shared read capacity, frame correctness, and polling behavior survive |
| Chart full/delta responses and unavailable base | Existing browser reconstructs the same data and follows its bounded recovery rules |

Run the focused Python suite with the existing frozen environment:

```bash
uv run --frozen pytest -q tests/test_play_chart_history.py tests/test_play_reward_history.py tests/test_play_chart_transport.py tests/test_playback_responsiveness.py tests/test_playback_episode_history.py tests/test_play_trajectory.py tests/test_play_application.py tests/test_play_web.py tests/test_playback_pacing.py
```

Add the new lifecycle tests to that command. Run Ruff on modified Python files and the existing browser tests with `node --test tests/web_player/*.test.mjs`. Preserve dependency hardening and the lockfile; no additional dependencies are planned.

For browser verification, use the native Codex in-app browser and the credential-free fixtures documented on current main in `tests/web_player/fixtures/README.md`. Exercise the complete player with delayed reads, zoom changes, paused inspection, episode replacement, demand cancellation, recovery, and synchronized windows. The existing complete-player fixture covers a live runner; add a fixture mode or companion fixture using an exported synthetic recording to exercise imported Playback. Verify that imported data never executes its attached Checkpoint. Record outcomes and exact commands, distinguishing browser checks from Python tests.

## Completion evidence

The final implementation report must identify the tested revision, any preserved concurrent work, compatibility-test results, race-test results, and live/imported browser results. Record pre-existing failures separately; do not claim the refactor passed an unavailable check. There must be no production browser refactor, altered scientific calculation, capacity change, or cancellation/error normalization hidden in the patch.

Planning verification: remote main was fetched and its complete change set was compared with the original review baseline. The following command passed **64 tests in 102.40 seconds** on the local working tree, including its concurrent responsiveness edits:

```bash
uv run --frozen --no-sync pytest -q tests/test_play_chart_history.py tests/test_play_reward_history.py tests/test_play_chart_transport.py tests/test_playback_responsiveness.py tests/test_playback_episode_history.py tests/test_play_trajectory.py
```

This is a baseline result, not a refactor result or a clean remote-main checkout test. The full implementation verification command and browser checks above remain required. No browser verification was performed during planning. Only this plan was created; the other working-tree changes predate this task.
