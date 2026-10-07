# Svelte player verification (issue 54)

Measured 2026-09-22 against baseline
`e275acb2a722a1b0727772afcd360085e0d42fc5`. Both frontends ran against the same
current deterministic Python HTTP/WebSocket fixture. Baseline assets were
extracted with `git archive`; candidate assets were produced by the locked Vite
build at `d49a96ce`. The later overlay-cache restoration repair affects diagnostic
overlay restoration, which is inactive in this workload; it has its own regression
test. No Policy or external service was used.

## Architecture and scope

Svelte owns the shell markup, live transport projections, settings, keyed panel
membership, panel editor/shelf, and all panel contents. Small reactive projections
hold control values; snapshots and history use raw identity-based state. Existing
Playback inspection, synchronized presentation, selection, chart history, workspace,
desktop, discovery, document and publication controllers retain their contracts.
GridStack continues to own geometry through the existing app adapter. Frame
preparation/commit remains an explicit synchronized-presentation operation.

Line charts cache their series raster separately from cursor/hover. Input and CNN
surfaces skip unchanged images and reject obsolete asynchronous decodes. Canvas
buffers only resize when their dimensions change. The old imperative panel mounts,
panel runtime, editor and tooltip builder were removed. Discovery, document and
publication adapters remain, as required by the issue.

## Reproduction and measurement limits

Follow the [fixture guide](../../tests/web_player/fixtures/README.md). Use
`chart_player --recorded-steps 3000`, with the paired Player and Stats views open.
The Stats button runs three repetitions of settled idle, 90 hover events, and 12
alternating seeks between steps 1 and 3000. Completion checks the selected metadata
and actual game pixel in Player. Both views report their work. Wait for the JSONL
status to become `complete` before comparing or interacting with another fixture.

Measurements used the native Codex in-app Chromium 153 browser, macOS, eight
reported logical CPUs and device-pixel ratio 2, with the same viewport. The final
comparison ran sequentially, with the repository test process paused and no
concurrent scripted browser scenarios. Unrelated desktop/background activity was
left intact; results include its variability. Earlier exploratory runs that
overlapped other test scenarios are excluded from the timing comparison.

`scheduledUiMs + inputDispatchMs` measures intercepted RAF/microtask callbacks and
synchronous input dispatch, **not total renderer CPU time**. Canvas counts cover
`clearRect`, `drawImage`, `stroke` and `fillText`. DOM allocation counts cover
creation/clone API calls, not all JavaScript allocations. Coarse heap occupancy
was also recorded; GC makes its deltas unsuitable for a memory-savings claim.
This is a reproducible within-host comparison, not a general framework benchmark.
Raw local JSONL is under ignored `logs/issue-54/*-paired-isolated.jsonl`.

## Paired results

Ranges below span the three repetitions and combine both views unless stated.

| Workload / measurement | Baseline | Svelte |
| --- | ---: | ---: |
| Settled idle canvas calls / decodes | 0 / 0 | 0 / 0 |
| 90 hovers: connected canvas calls | 6,279 | 1,092 |
| 90 hovers: series raster calls on detached buffers | 0 (draws directly) | 0 (reuses cache) |
| 90 hovers: DOM mutations | 5,637–5,640 | 1,066–1,067 |
| 90 hovers: DOM allocation API calls | 5,698 | 12 |
| 90 hovers: measured UI work, ms | 475.2–493.5 | 270.5–346.7 |
| Hover input-to-next-refresh median, ms | 8.3–8.4 | 8.3 |
| Hover maximum, ms | 14.0–26.8 | 13.1–27.9 |
| 12 seeks: connected canvas calls | 2,609 | 180–192 |
| 12 seeks: detached series raster calls | 0 | 65 |
| 12 seeks: Input image draws | 22 | 12 |
| 12 seeks: game image draws | 12 | 12 |
| 12 seeks: DOM allocation API calls | 2,734–2,735 | 0 |
| 12 seeks: measured UI work, ms | 143.7–160.4 | 72.1–75.5 |
| Seek-to-selected-frame median, ms | 67.1–83.3 | 62.0–75.0 |
| Seek-to-selected-frame maximum, ms | 165.9–533.6 | 142.7–158.6 |
| 12 seeks: image decodes / peak concurrent decodes | 24 / 2 | 24 / 2 |

Hover never redraws game or Input images and never decodes a bitmap. No decodes
remain pending after a workload. A bounded chart refresh may remain in flight
immediately after seeking; every subsequent settled-idle sample has zero pending
reads/decodes. Counts do not grow across repetitions. Latency remains within
observed baseline variability. Heap deltas vary with GC (including negative hover
deltas and a positive candidate seek delta up to 6.8 MB); no allocation or retained
heap reduction beyond the counted DOM API calls is claimed.

## Source and distribution size

Source counts include authored JS, TS, Svelte and HTML under the frontend/player
roots, excluding vendor and generated bundles. Formatting is included, so line
count is not an algorithmic-complexity measure.

| Size | Baseline | Svelte |
| --- | ---: | ---: |
| Application source bytes | 580,450 | 581,620 |
| Application source lines | 15,077 | 16,150 |
| Shipped assets, including fonts/vendor/build manifest | 1,834,431 B | 1,644,103 B |
| Application JavaScript shipped, excluding vendor | 558,504 B | 383,273 B |
| Sum of individually gzipped application JS files | 138,258 B | 119,598 B |

Authored source size is essentially unchanged; this migration does not establish a
source-size reduction. Its demonstrated benefits are less redundant display work,
declarative panel ownership, and a smaller shipped application bundle.

## Behavioral and packaging verification

- Initial controls/game/reward-table slice passed the five original complete-player
  inspection checks before the remaining panels migrated.
- Final inspection fixture: six checks pass, covering settings command arrival,
  seed preservation, processing toggles/editor, paused Policy activation, rapid
  seeks, held reads/decodes, independent reward reference and replacement readiness.
- Selection fixture: eight checks pass (catalog recovery, observer refusal,
  rejection, keyboard selection/failure, Retry, delayed frames, local discovery,
  failed replacement). History/window suite: six checks pass, including browser
  history, RGB-hidden replacement, cancellation and observational imports.
- Compiled mounted chart fixture passes shared loading/refresh/error/Retry,
  obsolete-data hiding, range behavior, synchronized cursor/tooltips, recorded
  tooltip values, bounds and pointer leave.
- Held peer delivery retains step 203 while the initiating view moves to 201;
  releasing two observed messages settles the peer at 201 with no pending work.
- Native accessibility fullscreen entry, seek to 2999 and exit preserve the paired
  workspace; Stats follows the same step. Focused delivery tests retain fullscreen
  suspension and disabled-panel gating.
- Pure controller/calculation tests remain. Obsolete assertions about imperative
  source construction were removed or adapted to the compiled browser boundary.
- `pnpm check:web`, production/fixture builds, `uv run ruff check .`, and
  `uv run gradlab validate` pass.
- Node suite retains the same seven pre-existing source/style expectation failures
  seen on the baseline: two header/navigation assertions, three source-table
  assertions and two theme/typography assertions (337 passed, 7 failed). No new
  Node failure remains.
- `uv build` produces both distributions; a wheel builds from the sdist with Node
  absent from `PATH`. An installed wheel launches the real fixture and displays
  Playback with Node absent. The build hook rejects deliberately altered source
  instead of silently shipping stale assets. Generated bundles remain untracked.
- The full Python run produced 2,101 passes, 3 skips and 4 failures (including
  subtests). Two package-input failures were repaired together; the training-image
  suite then passed all 13 tests and 8 subtests. The remaining schema-version and
  metric-cardinality failures reproduce on the baseline and are unchanged here.
- The real Docker `app-package` target builds successfully, with asset verification
  and wheel installation in its Node-free Python stage. Frontend source/configuration
  now participates in the runtime identity; its mutation test failed before the fix
  and passes afterward. The full GPU image is left to CI.

## Standards

No findings in the initial review or the packaging follow-up.

## Spec

One overlay-cache restoration bug was found. Its exact→missing→exact test failed
before the repair, passes after it, and the reviewer confirmed the repair. No
additional findings in the packaging follow-up.

Review totals: Standards 0; Spec 1 fixed, 0 remaining.
