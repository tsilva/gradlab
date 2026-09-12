# Chart history browser verification

The fixtures use synthetic recorded transitions and controlled requests. They need
no credentials, environment assets, trained Policy, or external services.

## Mounted panels

From the checkout, run `python3 -m http.server 0 --bind 127.0.0.1`. Open
`/tests/web_player/fixtures/chart-panels.html` on the printed loopback address in
the Codex in-app browser. The page reports PASS or the first failed assertion.
It exercises the real telemetry panel renderer with a fake clock and controlled
chart responses: shared loading, obsolete-plot removal, retained current data,
permanent errors, shared Retry, and cursor/reference independence.

## Complete player

Run `uv run --frozen python -m tests.web_player.fixtures.chart_player` and open the
printed dashboard URL. The normal player starts paused at step 140. Its first
chart request fails with HTTP 503, then automatically recovers. `--chart-delay`
sets response latency in seconds; `--chart-failures` sets how many requests fail.
The terminal logs requests so cancellation and demand can be checked without
reading application internals.

Exercise these sequences:

- Drag a history chart to zoom, then rapidly adjust the timeline range handles.
  Loading must replace old plots and the latest range must win.
- Scrub while paused. The zoom and reward reference must stay fixed.
- Move a chart into a synchronized window and change zoom in either window.
  Both windows must use that range and manage requests independently.
- Disable or hide the last eligible chart in a window. Requests must stop.
  Restore it while paused and verify a refresh.
- Enter game fullscreen. Other panels in that window stop demanding history.
  Exit and verify a refresh at the current cursor. The fixture adds an explicit
  "Exit fullscreen test" button for browser drivers whose Escape stays in the page.
- Play, pause, and replace the episode. Live points must extend recorded history;
  replacement must remove the old plot and reset zoom to the full episode.

The scripted session intentionally omits some Policy diagnostics. Unavailable or
incomparable diagnostics are fixture data, not evidence of a solved Research Goal.

Use `--imported` with the complete-player command to export the synthetic episode
and inspect it through `TrajectoryPlaybackRunner`. It opens paused at step 140
with the same delayed chart reads and recovery controls. The Checkpoint attachment
is the fixture's opaque test data; the imported runner only reads recorded
transitions. Use the exact-transition field to inspect earlier steps, and Replay
to replay the captured prefix. An unfinished archive cannot generate new steps.

## Checkpoint selection

Run `uv run --frozen python -m tests.web_player.fixtures.selection_player` and open
its printed loopback URL in the Codex in-app browser. This sibling fixture starts
with a Run's two synthetic Checkpoints. It uses the production PlaybackHost,
PlaybackWebServer, source navigation, message handler, and frame presentation.
The loader generates three recorded transitions per Checkpoint and leaves the
runner paused. Manifest URLs are identifiers only; the loader never fetches them.

Use **Run selection checks** for catalog Retry, observer refusal, command rejection,
keyboard selection, initial failure and preparation Retry, adjacent navigation,
loading-mask accessibility, delayed frames, local browsing, and failed replacement.
Use **Run history and window checks** for Back/Forward, repeated selection,
inspection, RGB settings, cancellation characterization, and imported recordings.
Each assertion reports PASS or a failing condition in the fixture controls.
Run these suites on a fresh fixture. Imported Playback is the final case.

The fixture's buttons can release or fail preparation, withhold/release frames,
refuse a command, change control ownership, fail/recover catalog reads, delay the
source-browser module, and disconnect the socket. Fixture controls sit above the
production loading mask. The cancellation check activates the real DOM button
programmatically because the production mask covers it for pointer users; it
characterizes the existing command without changing its accessibility policy.
Preparation and frame release are explicit gates. Polling observes mounted DOM or
server status; elapsed time alone never proves a race assertion. Activation cleanup
must finish before an independent replacement case begins, as the production host
can reject new work while its previous worker is still draining.

For manual multi-window verification, open `/panel/game#token=<printed token>` on
the same origin. Select an adjacent Checkpoint there, release preparation from
either window, and verify that both display the authoritative session while only
the initiating window tracks its load. For a direct-link check, start a fresh
fixture and append `/checkpoints/checkpoint-200-bbbbbbbbbbbbbbbb` to its Run path.
Hold the browser module before opening that link, then release it and verify that
one preparation opens the full Environment/Goal/Variant/Run/Checkpoint route.

To replay an immutable frontend baseline, extract its web-player assets with
`git archive <revision> src/gradlab/web_player`, then pass the extracted directory
to `--assets-root`. The fixture and production Python host stay the same, allowing
comparison of the complete frontend behavior. Generated baseline trees belong in
ignored `logs/`, never source control.
