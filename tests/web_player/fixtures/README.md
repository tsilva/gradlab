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
