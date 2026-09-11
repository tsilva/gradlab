# Player trajectory capture measurements

Measured 2026-09-09 with the same scripted 600-decision episode in fresh processes
with capture disabled and enabled, at 60 decisions per second. These are local
CPU/Player measurements, not training throughput or claims about provider
performance. Run from a development checkout:

```bash
uv run --frozen python scripts/benchmark_player_trajectory.py --steps 600 --fps 60
```

The standard workload has a `(1, 4, 84, 84)` uint8 Policy input and a 240×256 RGB
image. The larger workload has a `(1, 32, 160, 160)` float32 input and a 480×640
RGB image. Both retain four display planes and ordinary transition facts. The
arrays are constant, so compressed archive sizes and export times do not predict
those of high-entropy game imagery. Capture writes uncompressed exact bytes.

| Workload | Capture | Step p50 / p99 ms | Start interval p50 / p99 ms | Intervals >25 ms | Maximum pending MiB | Peak RSS growth MiB |
| --- | --- | --- | --- | --- | --- | --- |
| Standard | off | 0.93 / 5.55 | 16.66 / 19.58 | 2 / 599 | 0 | 9.30 |
| Standard | on | 1.47 / 8.38 | 16.66 / 21.26 | 3 / 599 | 0.87 | 16.22 |
| Larger | off | 4.58 / 18.45 | 16.68 / 23.18 | 4 / 599 | 0 | 18.42 |
| Larger | on | 7.65 / 25.58 | 16.66 / 26.13 | 8 / 599 | 48.24 | 88.16 |

Capture completed every decision without a storage pause. Standard input added
about 0.54 ms to median step work; the larger input added about 3.08 ms. The
median start interval stayed at the requested 16.67 ms. Larger input increased
tail latency and transient memory; disk storage does not remove that cost.
Recorded standard/larger prefix preparation took 2.91/2.10 ms; Parquet and ZIP
finalization took 2.03/6.93 seconds in a separate thread. Prepared files were
13.66/13.84 MiB with these highly compressible fixtures.

Shorter 180-decision checks at 120 decisions per second also retained a median
start interval near 8.33 ms for both workloads, with no intervals exceeding
12.5 ms. These checks used the same dimensions and constant content. Workstation
load was not controlled, so small tail differences are descriptive, not a
statistically established regression bound. The script emits RSS samples every
30 decisions; existing live histories also grow until their retention limit.

The selected UI starts capture automatically and exposes Download episode.
The measurements support that choice for these workloads; they do not establish
smoothness for every provider, observation size, disk, or unlimited-speed run.
The bounded queue and explicit pause on storage failure/backpressure remain
necessary. The integration suite separately forces a slow/failing sink and checks
lossless continuation, and verifies an episode longer than the live history cap.

## Recording hot-path optimization, 2026-09-10

A real cached Breakout PPO checkpoint exposed work absent from the scripted
benchmark: repeated transition projections and a complete inspection snapshot
serialized at each decision. Recording now shares transition projections with
live history and publication, caches only bounded metadata field-name
classification, and handles native JSON scalar leaves without NumPy dispatch.
The archive format, exact array ownership, recording defaults, storage bounds,
and first-step inspection are unchanged.

The differential probe alternated three pairs of 200-decision blocks in one
process, resetting the same checkpoint and seed before every block. It used
the real native Breakout environment, CPU PolicyRuntime, recording queue, and
frame encoder, with four 84×84 image planes and eight scalar context fields.
Recording stayed enabled and attribution/CNN inspection stayed off. Baseline
blocks used the recording, snapshot, publication, and encoding functions from
`82b95af8`; optimized blocks used the working implementation, including concurrent
episode reward-summary support. This was a function-level comparison, not a
complete checkout comparison. No browser or other benchmark ran during these
blocks, though background workstation load was not controlled.

| Pair | Before step p50 / p99 ms | After step p50 / p99 ms |
| --- | --- | --- |
| 1 | 6.84 / 9.37 | 4.49 / 5.37 |
| 2 | 6.73 / 7.54 | 4.49 / 5.53 |
| 3 | 6.98 / 9.02 | 4.56 / 5.34 |

Median full-step work fell about 34%. Separate 1,200-decision stage measurements
put median recording work at 3.14 ms before and 1.26 ms after; the latter excludes
the shared transition projection now built once before recording. Background
CPU contention still produced occasional long pauses in those separate runs,
so these results do not promise elimination of all browser or scheduler stalls.

The projection regression test covers both full and filtered live views and
checks unchanged recorded diagnostics. Codec tests cover private-field removal,
mutable metadata, scalar types, non-finite values, noncontiguous arrays, and
byte order. The existing trajectory suite covers export/import, writer failures,
storage limits, and inspection beyond the live-history retention window.

## Playback deadline correction, 2026-09-11

Live publication previously restarted its 30 FPS interval from the completion
time of each published decision. Replay waited a complete interval after each
recorded-step read. Both accumulated work or timer delay rather than preserving
the intended display schedule.

The live publisher now advances its previous deadline and discards missed
display slots. Explicit command publications rebase the schedule, including
resume and FPS changes. Replay uses a monotonic deadline, accounts for read
work, and discards timing debt after stalls. It still visits recorded steps in
order, invalidates pending results when stopped, and keeps at most one replay
timer. Unlimited playback remains available. These changes do not pace Policy
inference or change recorded transitions.

Deterministic tests exercise the production publisher and replay transport:

```bash
uv run --frozen pytest -q tests/test_playback_pacing.py
node --test tests/web_player/playback-transport.test.mjs
```

| Controlled workload | Before | After |
| --- | --- | --- |
| Live, 25 ms decisions with repeating 0–0.4 ms jitter, 10 seconds | 200 publications | 299–301 publications |
| Replay, 300 steps with 16.667 ms reads and repeating 0–2 ms timer delays | 15.300 seconds | 10.017–10.020 seconds |

Additional tests cover 5/10/20 ms producers, slower-than-target producers,
five-second timer stalls, slow reads, pause/resume, FPS changes, unlimited mode,
terminal frames, and cancellation during an outstanding read.

Browser replay measurements also exposed recorded-step requests waiting behind
Policy decisions. Recording identity and read pins now use a separate short
lock; loading the pinned records and encoding their PNGs happens outside it.
The application host releases its lock during the read and rechecks the Session
epoch afterward. The isolated worker dispatches recorded-step reads through its
bounded background-read queue, preserving control and frame-pump responsiveness.
Tests hold an encoder or Policy decision indefinitely and check that the other
operation can proceed, and replace an episode during encoding to verify stale
results are rejected and pinned files are released.
Adjacent history pages reuse their 64 overlapping points, loading only the 64
new records while keeping the cache capped at 128 points.

Replay prefetches up to four upcoming steps through one sequential background
request at a time. Cached responses are limited to 8 MiB of serialized UTF-8
JSON, in addition to the current in-flight response; JavaScript object overhead
is not included in that byte count. Oversized or failed speculative reads fall
back to ordinary demand reads. Pause, seek, return to live, and Session
replacement clear the buffer and abort outstanding prefetch. Prefetch never
advances the cursor or Policy, and all displayed frames keep their exact step
and diagnostic-generation checks.
Telemetry panels receive matching history in their first snapshot render and
skip a queued history render when its snapshot, history, and view are unchanged.
During replay, background inference no longer schedules a duplicate history
redraw for the same recorded cursor. Recorded chart responses still trigger
redraws when new data arrives, and resizing always redraws the panels.

The native Codex in-app browser then exercised the actual server, recording,
HTTP/WebSocket transport, and default player panels for 30 seconds per mode.
The synthetic session used 25 ms Policy decisions, changing 240×256 RGB frames,
and four 84×84 uint8 input planes. Recording was enabled, with 1,500 steps
prefilled before replay. Inference continued while replay inspected older steps.
The tab stayed visible, and no test suite ran during either measurement.

| Mode | Changed canvas FPS | Frame interval p50 / p99 ms |
| --- | --- | --- |
| Live | 29.57 | 30.4 / 67.2 |
| Recorded replay with concurrent inference | 29.50 | 33.3 / 65.1 |

Recorded-step requests took 7.9 ms at p50 and 105.6 ms at p99, measured through
response headers; these request timings exclude response-body parsing. Pause
held the inspected cursor and stopped inference, and the browser reported no
console errors. These are changed-canvas measurements, matching the FPS overlay,
not physical GPU presentation measurements. The workload does not execute the
reported Breakout checkpoint, and workstation background load was uncontrolled;
it demonstrates the corrected pipeline near its target, not a guarantee for
every model or elimination of occasional stalls. Earlier browser samples from
an incorrectly guarded subprocess fixture were discarded.
