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
