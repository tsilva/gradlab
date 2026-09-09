# Live trajectory restoration

Scrub or enter an exact position to inspect the current episode, including
positions beyond the in-memory display history. A position is the after-state of
that transition, before the next Policy decision. Step zero is the initial state.
Ordinary Play replays the recorded suffix; only **Discard future** or **Resample
from bookmark** restores execution.

**Bookmark position** saves a name, thumbnail, episode and step. Clicking its
thumbnail/name only navigates. Edit the name inline or use Delete. Bookmarks
survive reconnects to the same live session and clear when the episode or
Checkpoint is replaced. They are not a historical library.

**Discard future** restores a captured decision boundary and stays paused.
**Resample from bookmark** restores, cuts, and starts Policy execution. Both keep
bookmarks at the cut and remove later bookmarks. A confirmation appears only
when later bookmarks would be deleted; it is bound to the episode, trajectory
revision and bookmark inventory. A stale confirmation cannot delete new work.

Restoration preserves provider randomness, task/wrapper state, observations and
stacks, episode accounting, and supported Policy memory. It preserves the current
Policy sampling stream and action-selection setting. It never reseeds sampling;
fresh draws may select the same actions, and deterministic execution may produce
the same future. Resampled sessions are visibly Counterfactual Playback and never
provide Training Success, Acceptance or Promotion evidence.

## Support and capture

The authoritative combination registry is `gradlab.play_restoration.LIVE_RESTORATION`.
It covers native Mario and native Breakout with registered PPO, A2C and action-program
implementations using GradLab's single-lane wrapper and strict portable snapshot
codecs. PPO/A2C actor execution is stateless; action programs restore their exact
cursor. State-dependent exploration, cell-graph Policies, other providers, custom
wrappers and archive curricula are disabled with a reason. There is no probing,
approximate reconstruction or reset-to-start fallback.

The capture toggle shows the actual resumable ranges. Disabling capture preserves
existing points; enabling it midway captures subsequent boundaries and exposes
any gaps. A genuine episode boundary cannot continue within that episode; select
an earlier point. Ordinary pause/step/continue stops are restorable when captured.
Storage pressure pauses at a decision boundary; captured data remains available.
Retry recording for a writer failure, or explicitly disable restoration capture
when its capture preparation fails. Neither error silently drops a transition.

Native Breakout capture starts automatically only at a positive rate of at most
60 decisions/s, when its initial encoded restore payload is at most 256 KiB and
capture/encoding takes at most 3 ms. Outside that measured envelope the toggle is
explicit. An automatic admission failure leaves the toggle available. These are
admission checks, not guarantees about every later disk operation; the shared
bounded writer still pauses on failure or exhausted capacity.

## Transactions and exports

Restoration prepares a replacement disk store before changing the live state.
Provider/Policy restoration is verified against the captured state, with verified
rollback on failure. The runner serializes commands and stepping. Mutation requires
current control ownership, session identity and an expected trajectory revision.

A cut creates a new trajectory revision. Session decision sequences never rewind,
so repeated episode step numbers do not reuse transition identities. Original
prefix rows retain their original sequence, revision and classification. New rows
carry the replacement revision and Counterfactual classification. In-flight frame
encoding and queued presentations are invalidated; viewers receive the retained
prefix and replacement timeline.

Downloads already started keep their original append-only data inode, cutoff,
bookmarks and provenance. The replacement store owns a copied prefix, so cleanup
cannot modify those bytes. Copy preparation reads one bounded record at a time;
long prefixes incur proportional disk I/O and restoration latency. Later downloads
contain only the active prefix/continuation, annotations within their cutoff, and
resampling cuts. A cut does not invent termination or truncation.

Version 2 `.gradtraj` metadata includes `trajectory_revision`, `bookmark_revision`,
`bookmarks` and `resampling`. A bookmark has a UUID, name, episode, step and optional
bounded PNG thumbnail encoded as base64. A resampling cut records its revision,
cut step, first new decision sequence, action-selection mode and `sampling_stream:
continued`. Parquet adds `trajectory_revision`; ordering permits sequence gaps
only where the validated resampling provenance declares them. The supported
archive contract is version 2; earlier contracts are rejected explicitly.

Restore payloads are numeric/data-only local episode records, excluded from
Parquet and the export inventory. Imported bookmarks navigate stored data only;
import never builds an environment, loads the opaque Checkpoint, or executes a
Policy. Annotation shape, identities, ranges, thumbnails and provenance are
validated before import commits.

## Capture measurements

Measured 2026-09-09 in fresh processes, using the checked-in native Breakout goal
and recipe environment, an untrained CPU PPO, 300 decisions at 60 Hz, and warmed
runtime paths. No training or Acceptance evaluation ran. Reproduce from a checkout:

```bash
uv run --frozen python scripts/benchmark_player_restoration.py --steps 300
uv run --frozen python scripts/benchmark_player_restoration.py --capture --steps 300
```

| Capture | Step p50/p99 ms | Interval p50/p99 ms | Intervals >25 ms | Restore payload disk | Peak RSS growth |
| --- | --- | --- | --- | --- | --- |
| off | 3.31 / 9.65 | 16.50 / 22.99 | 0 | 0 MiB | 2.95 MiB |
| on | 2.79 / 11.80 | 16.44 / 23.69 | 1 | 62.43 MiB | 4.00 MiB |

Capture/encoding/flush took 0.58/2.91 ms p50/p99. Verified restoration took
1.40/3.24 ms p50/p99 with capture enabled. This component benchmark writes restore
payloads synchronously, with no write backlog; it measures additional restoration
cost rather than full browser/recording throughput. The existing episode-recording
measurements cover its shared bounded background writer. Integration tests inject
storage failure/backpressure and check preserved data and visible pause.
Uncontrolled host load explains why the capture-on median can be lower; this is
not a claim that capture accelerates execution. Mario, larger payloads and higher
rates have not been measured and use the explicit toggle.
