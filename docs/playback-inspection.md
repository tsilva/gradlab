# Playback Session inspection

`createPlaybackInspection` in `src/gradlab/web_player/playback-inspection.js`
owns each window's inspection cursor, selected and live presentation, recorded
reads, replay clock, retained snapshots and frames, and peer inspection messages.
`app.js` admits foreground snapshots after `CheckpointSelection.receive` and
renders the read-only view. It does not write inspection state.

## Inputs and effects

| Input | Payload and meaning |
| --- | --- |
| `admitSnapshot(snapshot, ticket)` | An admitted protocol snapshot with `session_epoch`, `sequence`, `transition`, `session`, `trajectory`, and `control`. Keep its original object identity and Checkpoint presentation ticket. Source discovery/background admission stays outside inspection. |
| `reset(epoch)` | Replace the Playback Session, invalidating reads, preparation, replay, frames and episode history. A changed numeric episode or recorded episode ID in an admitted snapshot also resets episode work, even within one epoch. |
| `receiveHistory` | `{session_epoch, points, timeline}` from the history transport. Normalize and retain bounded points and the episode event overview. |
| `receiveFrame` | `{epoch, episodeId?, sequence, kind, generation, blob}` parsed from a frame transport. The RLP3 socket envelope supplies epoch, sequence, kind and generation; its sequence identifies the transition within the session. |
| `receivePeer` | Existing `inspection-cursor`, `inspection-frame-request`, and targeted `inspection-frame` messages. Validate session, episode where supplied, source, target and diagnostic identity before acting. Received cursors never rebroadcast or infer another Pause. |
| `updateConnection` | `{connected?, hasControl?, historyLimit?}` from the application connection and authority adapters. Retention honors the server limit. Disconnect/control loss preserves existing local replay behavior; the UI and command adapter enforce authority. |
| `setFrameDemand` | `{kinds?, rgbEnabled?}` from panel visibility and authoritative RGB settings. Changes frame eligibility and requests while preserving the cursor. |
| `commandResult` | `{id, ok}` clears only the matching rejected inspection Pause. Explicit Pause always sends the final command, even with earlier acknowledgements pending. |

User intents are `selectStep`, `selectSequence`, `play`, `pause`, and
`returnToLive`. `view` and `subscribe(listener)` expose selected/live snapshots,
selected sequence, pending step, replay/running state, normalized history, event
points, RGB state, session epoch, and the full navigable episode range. Snapshots
and view data are immutable. Neither caches nor cancellation flags are public.
`dispose()` cancels owned timers, invalidates work and suppresses late outputs.

Adapters perform authenticated recorded reads, command transmission, peer/frame
requests, frame preparation/display, error display and time. The recorded-read
adapter returns `{snapshot, points, frames}`; frames contain `{kind, generation,
png}` with base64 PNG bytes. Inspection reuses the production `RecordedStepReader`
and `RecordedStepPrefetch`, including serialized/coalesced demand, four-entry
lookahead and the 8 MiB serialized UTF-8 budget. Reader invalidation suppresses
results without requiring ordinary demand requests to abort. Prefetch owns its
abort controllers, and correctness also holds when cancellation is ignored.

Frame adapters receive `{sequence, generation, isCurrent}`. The existing game,
Input and CNN renderers retain their decode guards and check `isCurrent()` after
asynchronous bitmap decoding. Frame preparation for live updates does not replace
a pending inspection decode. Missing selected frames retain the current display
while a coalesced request asks for the exact target.

Checkpoint presentation completion uses the original ticket and original
snapshot object after required preparation and display. Returning to live may
recover an evicted recorded presentation, but keeps current execution settings.
Chart history, zoom range, reward reference, diagnostics, Policy execution,
Acceptance and Promotion retain their existing owners.
