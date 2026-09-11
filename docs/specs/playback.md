# Playback

This specification applies to the public checkpoint browser and interactive web player.

## Discovery

- `gradlab play --latest` must open the highest-step published Checkpoint from the newest Run by creation time directly in the player, paused; if that Run has no published Checkpoint, report it without substituting an older Run.
- Discovery must provide a searchable Environment → Research Goal → Goal Revision or Goal Variant → Run → public Checkpoint flow.
- Discovery must preserve browser history for each navigable screen; selecting a Run from a Goal Revision or Goal Variant must open its public Checkpoints directly without an intermediate Run-selection screen or route.
- Discovery must use a rebuildable precomputed catalog and must not perform synchronous storage scans.
- Discovery must show only available Acceptance evidence and must resolve CLI references progressively.
- Goal selection must distinguish the current Goal Revision, current Goal Variants, and historical Goal Revisions in plain language.
- Each Goal Variant entry must show its normalized scientific difference, first-used date, last-activity date, and Run count.
- Run selection must distinguish checked-in recipes from launch-time overrides without requiring a new recipe.
- Playback must expose the resolved Research Goal and Run Configuration as YAML.
- Goal Variants and overridden Run Configurations must show their proven differences from their authoritative bases.
- A selected Run's Checkpoint list must show its authoritative current Run state and make unavailable state explicit.

## Evidence and Inspection

- Launching gradplay or loading a Policy must leave Playback and inference paused until the user presses Play in the player.
- Checkpoint lists must show the exact Goal Variant's Acceptance and ranking measures with applicable Training Success proxies.
- Training Success proxies must remain visibly distinct from authoritative checkpoint-evaluation evidence.
- Interactive Playback must provide independently arranged and synchronized views of frames, Policy inputs and decisions, transition facts, and bounded histories.
- Selecting a step in the reward panel, Events panel, or playbar must move the shared Playback cursor and synchronize the frame and matching row highlights; Events must highlight only an event at the exact cursor step.
- The reference step for discounted reward inspection must remain separate from the Playback cursor and change only through an explicit user action, so users can scrub without changing the calculation reference.
- Users must be able to change the active Policy’s supported action-selection mode at any point, taking effect on the next Policy decision without resetting or otherwise altering the existing trajectory.
- History charts must show the full recorded episode by default. Dragging with the primary mouse button selects a shared step window across history panels; the playbar must show the selected window offsets, and users must be able to reset to the full episode. Seeking must preserve the selected chart window.
- Inspection must not alter the active trajectory or Policy randomness.
- Pause must stop both Playback and Policy inference. Play must advance recorded Playback from the selected cursor and resume unfinished Policy inference from the live head, even when the cursor is behind it; scrubbing while paused must not run either. Episode boundaries, storage limits, execution errors, and human-control safety must still stop execution when required.
- Recorded episodes must remain seekable from their first captured step as they grow, with bounded memory, synchronized recorded frames and diagnostics, and step navigation through the playbar. Reaching the episode storage limit must pause Playback and preserve captured steps until the user replaces the episode.
- Trajectory archives may contain exact Checkpoints as opaque, hash-verified attachments. Imported Playback must never load or execute them.
- Playback must expose only actor, critic, action-value, program, attribution, and calibration diagnostics that apply to the selected Policy and recorded contract.
- Unsupported, missing, and scientifically incomparable diagnostics must remain visibly distinct without fabricated values.
- Live-Policy attribution must be disabled by default and must be switchable from the web player without restarting or changing the shared trajectory.
- Value calibration may compare predicted values with realized returns only when environment, reward, discount, action-selection, and episode-boundary rules match.

## Chart History

- History charts must match the active Playback Session, episode, and selected range. Changing any of these must immediately replace obsolete plots with loading; refreshing the same selection must retain valid data with visible status.
- Episode replacement must reset the chart window to the full episode. Seeking and chart navigation must preserve the independent reward reference and must not advance Playback or Policy inference.
- Recorded history must preserve its annotations and sampling. Eligible live points may extend only its tail, without duplicates or backfilling sampled gaps; the inspection cursor must limit only the live tail.
- Transient chart failures must retry three times after delays of 1, 2, and 4 seconds, including while paused. An unusable revision must receive one full-history recovery attempt without its unusable base; reconstruction failures must not create an unbounded loop.
- Exhausted recovery and permanent failures must leave a persistent error and Retry action inside affected panels, without repeated toasts. Live updates must not restart exhausted recovery; explicit Retry, a changed selection, or returning chart visibility may start a fresh attempt.
- Affected panels in one window must share data and recovery state. Retry in one panel must update every affected panel without duplicate requests.
- Hiding, disabling, or suspending the last eligible chart panel must cancel requests and timers; restoring demand must refresh automatically while paused. Cancelled, replaced, or disposed work must not publish obsolete data or errors or interfere with current work.
- Windows must share the selected zoom range with session and episode isolation while independently managing chart requests and retaining only their current selection's cached history.
