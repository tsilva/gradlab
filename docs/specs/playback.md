# Playback

This specification applies to the public checkpoint browser and interactive web player.

## Discovery

- Discovery must provide a searchable Environment → Research Goal → Goal Revision or Goal Variant → Run → public Checkpoint flow.
- Discovery routes must preserve browser history at each hierarchy level.
- Discovery must use a rebuildable precomputed catalog and must not perform synchronous storage scans.
- Discovery must show only available Acceptance evidence and must resolve CLI references progressively.
- Goal selection must distinguish the current Goal Revision, current Goal Variants, and historical Goal Revisions in plain language.
- Each Goal Variant entry must show its normalized scientific difference, first-used date, last-activity date, and Run count.
- Run selection must distinguish checked-in recipes from launch-time overrides without requiring a new recipe.
- Playback must expose the resolved Research Goal and Run Configuration as YAML.
- Goal Variants and overridden Run Configurations must show their proven differences from their authoritative bases.

## Evidence and Inspection

- Checkpoint lists must show the exact Goal Variant's Acceptance and ranking measures with applicable Training Success proxies.
- Training Success proxies must remain visibly distinct from authoritative checkpoint-evaluation evidence.
- Interactive Playback must provide independently arranged and synchronized views of frames, Policy inputs and decisions, transition facts, and bounded histories.
- Inspection must not alter the active trajectory or Policy randomness.
- Playback must expose only actor, critic, action-value, program, attribution, and calibration diagnostics that apply to the selected Policy and recorded contract.
- Unsupported, missing, and scientifically incomparable diagnostics must remain visibly distinct without fabricated values.
- Live-Policy attribution must be disabled by default and must be switchable from the web player without restarting or changing the shared trajectory.
- Value calibration may compare predicted values with realized returns only when environment, reward, discount, action-selection, and episode-boundary rules match.
