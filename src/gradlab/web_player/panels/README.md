# Player panel contract

The player is a versioned workspace of panel instances. The shell owns the
synchronized session, workspace persistence, GridStack placement, transport,
and bounded trajectory cursor. A panel module owns its DOM and visualization.

`catalog.js` has two registries:

- `PANEL_TYPES` declares lazy module paths, minimum sizes, frame subscriptions,
  and singleton behavior for reusable panel types.
- `BUILTIN_PANEL_PRESETS` instantiates the default research workspace. Policy,
  reward, action, and signal views are presets of the same `telemetry` type;
  game, controls, observation, attribution, CNN feature inspection, events, and
  raw transition inspection remain specialized types.

`workspace.js` owns the v6 persisted shape. Each instance has `type`, `title`,
`config`, `enabled`, `builtin`, and a zero-based GridStack `placement`. Custom
instances are telemetry panels. Only the current schema is read; other versions
are ignored. Visibility controls placement; the standardized Enabled switch
controls processing. A disabled panel stays in its workspace position but is
excluded from browser rendering, frame subscriptions, retained-history demand,
inspection work, and policy-diagnostic demand.

Telemetry configuration is a list of visualization blocks:

- `stats`: current or selected-transition values for multiple metrics.
- `line`: compatible scalar history series sharing one unit.
- `histogram`: retained categorical history.
- `distribution`: the current policy distribution; discrete policies also compare
  retained-episode executed-action frequency.
- `namespace-explorer`: dynamically discovered environment signals or reward
  components.
- `reward-breakdown`: a signed, reconciling reward ledger at the selected step
  or from episode step 1 through the selected cursor. It has a persisted `scope`
  (`step` or `episode`) and no metric selector.

`telemetry.js` is the descriptor registry for the live playback protocol. These
keys are local visualization descriptors, not W&B metric names. A descriptor
defines its label, value type, unit, transition phase, formatting, and accessors
for snapshots and history points.

Playback protocol v8 and newer supply reward-accounting data separately from general
telemetry. `session.reward_accounting` declares availability, the unit-interval
`reward_scale`, and optional `clip_bounds`; each transition supplies
`reward.raw`, `reward.components`, and `reward.accounting_error`, with matching
`reward_raw` and `reward_accounting_error` history fields. Components use the
explicit player wire IDs `native_reward`, `cell_novelty_reward`, `event_reward`,
`progress_reward`, `score_reward`, `completion_reward`, `death_penalty`, and
`time_penalty`. Other task signals are not inferred as reward components.

For each transition the ledger converts a raw component `c` to final-reward
units as `c * reward_scale`, attributes any raw residual separately, and adds
the clip adjustment `final - raw * reward_scale`. Signed contribution is
`100 * impact / abs(final)` and can be negative or exceed 100%; activity share
uses absolute per-transition impact. Episode scope sums those per-transition
quantities and applies clipping per transition. It fails closed when retained
history does not begin at episode step 1 or any point is malformed, while step
scope remains usable. Recorded dataset and human-recording sessions explicitly
report accounting as unavailable.

The built-in Reward analysis panel is visible in newly created workspaces.
Normalization adds it hidden on the shelf when an existing v6 workspace does
not contain it, preserving the existing layout without a schema-version reset.

The built-in Input panel owns the one canonical model-input viewport and
requests only base-observation processing. Enabling the standalone Attribution
or CNN feature explorer panel changes that viewport with its exact-transition
overlay; the most recently enabled diagnostic wins while both remain active.
Input subscribes to both generated frame kinds without requesting their
processing, so neither overlay is computed merely because Input is open.
Base observations and generated overlays must match the selected transition
sequence, and generated overlays must also match its generation.

The built-in Attribution panel is an opt-in specialized control panel for live
policy attribution. Its standardized Enabled switch controls attribution
processing and whether the resulting overlay changes Input. Its shared
controls select Grad-CAM or occlusion and their capture cadence. It does not own
an input viewport.

The built-in CNN feature explorer is an opt-in specialized panel for a live
policy's actor image encoder. It is disabled by default. Its standardized
Enabled switch controls both panel processing and CNN capture, while its shared
controls select one discovered Conv2d layer, capture cadence, and top-filter
count. Each exact transition ranks filters by peak raw positive post-activation
response. The generation-tagged binary atlas holds a categorical winner map, a
signed per-group-input-channel kernel mosaic, and a per-filter activation map.
The explorer renders the kernel, activation, response, and receptive-field
details; Input renders the spatial winner map over the model input.
Winner colors compare only the displayed filters and describe activation, not
selected-action attribution.

A panel module exports:

```js
export function mount({ definition, services }) {
  return {
    element,
    render(snapshot, view) {},
    renderHistory(history, snapshot, view) {},
    renderFrame(kind, blob) {},
    resize() {},
    destroy() {},
  };
}
```

Only `element` is required. The shared `view` identifies the selected and live
transition with `sessionEpoch`, `selectedSequence`, `liveSequence`, and
`inspection`. History is already filtered to the active episode. Panels render
the supplied selected snapshot; controls intentionally read the live snapshot.

Add a visualization metric by adding a descriptor. Add a reusable visualization
shape by extending telemetry block validation, the editor, and the generic
renderer. Add a specialized panel only when it needs behavior that does not fit
the telemetry contract.
