# Player panel contract

The player is a versioned workspace of panel instances. The shell owns the
synchronized session, workspace persistence, GridStack placement, transport,
and bounded trajectory cursor. A panel module owns its DOM and visualization.

`catalog.js` has two registries:

- `PANEL_TYPES` declares lazy module paths, minimum sizes, frame subscriptions,
  and singleton behavior for reusable panel types.
- `BUILTIN_PANEL_PRESETS` instantiates the default research workspace. Policy,
  reward, action, and signal views are presets of the same `telemetry` type;
  game, controls, observation, events, and raw transition inspection remain
  specialized types.

`workspace.js` owns the v5 persisted shape. Each instance has `type`, `title`,
`config`, `builtin`, and a zero-based GridStack `placement`. Custom instances
are telemetry panels. Only the current schema is read; other versions are
ignored.

Telemetry configuration is a list of visualization blocks:

- `stats`: current or selected-transition values for multiple metrics.
- `line`: compatible scalar history series sharing one unit.
- `histogram`: retained categorical history.
- `distribution`: the current policy distribution.
- `namespace-explorer`: dynamically discovered environment signals or reward
  components.

`telemetry.js` is the descriptor registry for the live playback protocol. These
keys are local visualization descriptors, not W&B metric names. A descriptor
defines its label, value type, unit, transition phase, formatting, and accessors
for snapshots and history points.

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
