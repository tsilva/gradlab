# Player panel contract

The player shell owns the synchronized session, workspace layout, panel placement,
and transport connection. It does not own panel markup or visualization logic.

`catalog.js` is the single registration point for a panel. Each entry declares
its label, lazy module path, default and minimum layout, frame subscriptions, and
frame kinds. The runtime imports a module only while that panel is placed in the
current window.

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

Only `element` is required. A module owns its DOM, event listeners, local view
state, and cleanup. `services` provides the narrow shared capabilities
`getState`, `send`, `command`, and `showToast`.

The shared `view` identifies the selected and live transition with
`sessionEpoch`, `selectedSequence`, `liveSequence`, and `inspection`. History is
already filtered to the active episode. Panels must render the supplied selected
snapshot while the controls panel intentionally reads the live snapshot.

To add a panel, add one module and one catalog entry. Do not add its markup,
subscriptions, rendering, or controls to `index.html` or `app.js`.
