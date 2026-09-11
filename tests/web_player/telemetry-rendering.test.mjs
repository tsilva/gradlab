import assert from "node:assert/strict";
import test from "node:test";
import { createTelemetryRenderer } from "../../src/gradlab/web_player/panels/telemetry-panel.js";

test("snapshot and scheduled history presentation render the same panel only once", () => {
  const calls = [];
  const renderer = createTelemetryRenderer([{ render: context => calls.push(context) }]);
  const history = [{ step: 1 }], snapshot = { sequence: 1 };
  const view = { history, chartHistory: history, inspection: true, selectedSequence: 1 };
  renderer.render(snapshot, view);
  assert.equal(calls[0].history, history, "the first render must already have matching history");
  renderer.renderHistory(history, snapshot, { ...view });
  assert.equal(calls.length, 1);
  renderer.resize();
  assert.equal(calls.length, 2, "resizing still redraws the canvases");
  renderer.renderHistory(history, snapshot, { ...view, selectedSequence: 2 });
  assert.equal(calls.length, 3);
  renderer.renderHistory([...history], snapshot, view);
  assert.equal(calls.length, 4);
  renderer.render({ sequence: 2 }, view);
  assert.equal(calls.length, 5);
});
