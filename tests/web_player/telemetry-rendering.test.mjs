import assert from "node:assert/strict";
import test from "node:test";
import { createTelemetryRenderer } from "../../src/gradlab/web_player/panels/telemetry-panel.js";
import { chartHarness, full, flush } from "./helpers/chart-history.mjs";

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

test("chart status and panel redraws stay synchronized without duplicate history renders", async () => {
  const h = chartHarness();
  const calls = [], statuses = [];
  const renderer = createTelemetryRenderer([{ render: context => calls.push(context) }],
    context => statuses.push(context.view.chartStatus.status));
  const history = [], snapshot = { sequence: 10 };
  const view = () => {
    const chart = h.history.read();
    return { history, chartHistory: chart.data, chartStatus: chart };
  };
  h.history.setDemand(true);
  renderer.render(snapshot, view());
  renderer.renderHistory(history, snapshot, view());
  assert.deepEqual(statuses, ['loading']);
  h.requests[0].resolve(full()); await flush();
  renderer.render(snapshot, view());
  renderer.renderHistory(history, snapshot, view());
  assert.deepEqual(statuses, ['loading', 'ready']);
  assert.equal(calls.length, 2);
  h.update({ liveHistory: [{ step: 11, episode: 1 }], throughStep: 11 });
  renderer.render(snapshot, view());
  assert.deepEqual(calls.at(-1).view.chartHistory.map(point => point.step), [1, 10, 11]);
  h.history.selectRange({ first: 1, last: 5 });
  renderer.renderHistory(history, snapshot, view());
  assert.equal(statuses.at(-1), 'loading');
  assert.equal(calls.at(-1).view.chartHistory, null);
});
