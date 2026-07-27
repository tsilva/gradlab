import assert from "node:assert/strict";
import test from "node:test";

import {
  compatibleMetricKeys,
  decodeDynamicSegment,
  descriptorCatalog,
  descriptorFor,
  descriptorValue,
  dynamicDescriptorKey,
  encodeDynamicSegment,
  metricOptions,
  seriesForMetric,
} from "../../src/gradlab/web_player/panels/telemetry.js";
import { cursorIndex } from "../../src/gradlab/web_player/panels/telemetry-panel.js";
import { lineCursorX } from "../../src/gradlab/web_player/panels/shared.js";

test("dynamic metric names round-trip without path ambiguity", () => {
  const name = "coins / bonus%";
  const encoded = encodeDynamicSegment(name);
  assert.equal(decodeDynamicSegment(encoded), name);
  const key = dynamicDescriptorKey("signal", name);
  const descriptor = descriptorFor(key);
  assert.equal(descriptor.name, name);
  assert.equal(descriptor.namespace, "signal");
});

test("catalog discovers signal and reward-component descriptors", () => {
  const snapshot = {
    transition: {
      signals: { x_pos: 12 },
      reward: { components: { progress: 1.25 } },
    },
  };
  const catalog = descriptorCatalog(snapshot, [
    { signals: { coins: 3 }, components: { terminal: -1 } },
  ]);
  assert.ok(catalog.has(dynamicDescriptorKey("signal", "x_pos")));
  assert.ok(catalog.has(dynamicDescriptorKey("signal", "coins")));
  assert.ok(catalog.has(dynamicDescriptorKey("reward-component", "progress")));
  assert.ok(catalog.has(dynamicDescriptorKey("reward-component", "terminal")));
});

test("action descriptors distinguish policy-selected and executed action", () => {
  const point = { policy_action: 2, executed_action: 1 };
  assert.equal(
    descriptorValue(descriptorFor("action/policy"), { point }),
    2,
  );
  assert.equal(
    descriptorValue(descriptorFor("action/executed"), { point }),
    1,
  );
});

test("numeric series preserve gaps and unit compatibility is explicit", () => {
  const series = seriesForMetric("reward/shaped", [
    { reward_shaped: 1 },
    { reward_shaped: null },
    { reward_shaped: -0.5 },
  ]);
  assert.equal(series[0], 1);
  assert.ok(Number.isNaN(series[1]));
  assert.equal(series[2], -0.5);
  const catalog = descriptorCatalog(null, []);
  assert.equal(
    compatibleMetricKeys(["reward/provider", "reward/shaped"], catalog),
    true,
  );
  assert.equal(
    compatibleMetricKeys(["reward/shaped", "reward/return"], catalog),
    false,
  );
  assert.ok(metricOptions(catalog, "histogram").some(
    (descriptor) => descriptor.key === "action/executed",
  ));
});

test("the chart cursor remains on the newest live transition", () => {
  const history = [{ sequence: 4 }, { sequence: 5 }];
  assert.equal(
    cursorIndex(history, {
      inspection: false,
      selectedSequence: 5,
    }),
    1,
  );
});

test("the final chart cursor stays inside the canvas clipping edge", () => {
  const plot = { left: 20, right: 200 };
  assert.equal(lineCursorX(plot, 0, 5), 21);
  assert.equal(lineCursorX(plot, 4, 5), 199);
});
