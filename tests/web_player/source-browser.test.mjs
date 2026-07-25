import assert from "node:assert/strict";
import test from "node:test";

import {
  formatMetricValue,
  metricLabel,
  sortRunItems,
} from "../../src/rlab/web_player/sources/browser.js";

const METRIC = "eval/full/episode/return/mean";

test("run metrics use compact labels and values", () => {
  assert.equal(metricLabel("leader/checkpoint/step"), "Checkpoint step");
  assert.equal(metricLabel(METRIC), "Mean return");
  assert.equal(
    formatMetricValue("eval/full/outcome/success/rate/min", 0.875),
    "87.5%",
  );
  assert.equal(formatMetricValue(METRIC, null), "—");
});

test("run metric sorting respects direction and keeps missing values last", () => {
  const items = [
    { run_id: "missing", metrics: { [METRIC]: null } },
    { run_id: "low", metrics: { [METRIC]: 10 } },
    { run_id: "high", metrics: { [METRIC]: 20 } },
  ];

  assert.deepEqual(
    sortRunItems(items, { metric: METRIC, direction: "descending" })
      .map((item) => item.run_id),
    ["high", "low", "missing"],
  );
  assert.deepEqual(
    sortRunItems(items, { metric: METRIC, direction: "ascending" })
      .map((item) => item.run_id),
    ["low", "high", "missing"],
  );
});
