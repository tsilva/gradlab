import test from "node:test";
import assert from "node:assert/strict";
import { rewardInspectionRows, formatRewardCell } from "../../src/gradlab/web_player/panels/reward-inspector.js";

test("inspection stays bounded and hovering never changes discount reference", () => {
  const history = Array.from({ length: 1000 }, (_, step) => ({ step, reward_provider: 0, reward_shaped: step % 10 === 0 ? 1 : 0 }));
  const selected = history[100];
  const rows = rewardInspectionRows(history, selected, 895, .99);
  assert.equal(rows.length, 5);
  const inspected = rows.find((row) => row.inspected);
  assert.equal(inspected.step, 895);
  assert.equal(inspected.delay, 795);
  assert.equal(inspected.weight, .99 ** 795);
  assert.equal(inspected.contribution, 0);
  assert.deepEqual(history[100], selected);
});

test("an omitted selected sample is restored exactly within the chart window", () => {
  const history = [{ step: 10, reward_shaped: 1 }, { step: 30, reward_shaped: 2 }];
  const selected = { step: 20, reward_provider: 4, reward_shaped: -1 };
  const rows = rewardInspectionRows(history, selected, 20, .99);
  assert.deepEqual(rows.map((row) => row.step), [10, 20, 30]);
  assert.equal(rows[1].contribution, -1);
  assert.equal(rows[0].past, true);
  assert.equal(rows[0].contribution, null);
  assert.equal(rows[0].weight, null);
  assert.equal(rows[2].contribution, 2 * .99 ** 10);
  assert.equal(rewardInspectionRows(history, { step: 50 }, 50, .99).length, 2);
});

test("missing discount and reward stay unavailable while zero is a value", () => {
  const history = [{ step: 1, reward_shaped: null }, { step: 2, reward_shaped: 0 }];
  const missing = rewardInspectionRows(history, history[0], 1, null)[0];
  assert.equal(missing.contribution, null);
  assert.equal(missing.delay, 0);
  const zero = rewardInspectionRows(history, history[1], 2, 0)[0];
  assert.equal(zero.weight, 1);
  assert.equal(zero.contribution, 0);
  assert.equal(formatRewardCell(null), "—");
  assert.equal(formatRewardCell(0), "0.000");
  assert.equal(formatRewardCell(-1), "-1.000");
  assert.equal(formatRewardCell(1e-12, 5), "1.00e-12");
  assert.deepEqual(rewardInspectionRows([], null, null, .99), []);
});
