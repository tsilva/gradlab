import test from "node:test";
import assert from "node:assert/strict";
import { rewardInspectionRows, formatRewardCell } from "../../src/gradlab/web_player/panels/reward-inspector.js";

test("rows and discounts stay anchored to the selected playback step", () => {
  const history = Array.from({ length: 1000 }, (_, step) => ({ step, reward_provider: 0, reward_shaped: step % 10 === 0 ? 1 : 0 }));
  const rows = rewardInspectionRows(history, history[100], .99);
  assert.deepEqual(rows.map(row => row.step), [100, 110, 120, 130, 140]);
  assert.equal(rows.find(row => row.inspected).step, 100);
  assert.equal(rows[0].delay, 0);
  assert.equal(rows[0].weight, 1);
  assert.equal(rows[1].contribution, .99 ** 10);
  const afterSeek = rewardInspectionRows(history, history[895], .99);
  assert.deepEqual(afterSeek.map(row => row.step), [895, 900, 910, 920, 930]);
  assert.equal(afterSeek.find(row => row.inspected).step, 895);
  assert.equal(afterSeek[1].delay, 5);
  assert.equal(afterSeek[1].contribution, .99 ** 5);
});

test("an omitted selected sample is restored exactly within the chart window", () => {
  const history = [{ step: 10, reward_shaped: 1 }, { step: 30, reward_shaped: 2 }];
  const selected = { step: 20, reward_provider: 4, reward_shaped: -1 };
  const rows = rewardInspectionRows(history, selected, .99);
  assert.deepEqual(rows.map((row) => row.step), [20, 30]);
  assert.equal(rows[0].contribution, -1);
  assert.equal(rows[0].past, false);
  assert.equal(rows[0].weight, 1);
  assert.equal(rows[1].contribution, 2 * .99 ** 10);
  assert.equal(rewardInspectionRows(history, { step: 50 }, .99).length, 0);
});

test("missing discount and reward stay unavailable while zero is a value", () => {
  const history = [{ step: 1, reward_shaped: null }, { step: 2, reward_shaped: 0 }];
  const missing = rewardInspectionRows(history, history[0], null)[0];
  assert.equal(missing.contribution, null);
  assert.equal(missing.delay, 0);
  const zero = rewardInspectionRows(history, history[1], 0)[0];
  assert.equal(zero.weight, 1);
  assert.equal(zero.contribution, 0);
  assert.equal(formatRewardCell(null), "—");
  assert.equal(formatRewardCell(0), "0.000");
  assert.equal(formatRewardCell(-1), "-1.000");
  assert.equal(formatRewardCell(1e-12, 5), "1.00e-12");
  assert.deepEqual(rewardInspectionRows([], null, .99), []);
});

test("scrubbing preserves reference rows and discounts while moving only the highlight", () => {
  const history = Array.from({ length: 100 }, (_, step) => ({ step, reward_shaped: step % 10 === 0 ? 1 : 0 }));
  const baseline = rewardInspectionRows(history, history[40], .99, 5, 40);
  const withoutHighlight = rows => rows.map(({ inspected, ...row }) => row);
  for (const cursor of [80, 60, 20, 45, 99, 40]) {
    const rows = rewardInspectionRows(history, history[cursor], .99, 5, 40);
    assert.deepEqual(rows.map(row => row.step), [40, 50, 60, 70, 80]);
    assert.deepEqual(withoutHighlight(rows), withoutHighlight(baseline));
    assert.equal(rows.find(row => row.inspected)?.step,
      [40, 50, 60, 70, 80].includes(cursor) ? cursor : undefined);
  }
  const changedReference = rewardInspectionRows(history, history[60], .99, 5, 60);
  assert.deepEqual(changedReference.map(row => row.step), [60, 70, 80, 90]);
  assert.equal(changedReference[0].delay, 0);
});

test("seeking to a sample omitted from chart history does not insert a row", () => {
  const history = [10, 30, 50].map(step => ({ step, reward_shaped: 1 }));
  const rows = rewardInspectionRows(history, { step: 20, reward_shaped: 2 }, .99, 5, 10);
  assert.deepEqual(rows.map(row => row.step), [10, 30, 50]);
});

test("return reference 564 excludes earlier reward events", () => {
  const history = [457, 552, 564, 647, 741].map(step => ({
    step, reward_shaped: step === 564 ? 0 : 1,
  }));
  const rows = rewardInspectionRows(history, history[2], .99, 5, 564);
  assert.deepEqual(rows.map(row => row.step), [564, 647, 741]);
  assert.equal(rows[0].weight, 1);
  assert.equal(rows[1].delay, 83);
});

test("selected reward samples retain full-episode return and critic evidence", () => {
  const history = [{ step: 10, reward_shaped: 1, value: 2.5,
    realized_return: 3.25, realized_return_bootstrapped: true,
    value_comparison_reasons: [] }];
  const [row] = rewardInspectionRows(history, { step: 10, reward_shaped: 1 }, .99);
  assert.equal(row.realized_return, 3.25);
  assert.equal(row.value, 2.5);
  assert.equal(row.realized_return_bootstrapped, true);
  const [missing] = rewardInspectionRows([{ step: 10, reward_shaped: 1 }], null, .99);
  assert.equal(missing.realized_return, undefined);
  assert.equal(missing.value, undefined);
});
