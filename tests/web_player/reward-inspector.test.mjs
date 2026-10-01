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

test("scrubbing follows the cursor while preserving the independent discount reference", () => {
  const history = Array.from({ length: 100 }, (_, step) => ({ step, reward_shaped: step % 10 === 0 ? 1 : 0 }));
  const expected = new Map([
    [80, [80, 90]], [60, [60, 70, 80, 90]], [20, [40, 50, 60, 70, 80]],
    [45, [45, 50, 60, 70, 80]], [99, [99]], [40, [40, 50, 60, 70, 80]],
  ]);
  for (const [cursor, steps] of expected) {
    const rows = rewardInspectionRows(history, history[cursor], .99, 5, 40);
    assert.deepEqual(rows.map(row => row.step), steps);
    assert.equal(rows.find(row => row.inspected)?.step, cursor >= 40 ? cursor : undefined);
    for (const row of rows) {
      assert.equal(row.delay, row.step - 40);
      assert.equal(row.weight, .99 ** (row.step - 40));
    }
  }
  const changedReference = rewardInspectionRows(history, history[60], .99, 5, 60);
  assert.deepEqual(changedReference.map(row => row.step), [60, 70, 80, 90]);
  assert.equal(changedReference[0].delay, 0);
});

test("an omitted cursor sample is restored without changing the discount reference", () => {
  const history = [10, 30, 50].map(step => ({ step, reward_shaped: 1 }));
  const rows = rewardInspectionRows(history, { step: 20, reward_shaped: 2 }, .99, 5, 10);
  assert.deepEqual(rows.map(row => row.step), [20, 30, 50]);
  assert.equal(rows[0].inspected, true);
  assert.equal(rows[0].delay, 10);
  assert.equal(rows[0].contribution, 2 * .99 ** 10);
});

test("late episode inspection includes the exact cursor and no earlier rows", () => {
  const history = [1, 3, 6, 8, 9, 520, 528].map(step => ({ step, reward_shaped: -0.001 }));
  const rows = rewardInspectionRows(history, { step: 527, reward_shaped: 0 }, .9, 5, 0);
  assert.deepEqual(rows.map(row => row.step), [527, 528]);
  assert.equal(rows[0].inspected, true);
  assert.equal(rows[0].delay, 527);
  assert.equal(rows[0].contribution, 0);
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

test("an exact pinned reference survives sampled gaps when the cursor is earlier", () => {
  const history = [1, 298, 300].map(step => ({ step, reward_shaped: .5 }));
  const reference = { step: 299, reward_shaped: .5 };
  const rows = rewardInspectionRows(history, history[1], .9, 5, 299, reference);
  assert.deepEqual(rows.map(row => row.step), [299, 300]);
  assert.equal(rows[0].delay, 0);
  assert.equal(rows[0].weight, 1);
  assert.equal(rows[1].delay, 1);
  assert.equal(rows.some(row => row.inspected), false);
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
