import test from "node:test";
import assert from "node:assert/strict";
import { rewardReferenceLabel, rewardReferenceStore } from "../../src/gradlab/web_player/panels/reward-reference.js";

import { rewardContribution } from "../../src/gradlab/web_player/panels/reward-discount.js";

const snapshot = (step, episode = "episode-a") => ({
  trajectory: { episode_id: episode },
  transition: { step, sequence: step, episode: 1, reward: { provider: 1, shaped: 2 }, decision: { value: 3 } },
});

test("independent windows preserve one explicit reference through seeking and remounts", () => {
  const data = new Map();
  const storage = { getItem: key => data.get(key), setItem: (key, value) => data.set(key, value) };
  const chart = rewardReferenceStore(storage, "workspace");
  const table = rewardReferenceStore(storage, "workspace");
  assert.equal(chart.get(snapshot(10), 1).step, 1);
  assert.equal(table.get(snapshot(40), 1).step, 1);
  table.set(snapshot(40), 1);
  assert.equal(chart.get(snapshot(20), 1).step, 40);
  const remounted = rewardReferenceStore(storage, "workspace");
  assert.equal(remounted.get(snapshot(80), 1).step, 40);
  assert.deepEqual(remounted.get(snapshot(80), 1).sample, {
    step: 40, sequence: 40, reward_provider: 1, reward_shaped: 2, value: 3,
  });
  assert.equal(table.get(snapshot(0, "episode-b"), 1).step, 1);
  // A delayed old-episode window must not overwrite the new episode's reference.
  chart.get(snapshot(90), 1);
  assert.equal(table.get(snapshot(50, "episode-b"), 1).step, 1);
  assert.equal(table.get(snapshot(5, "episode-b"), 2).step, 1);
  assert.equal(rewardReferenceStore(storage, "another-workspace").get(snapshot(70), 1).step, 1);
});

test("invalid persisted state resets and retained reference storage remains bounded", () => {
  let value = "malformed";
  const storage = { getItem: () => value, setItem: (_key, next) => { value = next; } };
  const reference = rewardReferenceStore(storage, "workspace");
  assert.equal(reference.get(null, 1), null);
  for (let i = 0; i < 20; i++) reference.get(snapshot(i, `episode-${i}`), 1);
  assert.equal(JSON.parse(value).length, 8);
  assert.equal(reference.get(snapshot(100, "episode-19"), 1).step, 1);
});

test("late episode delivery does not move the reference or fabricate a sample", () => {
  const data = new Map();
  const storage = { getItem: key => data.get(key), setItem: (key, value) => data.set(key, value) };
  const reference = rewardReferenceStore(storage, "workspace");
  assert.equal(reference.get(snapshot(0), 1).step, 1);
  const next = reference.get(snapshot(6, "episode-b"), 1);
  assert.equal(next.step, 1);
  assert.equal(next.sample, null);
  assert.equal(reference.get(snapshot(12, "episode-b"), 1).step, 1);
  reference.set(snapshot(6, "episode-b"), 1);
  assert.equal(reference.get(snapshot(12, "episode-b"), 1).step, 6);
});


test("the default first-decision reference discounts the completion bonus exactly487 times", () => {
  const storage = { getItem: () => null, setItem: () => {} };
  const store = rewardReferenceStore(storage, "new-episode");
  const reference = store.get(snapshot(488), 1);
  assert.equal(reference.step, 1);
  assert.equal(reference.sample, null, "late delivery must not fabricate the first reward");
  const { contribution } = rewardContribution({step: 488, reward_shaped: 1}, reference.step, .999);
  assert.equal(contribution, .999 ** 487);
  assert.equal(store.set(snapshot(0), 1).step, 1);
  assert.equal(store.set(snapshot(0), 1).sample, null);
});

test("existing zero references normalize without moving explicit positive references", () => {
  const values = new Map();
  const storage = {getItem: key => values.get(key), setItem: (key,value) => values.set(key,value)};
  const store = rewardReferenceStore(storage, "workspace");
  const old = {episode: JSON.stringify([1, "episode-a", 1]), step: 0, sample: {step: 0, reward_shaped: 99}};
  values.set("gradlab-reward-reference-workspace", JSON.stringify([old]));
  assert.deepEqual(store.get(snapshot(488), 1), {...old, step: 1, sample: null});
  assert.equal(JSON.parse(values.get("gradlab-reward-reference-workspace"))[0].step, 1);
  store.set(snapshot(240), 1);
  assert.equal(store.get(snapshot(12), 1).step, 240);
});

test("the visible reference caption distinguishes missing discount data from zero", () => {
  assert.equal(rewardReferenceLabel(1, .999), "Reference: step 1 · γ = 0.999");
  assert.equal(rewardReferenceLabel(240, 0), "Reference: step 240 · γ = 0");
  assert.equal(rewardReferenceLabel(null, NaN), "Reference: unavailable · γ = unavailable");
});
