import test from "node:test";
import assert from "node:assert/strict";
import { rewardReferenceStore } from "../../src/gradlab/web_player/panels/reward-reference.js";

const snapshot = (step, episode = "episode-a") => ({
  trajectory: { episode_id: episode },
  transition: { step, sequence: step, episode: 1, reward: { provider: 1, shaped: 2 }, decision: { value: 3 } },
});

test("independent windows preserve one explicit reference through seeking and remounts", () => {
  const data = new Map();
  const storage = { getItem: key => data.get(key), setItem: (key, value) => data.set(key, value) };
  const chart = rewardReferenceStore(storage, "workspace");
  const table = rewardReferenceStore(storage, "workspace");
  assert.equal(chart.get(snapshot(10), 1).step, 0);
  assert.equal(table.get(snapshot(40), 1).step, 0);
  table.set(snapshot(40), 1);
  assert.equal(chart.get(snapshot(20), 1).step, 40);
  const remounted = rewardReferenceStore(storage, "workspace");
  assert.equal(remounted.get(snapshot(80), 1).step, 40);
  assert.deepEqual(remounted.get(snapshot(80), 1).sample, {
    step: 40, sequence: 40, reward_provider: 1, reward_shaped: 2, value: 3,
  });
  assert.equal(table.get(snapshot(0, "episode-b"), 1).step, 0);
  // A delayed old-episode window must not overwrite the new episode's reference.
  chart.get(snapshot(90), 1);
  assert.equal(table.get(snapshot(50, "episode-b"), 1).step, 0);
  assert.equal(table.get(snapshot(5, "episode-b"), 2).step, 0);
  assert.equal(rewardReferenceStore(storage, "another-workspace").get(snapshot(70), 1).step, 0);
});

test("invalid persisted state resets and retained reference storage remains bounded", () => {
  let value = "malformed";
  const storage = { getItem: () => value, setItem: (_key, next) => { value = next; } };
  const reference = rewardReferenceStore(storage, "workspace");
  assert.equal(reference.get(null, 1), null);
  for (let i = 0; i < 20; i++) reference.get(snapshot(i, `episode-${i}`), 1);
  assert.equal(JSON.parse(value).length, 8);
  assert.equal(reference.get(snapshot(100, "episode-19"), 1).step, 0);
});

test("late episode delivery does not move the reference or fabricate a sample", () => {
  const data = new Map();
  const storage = { getItem: key => data.get(key), setItem: (key, value) => data.set(key, value) };
  const reference = rewardReferenceStore(storage, "workspace");
  assert.equal(reference.get(snapshot(0), 1).step, 0);
  const next = reference.get(snapshot(6, "episode-b"), 1);
  assert.equal(next.step, 0);
  assert.equal(next.sample, null);
  assert.equal(reference.get(snapshot(12, "episode-b"), 1).step, 0);
  reference.set(snapshot(6, "episode-b"), 1);
  assert.equal(reference.get(snapshot(12, "episode-b"), 1).step, 6);
});
