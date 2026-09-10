import assert from "node:assert/strict";
import test from "node:test";
import { episodeStepRange, zoomedStepRange, RecordedStepReader, EventOverview, timelineEventMarkers } from "../../src/gradlab/web_player/episode-timeline.js";

test("episode range retains the first recorded step after cache eviction", () => {
  const snapshots = [{ transition: { step: 13910 } }, { transition: { step: 18005 } }];
  assert.deepEqual(episodeStepRange({ episode_id: "episode", transitions: 18005, first_step: 1, last_step: 18005 }, snapshots),
    { first: 1, last: 18005 });
  assert.deepEqual(episodeStepRange({ episode_id: "partial", transitions: 100, first_step: 90, last_step: 189 }, []),
    { first: 90, last: 189 });
  assert.deepEqual(episodeStepRange({ imported: true, first_step: 1, last_step: 18005 }, []),
    { first: 1, last: 18005 });
});

test("zoom clamps at episode ends and keeps the track stable during dragging", () => {
  const full = { first: 1, last: 18005 };
  assert.deepEqual(zoomedStepRange(full, 1, 100), { first: 1, last: 100, span: 100 });
  assert.deepEqual(zoomedStepRange(full, 18005, 100), { first: 17906, last: 18005, span: 100 });
  const window = zoomedStepRange(full, 500, 100);
  assert.equal(zoomedStepRange(full, 501, 100, window), window);
  assert.equal(zoomedStepRange(full, 500, 0, window), full);
});

test("rapid scrubbing coalesces reads and suppresses stale responses", async () => {
  const requests = [];
  const reader = new RecordedStepReader((step) => new Promise((resolve) => requests.push({ step, resolve })));
  const first = reader.read(1);
  await Promise.resolve();
  const second = reader.read(2);
  const third = reader.read(3);
  assert.deepEqual(requests.map((item) => item.step), [1]);
  requests[0].resolve("old step");
  assert.equal(await first, null);
  assert.equal(await second, null);
  assert.deepEqual(requests.map((item) => item.step), [1, 3]);
  requests[1].resolve("latest selection");
  assert.equal(await third, "latest selection");
});

test("return to latest or episode replacement invalidates an outstanding seek", async () => {
  let resolve;
  const reader = new RecordedStepReader(() => new Promise((done) => { resolve = done; }));
  const pending = reader.read(1);
  await Promise.resolve();
  reader.invalidate();
  resolve("old episode");
  assert.equal(await pending, null);
});

test("a failed read releases the queue for the next seek", async () => {
  const reader = new RecordedStepReader(async (step) => {
    if (step === 1) throw new Error("storage unavailable");
    return step;
  });
  await assert.rejects(reader.read(1), /storage unavailable/);
  assert.equal(await reader.read(2), 2);
});

test("episode markers stay fixed while inspection revisits old steps", () => {
  const overview = new EventOverview();
  overview.reset("episode");
  for (let step = 1; step <= 7453; step++) {
    overview.append({ step, events: step % 100 === 0 ? ["brick"] : [] });
  }
  const range = { first: 1, last: 7453 };
  const before = timelineEventMarkers([...overview.buckets.values()], range);
  for (let step = 1745; step <= 1873; step++) {
    overview.append({ step, events: step % 100 === 0 ? ["brick"] : [] });
  }
  const after = timelineEventMarkers([...overview.buckets.values()], range);
  assert.deepEqual(after, before);
  assert.equal(after[0].step, 100);
  assert.equal(after.at(-1).step, 7400);
});

test("dense markers combine colors and counts with room between dots", () => {
  const points = Array.from({ length: 7453 }, (_, index) => ({
    step: index + 1, last_step: index + 1, count: 1,
    events: [index % 2 ? "brick" : "life_loss"], boundary: false,
  }));
  const markers = timelineEventMarkers(points, { first: 1, last: 7453 }, 100);
  assert.ok(markers.length <= 100);
  assert.equal(markers.reduce((total, point) => total + point.count, 0), 7453);
  assert.deepEqual(markers[0].events, ["brick", "life_loss"]);
  for (let i = 1; i < markers.length; i++) assert.ok(markers[i].position - markers[i - 1].position >= 0.01);
  const zoom = timelineEventMarkers(points, { first: 1800, last: 1899 }, 100);
  assert.equal(zoom.length, 100);
  assert.equal(zoom[0].step, 1800);
});

test("overview remains bounded and reconnect restores the full episode", () => {
  const overview = new EventOverview(4);
  for (let step = 1; step <= 100; step++) overview.append({ step, events: ["brick"] });
  assert.ok(overview.buckets.size <= 4);
  assert.equal([...overview.buckets.values()].reduce((n, point) => n + point.count, 0), 100);
  const restored = new EventOverview(4);
  restored.load({ episode_id: "episode", bucket_size: overview.width, through_step: overview.throughStep,
    points: [...overview.buckets.values()] });
  assert.deepEqual([...restored.buckets.values()], [...overview.buckets.values()]);
  restored.append({ step: 100, events: ["brick"] });
  assert.equal([...restored.buckets.values()].reduce((n, point) => n + point.count, 0), 100);
  restored.reset("next-episode");
  assert.equal(restored.buckets.size, 0);
});
