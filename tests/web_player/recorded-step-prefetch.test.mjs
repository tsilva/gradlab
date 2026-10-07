import assert from "node:assert/strict";
import test from "node:test";
import { RecordedStepPrefetch } from "../../src/gradlab/web_player/recorded-step-prefetch.js";
import { RecordedStepReader } from "../../src/gradlab/web_player/episode-timeline.js";

const request = (step, epoch = 1) => ({ epoch, episode_id: "episode-a", step });

test("prefetch hides future reads, stays sequential, and stops at the recorded head", async () => {
  const calls = [];
  let active = 0, maximum = 0;
  const buffer = new RecordedStepPrefetch(async (query) => {
    calls.push(query.step);
    maximum = Math.max(maximum, ++active);
    await Promise.resolve();
    active--;
    return { step: query.step };
  });
  await buffer.ahead(request(1), 100);
  assert.deepEqual(calls, [1, 2, 3, 4]);
  assert.deepEqual(await buffer.read(request(1)), { step: 1 });
  assert.equal(calls.length, 4);
  await buffer.ahead(request(2), 5);
  assert.deepEqual(calls, [1, 2, 3, 4, 5]);
  await buffer.ahead(request(5), 5);
  assert.deepEqual(await buffer.read(request(5)), { step: 5 });
  assert.equal(maximum, 1);
  assert.equal(buffer.cache.size, 1);
});

test("a demanded step joins its in-flight prefetch", async () => {
  let resolve, calls = 0;
  const buffer = new RecordedStepPrefetch(() => {
    calls++;
    return new Promise(done => { resolve = done; });
  });
  const warming = buffer.ahead(request(1), 1);
  const read = buffer.read(request(1));
  resolve({ step: 1 });
  assert.deepEqual(await read, { step: 1 });
  await warming;
  assert.equal(calls, 1);
});

test("invalidation aborts prefetch and discards even responses that ignore cancellation", async () => {
  let resolve, signal;
  const buffer = new RecordedStepPrefetch((_query, options) => {
    signal = options.signal;
    return new Promise(done => { resolve = done; });
  });
  const warming = buffer.ahead(request(1), 4);
  buffer.clear();
  assert.equal(signal.aborted, true);
  resolve({ step: 1 });
  await warming;
  assert.equal(buffer.cache.size, 0);
  assert.equal(buffer.bytes, 0);
});

test("cached steps are isolated by session and episode", async () => {
  const calls = [];
  const buffer = new RecordedStepPrefetch(async query => { calls.push(query); return query; });
  await buffer.ahead(request(1), 1);
  assert.deepEqual(await buffer.read(request(1, 2)), request(1, 2));
  const otherEpisode = { ...request(1), episode_id: "episode-b" };
  assert.deepEqual(await buffer.read(otherEpisode), otherEpisode);
  assert.equal(calls.length, 3);
});

test("prefetch obeys its byte budget and oversized steps remain readable on demand", async () => {
  let calls = 0;
  const buffer = new RecordedStepPrefetch(async query => {
    calls++;
    return { step: query.step, text: "x".repeat(100) };
  }, { maxBytes: 150 });
  await buffer.ahead(request(1), 100);
  assert.equal(buffer.cache.size, 1);
  assert.ok(buffer.bytes <= 150);
  assert.equal(calls, 2);
  buffer.clear();
  assert.equal(buffer.bytes, 0);
  const oversized = new RecordedStepPrefetch(async query => ({ ...query, text: "x".repeat(200) }), { maxBytes: 100 });
  await oversized.ahead(request(1), 4);
  assert.equal(oversized.cache.size, 0);
  assert.equal((await oversized.read(request(1))).text.length, 200);
});

test("a failed speculative request can be retried when that step is demanded", async () => {
  let calls = 0;
  const buffer = new RecordedStepPrefetch(async () => {
    if (++calls === 1) throw new Error("temporary read failure");
    return { step: 1 };
  });
  await buffer.ahead(request(1), 4);
  assert.deepEqual(await buffer.read(request(1)), { step: 1 });
});

test("pausing a demand read joined to prefetch suppresses cancellation errors", async () => {
  const buffer = new RecordedStepPrefetch((_query, { signal }) => new Promise((_resolve, reject) => {
    signal.addEventListener("abort", () => reject(new Error("aborted")));
  }));
  const reader = new RecordedStepReader(query => buffer.read(query), { onInvalidate: () => buffer.clear() });
  const warming = buffer.ahead(request(1), 4);
  const reading = reader.read(request(1));
  await Promise.resolve();
  reader.invalidate();
  await warming;
  assert.equal(await reading, null);
  assert.equal(buffer.cache.size, 0);
});
