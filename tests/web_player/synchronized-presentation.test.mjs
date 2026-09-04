import assert from "node:assert/strict";
import test from "node:test";

import { SynchronizedPresentation } from "../../src/gradlab/web_player/synchronized-presentation.js";

function deferred() {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
}

test("live presentation waits for the exact frame to be prepared", async () => {
  const ready = new Set();
  const preparation = deferred();
  const presented = [];
  const presentation = new SynchronizedPresentation({
    isReady: (snapshot) => ready.has(snapshot.sequence),
    prepare: async () => preparation.promise,
    present: (snapshot) => presented.push(snapshot.sequence),
  });

  await presentation.offer({ sequence: 7 });
  assert.deepEqual(presented, []);

  ready.add(7);
  const pending = presentation.notifyReady();
  await Promise.resolve();
  assert.deepEqual(presented, []);

  preparation.resolve();
  await pending;
  assert.deepEqual(presented, [7]);
});

test("live presentation coalesces a burst to the newest complete snapshot", async () => {
  const ready = new Set();
  const prepared = [];
  const presented = [];
  const presentation = new SynchronizedPresentation({
    isReady: (snapshot) => ready.has(snapshot.sequence),
    prepare: async (snapshot) => prepared.push(snapshot.sequence),
    present: (snapshot) => presented.push(snapshot.sequence),
  });

  await presentation.offer({ sequence: 10 });
  await presentation.offer({ sequence: 11 });
  await presentation.offer({ sequence: 12 });
  ready.add(10);
  ready.add(11);
  ready.add(12);

  await presentation.notifyReady();

  assert.deepEqual(prepared, [12]);
  assert.deepEqual(presented, [12]);
});

test("reset prevents an in-flight frame from advancing a new session", async () => {
  const preparation = deferred();
  const presented = [];
  const presentation = new SynchronizedPresentation({
    isReady: () => true,
    prepare: async () => preparation.promise,
    present: (snapshot) => presented.push(snapshot.sequence),
  });

  const pending = presentation.offer({ sequence: 3 });
  await Promise.resolve();
  presentation.reset();
  preparation.resolve();
  await pending;

  assert.deepEqual(presented, []);
});
