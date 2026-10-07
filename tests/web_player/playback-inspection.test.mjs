import assert from 'node:assert/strict';
import test from 'node:test';

import { deferred, snapshot, recorded, flush, harness } from './helpers/playback-inspection.mjs';

test('rapid seek A/B/C publishes only C and obsolete A cannot clear its loading state', async () => {
  const a = deferred(), c = deferred(), requests = [];
  const h = harness({ fetchStep(request) { requests.push(request.step); return request.step === 10 ? a.promise : c.promise; } });
  await h.inspection.admitSnapshot(snapshot(100));
  const first = h.inspection.selectStep(10); await flush();
  const middle = h.inspection.selectStep(20);
  const last = h.inspection.selectStep(30);
  a.resolve(recorded(10)); await first; await middle; await flush();
  assert.deepEqual(requests, [10, 30]);
  assert.equal(h.inspection.view.seekingStep, 30);
  assert.equal(h.inspection.view.snapshot.transition.step, 100);
  c.resolve(recorded(30)); await last;
  assert.equal(h.inspection.view.snapshot.transition.step, 30);
  assert.equal(h.inspection.view.seekingStep, null);
  assert.deepEqual(h.commands, []);
});

test('replacement drains after old frame preparation without another event', async () => {
  const held = deferred();
  const h = harness({ prepareFrame: (kind, blob) => blob === oldFrame ? held.promise : Promise.resolve() });
  const oldFrame = new Blob(['old']), nextFrame = new Blob(['next']);
  h.inspection.setFrameDemand({ kinds: [1] });
  h.inspection.reset(1);
  await h.inspection.receiveFrame({ epoch: 1, sequence: 1, kind: 1, generation: 0, blob: oldFrame });
  const old = h.inspection.admitSnapshot(snapshot(1, { transition: { step: 1, episode: 1, after: { game_frame: true } } }));
  await flush();
  h.inspection.reset(2);
  const ready = h.inspection.receiveFrame({ epoch: 2, sequence: 2, kind: 1, generation: 0, blob: nextFrame });
  const replacement = h.inspection.admitSnapshot(snapshot(2, { epoch: 2, transition: { step: 2, episode: 1, after: { game_frame: true } } }));
  held.resolve(); await Promise.all([old, replacement, ready]); await flush();
  assert.equal(h.inspection.view.snapshot.sequence, 2);
  assert.deepEqual(h.presentations.map(item => item.snapshot.sequence), [2]);
});

for (const replacement of [{ epoch: 2 }, { episode: 2, episodeId: 'episode-b' }]) {
  for (const fail of [false, true]) test(`replacement ${JSON.stringify(replacement)} suppresses old read ${fail ? 'error' : 'data'}`, async () => {
    const old = deferred(); let reads = 0;
    const h = harness({ fetchStep: ({ step }) => ++reads === 1 ? old.promise : Promise.resolve(recorded(step, replacement)) });
    await h.inspection.admitSnapshot(snapshot(100));
    const pending = h.inspection.selectStep(10); await flush();
    await h.inspection.admitSnapshot(snapshot(90, replacement));
    const current = h.inspection.selectStep(20);
    if (fail) old.reject(new Error('obsolete')); else old.resolve(recorded(10));
    await Promise.all([pending, current]);
    assert.equal(h.inspection.view.snapshot.transition.step, 20);
    assert.equal(h.inspection.view.snapshot.session_epoch, replacement.epoch ?? 1);
    assert.equal(h.inspection.view.snapshot.transition.episode, replacement.episode ?? 1);
    assert.deepEqual(h.errors, []);
  });
}

test('live head advances during inspection and a new seek supersedes live restoration', async () => {
  const restore = deferred(), seek = deferred();
  let hold = false;
  const h = harness({ fetchStep: ({ step }) => !hold ? Promise.resolve(recorded(step)) : step === 100 ? restore.promise : seek.promise });
  await h.inspection.admitSnapshot(snapshot(90));
  await h.inspection.selectStep(20);
  await h.inspection.admitSnapshot(snapshot(100, { session: { episode: 1, target_fps: 60, sampling_mode: 'stochastic' } }));
  assert.equal(h.inspection.view.snapshot.transition.step, 20);
  assert.equal(h.inspection.view.liveSnapshot.transition.step, 100);
  hold = true;
  h.inspection.returnToLive(); await flush();
  const pending = h.inspection.selectStep(30);
  restore.resolve(recorded(100)); await flush();
  assert.equal(h.inspection.view.seekingStep, 30);
  seek.resolve(recorded(30)); await pending;
  assert.equal(h.inspection.view.snapshot.transition.step, 30);
  assert.equal(h.inspection.view.liveSnapshot.session.target_fps, 60);
  assert.equal(h.inspection.view.liveSnapshot.session.sampling_mode, 'stochastic');
});

test('live restoration recovers presentation without reverting execution settings', async () => {
  const h = harness();
  await h.inspection.admitSnapshot(snapshot(100, { session: { episode: 1, target_fps: 60, sampling_mode: 'stochastic' } }));
  await h.inspection.selectStep(20);
  h.inspection.returnToLive(); await flush();
  assert.equal(h.inspection.view.snapshot.transition.step, 100);
  assert.equal(h.inspection.view.snapshot.session.target_fps, 60);
  assert.equal(h.inspection.view.snapshot.session.sampling_mode, 'stochastic');
});

const framed = (step, options = {}) => snapshot(step, { ...options,
  transition: { step, episode: options.episode ?? 1, before: { observation_frames: 1 }, after: { game_frame: true },
    attribution: { status: 'available', generation: 3 }, cnn: { status: 'available', generation: 4 } },
});

test('missing selected frames retain the display and requests coalesce at the final cursor', async () => {
  const requested = [];
  const h = harness({ send: message => requested.push(message), fetchStep: async ({ step }) => ({ snapshot: framed(step), frames: [], points: [] }) });
  await h.inspection.admitSnapshot(snapshot(100));
  await h.inspection.setFrameDemand({ kinds: [1, 2] });
  h.frames.length = 0;
  await h.inspection.selectStep(10); await flush();
  await h.inspection.selectStep(20); await flush();
  assert.deepEqual(h.frames, []);
  assert.equal(h.timers.size, 1);
  await h.tick();
  assert.deepEqual(requested.map(({ sequence, kinds }) => ({ sequence, kinds })), [{ sequence: 20, kinds: [1, 2] }]);
  const blob = new Blob(['matching']);
  await h.inspection.receiveFrame({ epoch: 1, kind: 1, sequence: 10, blob });
  assert.deepEqual(h.frames, []);
  await h.inspection.receiveFrame({ epoch: 1, kind: 1, sequence: 20, blob });
  assert.equal(h.frames.at(-1).sequence, 20);
});

test('diagnostic frames require exact generation across socket and peer inputs', async () => {
  const h = harness({ fetchStep: async ({ step }) => ({ snapshot: framed(step), frames: [], points: [] }) });
  await h.inspection.admitSnapshot(snapshot(100));
  await h.inspection.setFrameDemand({ kinds: [3, 4] });
  await h.inspection.selectStep(20); await flush(); h.frames.length = 0;
  const blob = new Blob(['diagnostic']);
  for (const kind of [3, 4]) {
    await h.inspection.receiveFrame({ epoch: 1, sequence: 20, kind, generation: 1, blob });
    h.inspection.receivePeer({ type: 'inspection-frame', source: 'stats', target: 'main', session_epoch: 1, sequence: 20, kind, generation: 1, blob });
  }
  assert.deepEqual(h.frames, []);
  await h.inspection.receiveFrame({ epoch: 1, sequence: 20, kind: 3, generation: 3, blob });
  h.inspection.receivePeer({ type: 'inspection-frame', source: 'stats', target: 'main', session_epoch: 1, sequence: 20, kind: 4, generation: 4, blob });
  await flush();
  assert.deepEqual(h.frames.map(({ kind, generation }) => [kind, generation]), [[3, 3], [4, 4]]);
});

test('obsolete decode guards fail on a new seek, demand change, episode replacement and disposal', async () => {
  const h = harness({ fetchStep: async ({ step }) => ({ snapshot: framed(step), frames: [], points: [] }) });
  await h.inspection.admitSnapshot(snapshot(100));
  await h.inspection.setFrameDemand({ kinds: [1] });
  await h.inspection.selectStep(20);
  await h.inspection.receiveFrame({ epoch: 1, sequence: 20, kind: 1, blob: new Blob(['20']) });
  const old = h.frames.at(-1);
  assert.equal(old.isCurrent(), true);
  const next = h.inspection.selectStep(30);
  assert.equal(old.isCurrent(), false); await next;
  await h.inspection.receiveFrame({ epoch: 1, sequence: 30, kind: 1, blob: new Blob(['30']) });
  const current = h.frames.at(-1);
  await h.inspection.setFrameDemand({ kinds: [] });
  assert.equal(current.isCurrent(), false);
  await h.inspection.setFrameDemand({ kinds: [1] });
  const beforeReset = h.frames.at(-1);
  await h.inspection.admitSnapshot(snapshot(90, { episode: 2, episodeId: 'episode-b' }));
  assert.equal(beforeReset.isCurrent(), false);
  h.inspection.dispose();
  assert.equal(h.frames.every(frame => !frame.isCurrent()), true);
});

test('peer cursor, live and frame routing preserve isolation and do not feed back or pause', async () => {
  const h = harness();
  await h.inspection.admitSnapshot(snapshot(100, { run_state: 'playing' }));
  const message = { type: 'inspection-cursor', session_epoch: 1, episode: 1, source: 'stats', sequence: 20, snapshot: snapshot(20), points: [] };
  for (const patch of [{ source: 'main' }, { target: 'elsewhere' }, { session_epoch: 2 }, { episode: 2 }, { snapshot: snapshot(20, { episodeId: 'other' }) }]) {
    h.inspection.receivePeer({ ...message, ...patch });
    assert.equal(h.inspection.view.inspectionSequence, null);
  }
  h.inspection.receivePeer(message); await flush();
  assert.equal(h.inspection.view.inspectionSequence, 20);
  assert.deepEqual(h.commands, []); assert.deepEqual(h.messages, []);
  h.inspection.receivePeer({ ...message, episode: 2, sequence: null });
  assert.equal(h.inspection.view.inspectionSequence, 20);
  h.inspection.receivePeer({ ...message, sequence: null });
  assert.equal(h.inspection.view.inspectionSequence, null);
  assert.deepEqual(h.commands, []); assert.deepEqual(h.messages, []);
});

test('bounded retention keeps the inspected transition and the full recorded range', async () => {
  const h = harness();
  h.inspection.updateConnection({ historyLimit: 2 });
  await h.inspection.admitSnapshot(snapshot(50));
  await h.inspection.selectStep(10);
  for (let step = 51; step <= 100; step++) await h.inspection.admitSnapshot(snapshot(step));
  assert.equal(h.inspection.view.snapshot.transition.step, 10);
  assert.deepEqual(h.inspection.view.range, { first: 0, last: 100 });
  await h.inspection.selectStep(0);
  assert.equal(h.inspection.view.snapshot.transition.step, 0);
  assert.deepEqual(h.errors, []);
  assert.throws(() => { h.inspection.view.snapshot.transition.step = 500; }, TypeError);
});

test('disposal cancels timers and prefetch and suppresses late reads, failures and presentation', async () => {
  const hold = deferred(); let signal;
  const h = harness({ fetchStep: ({ step }, options) => {
    if (!options?.signal) return Promise.resolve(recorded(step));
    signal = options.signal; return hold.promise;
  } });
  await h.inspection.admitSnapshot(snapshot(100));
  await h.inspection.selectStep(20); h.inspection.play(); await h.tick(); await flush();
  assert.ok(signal);
  const notifications = []; h.inspection.subscribe(view => notifications.push(view));
  h.inspection.dispose();
  assert.equal(signal.aborted, true);
  assert.equal(h.timers.size, 0);
  const count = h.presentations.length;
  hold.reject(new Error('obsolete speculative failure')); await flush();
  await h.inspection.selectStep(30); h.inspection.play(); h.inspection.pause();
  assert.deepEqual(notifications, []);
  assert.equal(h.presentations.length, count);
  assert.deepEqual(h.errors, []);
});

test('old preparation errors cannot leak through a replacement drain', async () => {
  const old = deferred(); let prepares = 0;
  const h = harness({ prepareFrame: () => ++prepares === 1 ? old.promise : Promise.resolve() });
  await h.inspection.setFrameDemand({ kinds: [1] }); h.inspection.reset(1);
  await h.inspection.receiveFrame({ epoch: 1, sequence: 1, kind: 1, blob: new Blob(['old']) });
  const pending = h.inspection.admitSnapshot(framed(1)); await flush();
  h.inspection.reset(2);
  const ready = h.inspection.receiveFrame({ epoch: 2, sequence: 2, kind: 1, blob: new Blob(['next']) });
  const replacement = h.inspection.admitSnapshot(framed(2, { epoch: 2 }));
  old.reject(new Error('obsolete preparation')); await Promise.all([pending, ready, replacement]); await flush();
  assert.equal(h.inspection.view.snapshot.sequence, 2);
  assert.deepEqual(h.errors, []);
});

test('Pause during a user seek cancels loading and retains the explicit paused cursor', async () => {
  const read = deferred();
  const h = harness({ fetchStep: () => read.promise });
  await h.inspection.admitSnapshot(snapshot(100, { run_state: 'playing' }));
  const pending = h.inspection.selectStep(20); await flush();
  h.inspection.pause();
  read.resolve(recorded(20)); await pending;
  assert.equal(h.inspection.view.snapshot.transition.step, 100);
  assert.equal(h.inspection.view.seekingStep, null);
  assert.equal(h.inspection.view.running, false);
});

test('selected frames survive bounded eviction while expired peer requests return nothing', async () => {
  const h = harness(); h.inspection.updateConnection({ historyLimit: 2 });
  await h.inspection.admitSnapshot(snapshot(100)); await h.inspection.selectStep(10);
  await h.inspection.receiveFrame({ epoch: 1, kind: 1, sequence: 10, blob: new Blob(['selected']) });
  for (let sequence = 101; sequence <= 110; sequence++) {
    await h.inspection.admitSnapshot(snapshot(sequence));
    await h.inspection.receiveFrame({ epoch: 1, kind: 1, sequence, blob: new Blob([String(sequence)]) });
  }
  h.messages.length = 0;
  for (const sequence of [10, ...Array.from({ length: 10 }, (_, i) => 101 + i)]) {
    h.inspection.receivePeer({ type: 'inspection-frame-request', source: 'stats', session_epoch: 1, sequence, kinds: [1] });
  }
  assert.deepEqual(h.messages.map(message => message.sequence), [10, 110]);
});

test('frame readiness and cancellation carry the original Checkpoint presentation ticket', async () => {
  const { CheckpointSelection } = await import('../../src/gradlab/web_player/checkpoint-selection.js');
  const selection = new CheckpointSelection(() => 'select-command');
  const held = deferred(); const original = [];
  const h = harness({ prepareFrame: () => held.promise, presented: (ticket, value) => { original.push([ticket, value]); selection.presented(ticket, value); } });
  const route = { checkpoint_id: 'checkpoint-a' };
  selection.select({}, route);
  const value = framed(100, { app: { phase: 'active', route } });
  selection.receive({ ...value, type: 'snapshot' });
  // Register against the exact admitted snapshot before any preparation starts.
  value.type = 'snapshot'; selection.receive(value);
  const ticket = selection.presentationFor(value);
  await h.inspection.setFrameDemand({ kinds: [1] }); h.inspection.reset(1);
  await h.inspection.receiveFrame({ epoch: 1, kind: 1, sequence: 100, blob: new Blob(['frame']) });
  const pending = h.inspection.admitSnapshot(value, ticket); await flush();
  assert.equal(selection.view.loading, true);
  selection.terminate(); selection.select({}, { checkpoint_id: 'checkpoint-b' });
  held.resolve(); await pending; await flush();
  assert.equal(original[0][0], ticket); assert.equal(original[0][1], value);
  assert.equal(selection.view.loading, true, 'obsolete presentation cleared the next load');
});

test('all history inputs normalize and limit the current episode without changing the cursor', async () => {
  const h = harness(); h.inspection.updateConnection({ historyLimit: 2 });
  await h.inspection.admitSnapshot(snapshot(100));
  h.inspection.receiveHistory({ session_epoch: 1, points: [
    { sequence: 2, episode: 1, step: 2 }, { sequence: 1, episode: 1, step: 1 },
    { sequence: 2, episode: 1, step: 2, reward: 5 }, { sequence: 3, episode: 1, step: 3 },
  ] });
  assert.deepEqual(h.inspection.view.currentHistory.map(point => point.sequence), [2, 3]);
  assert.equal(h.inspection.view.currentHistory[0].reward, 5);
  assert.equal(h.inspection.view.snapshot.transition.step, 100);
  h.inspection.receiveHistory({ session_epoch: 2, points: [] });
  assert.equal(h.inspection.view.currentHistory.length, 2);
});

test('replay demand joins pending lookahead and Pause suppresses its ignored cancellation', async () => {
  const next = deferred(); const requests = []; let signal;
  const h = harness({ fetchStep: ({ step }, options) => {
    requests.push(step);
    if (step === 22) { signal = options?.signal; return next.promise; }
    return Promise.resolve(recorded(step));
  } });
  await h.inspection.admitSnapshot(snapshot(100)); await h.inspection.selectStep(20);
  h.inspection.play(); await h.tick(); await flush();
  const demand = h.tick(); await flush();
  assert.equal(requests.filter(step => step === 22).length, 1);
  assert.equal(h.inspection.view.seekingStep, 22);
  h.inspection.pause(); assert.equal(signal.aborted, true);
  next.reject(new Error('ignored cancellation')); await demand;
  assert.equal(h.inspection.view.snapshot.transition.step, 21);
  assert.equal(h.inspection.view.seekingStep, null);
  assert.deepEqual(h.errors, []); assert.equal(h.timers.size, 0);
});

test('oversized speculative entries are evicted and retried through ordinary replay demand', async () => {
  const requests = []; const large = 'x'.repeat(8 * 1024 * 1024);
  const h = harness({ fetchStep: async ({ step }) => { requests.push(step); return { ...recorded(step), padding: step === 22 ? large : '' }; } });
  await h.inspection.admitSnapshot(snapshot(100)); await h.inspection.selectStep(20);
  h.inspection.play(); await h.tick(); await flush();
  assert.deepEqual(requests, [20, 21, 22]);
  await h.tick(); await flush();
  assert.equal(requests.filter(step => step === 22).length, 2);
  assert.equal(h.inspection.view.snapshot.transition.step, 22);
  h.inspection.dispose();
});

test('speculative failure is retried on replay demand without emitting a speculative error', async () => {
  let attempts = 0;
  const h = harness({ fetchStep: async ({ step }) => {
    if (step === 22 && attempts++ === 0) throw Error('speculative failure');
    return recorded(step);
  } });
  await h.inspection.admitSnapshot(snapshot(100)); await h.inspection.selectStep(20);
  h.inspection.play(); await h.tick(); await flush(); await h.tick();
  assert.equal(h.inspection.view.snapshot.transition.step, 22);
  assert.equal(attempts, 2); assert.deepEqual(h.errors, []); h.inspection.dispose();
});

for (const supersede of ['seek', 'live', 'demand', 'episode', 'dispose', 'none']) {
  test(`frame decode errors report only current work after ${supersede}`, async () => {
    const held = deferred();
    const h = harness({
      fetchStep: async ({ step }) => ({ snapshot: framed(step), frames: [], points: [] }),
      renderFrame: (kind, blob) => blob?.size ? held.promise : Promise.resolve(),
    });
    await h.inspection.admitSnapshot(snapshot(100));
    await h.inspection.setFrameDemand({ kinds: [1] });
    await h.inspection.selectStep(20); await flush();
    const pending = h.inspection.receiveFrame({ epoch: 1, sequence: 20, kind: 1, blob: new Blob(['20']) });
    if (supersede === 'seek') await h.inspection.selectStep(30);
    if (supersede === 'live') h.inspection.returnToLive();
    if (supersede === 'demand') await h.inspection.setFrameDemand({ kinds: [] });
    if (supersede === 'episode') await h.inspection.admitSnapshot(snapshot(90, { episode: 2, episodeId: 'b' }));
    if (supersede === 'dispose') h.inspection.dispose();
    held.reject(new Error('decode error'));
    await pending; await flush();
    assert.deepEqual(h.errors, supersede === 'none' ? ['decode error'] : []);
    h.inspection.dispose();
  });
}

test('replacing a diagnostic generation suppresses its pending decode and error', async () => {
  const held = deferred(); let metadata;
  const h = harness({ renderFrame: (kind, blob, value) => {
    if (blob) { metadata = value; return held.promise; }
  } });
  await h.inspection.setFrameDemand({ kinds: [3] });
  await h.inspection.admitSnapshot(framed(100));
  const pending = h.inspection.receiveFrame({ epoch: 1, sequence: 100, kind: 3, generation: 3, blob: new Blob(['diagnostic']) });
  assert.equal(metadata.isCurrent(), true);
  const replacement = framed(100);
  replacement.transition.attribution.generation = 5;
  await h.inspection.admitSnapshot(replacement);
  assert.equal(metadata.isCurrent(), false);
  held.reject(new Error('obsolete diagnostic'));
  await pending;
  assert.deepEqual(h.errors, []);
  h.inspection.dispose();
});
