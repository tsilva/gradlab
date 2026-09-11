import test from 'node:test';
import assert from 'node:assert/strict';
import { chartHarness, full, flush } from './helpers/chart-history.mjs';

test('a new range immediately clears the old plot and late work cannot replace the latest selection', async () => {
  const h = chartHarness();
  h.history.setDemand(true);
  h.requests[0].resolve(full()); await flush();
  assert.deepEqual(h.history.read().data.map(p => p.step), [1, 10]);
  h.history.selectRange({ first: 2, last: 5 });
  assert.equal(h.history.read().status, 'loading');
  assert.equal(h.history.read().data, null);
  h.history.selectRange({ first: 6, last: 9 });
  assert.equal(h.requests[1].signal.aborted, true);
  h.requests[1].resolve(full([2, 5], 'obsolete')); await flush();
  assert.equal(h.history.read().data, null);
  assert.deepEqual(h.history.read().range, { first: 6, last: 9 });
  h.requests[2].resolve(full([6, 9], 'current')); await flush();
  assert.deepEqual(h.history.read().data.map(p => p.step), [6, 9]);
  assert.equal(h.history.read().status, 'ready');
});

test('paused history retries after 1, 2, and 4 seconds, then stays failed until explicit Retry', async () => {
  const h = chartHarness();
  h.history.setDemand(true);
  for (const [index, delay] of [1000, 2000, 4000].entries()) {
    h.requests[index].reject(new TypeError('Network unavailable')); await flush();
    assert.equal(h.history.read().status, 'recovering');
    await h.advance(delay - 1);
    assert.equal(h.requests.length, index + 1);
    await h.advance(1);
    assert.equal(h.requests.length, index + 2);
  }
  h.requests[3].reject(new TypeError('Network unavailable')); await flush();
  assert.equal(h.history.read().status, 'error');
  h.update({ lastStep: 11 }); await h.advance(60000);
  assert.equal(h.requests.length, 4);
  assert.equal(h.history.read().status, 'error');
  h.history.retry();
  h.requests[4].resolve(full()); await flush();
  assert.equal(h.history.read().status, 'ready');
  assert.equal(h.history.read().error, null);
});

test('live updates coalesce behind one pending request and preserve recorded sampling and annotations', async () => {
  const h = chartHarness();
  h.history.setDemand(true);
  h.requests[0].resolve({ ...full(), constants: [[['episode'], 1], [['realized_return'], 7]] }); await flush();
  h.update({ lastStep: 11, liveHistory: [9, 10, 11].map(step => ({ step, episode: 1 })), throughStep: 11 });
  assert.deepEqual(h.history.read().data.map(p => p.step), [1, 10, 11]);
  await h.advance(1000);
  assert.equal(h.requests.length, 2);
  assert.equal(h.history.read().status, 'refreshing');
  assert.equal(h.requests[1].base, 'a');
  h.update({ lastStep: 12 }); h.update({ lastStep: 13 });
  await h.advance(3000);
  assert.equal(h.requests.length, 2);
  h.requests[1].resolve(full([1, 13], 'b')); await flush();
  assert.deepEqual(h.history.read().data.map(p => p.step), [1, 13]);
  await h.advance(999); assert.equal(h.requests.length, 2);
  await h.advance(1); assert.equal(h.requests.length, 3);
  h.requests[2].resolve(full([1, 13], 'c')); await flush();
  h.update({ throughStep: 1 });
  assert.deepEqual(h.history.read().data.map(p => p.step), [1, 13]);
  await h.advance(5000); assert.equal(h.requests.length, 3);
});

test('an unusable revision gets one full response recovery, with a persistent error if reconstruction fails again', async () => {
  const h = chartHarness(); h.history.setDemand(true);
  h.requests[0].resolve(full()); await flush();
  h.update({ lastStep: 11 }); await h.advance(1000);
  h.requests[1].resolve({ ...full(), base: 'missing' }); await flush();
  assert.equal(h.requests.length, 3);
  assert.equal(h.requests[2].base, null);
  assert.equal(h.history.read().status, 'recovering');
  assert.deepEqual(h.history.read().data.map(p => p.step), [1, 10]);
  h.requests[2].resolve({ ...full(), base: 'still-missing' }); await flush();
  assert.equal(h.history.read().status, 'error');
  await h.advance(30000); assert.equal(h.requests.length, 3);
  h.history.retry(); assert.equal(h.requests[3].base, null);
  h.requests[3].resolve(full([1, 11], 'recovered')); await flush();
  assert.equal(h.history.read().status, 'ready');
});

test('hiding all charts cancels requests and recovery timers, and showing them refreshes while paused', async () => {
  const h = chartHarness(); h.history.setDemand(true);
  h.history.setDemand(false);
  assert.equal(h.requests[0].signal.aborted, true);
  h.history.setDemand(true);
  h.requests[0].reject(new Error('obsolete failure')); await flush();
  assert.equal(h.history.read().status, 'loading');
  h.requests[1].resolve(full()); await flush();
  h.update({ lastStep: 11 });
  h.history.setDemand(false); await h.advance(2000);
  assert.equal(h.requests.length, 2);
  h.history.setDemand(true);
  assert.equal(h.history.read().status, 'refreshing');
  assert.deepEqual(h.history.read().data.map(p => p.step), [1, 10]);
  h.requests[2].reject(new TypeError('offline')); await flush();
  h.history.setDemand(false); await h.advance(20000);
  assert.equal(h.requests.length, 3);
  h.history.setDemand(true); assert.equal(h.requests.length, 4);
  h.history.dispose();
  h.requests[3].reject(new TypeError('disposed')); await flush();
  const final = h.history.read();
  h.history.retry(); h.update({ lastStep: 30 }); h.history.setDemand(true);
  await h.advance(60000);
  assert.equal(h.requests.length, 4);
  assert.deepEqual(h.history.read(), final);
});

for (const replacement of [{ episodeId: 'episode-b', episode: 2 }, { epoch: 2 }]) {
  test(`episode and session replacement reset zoom and reject late successes and failures: ${JSON.stringify(replacement)}`, async () => {
    const h = chartHarness(); h.history.setDemand(true);
    h.history.selectRange({ first: 2, last: 5 });
    h.update(replacement);
    assert.equal(h.history.read().range, null);
    assert.equal(h.history.read().data, null);
    assert.equal(h.requests[1].signal.aborted, true);
    h.requests[0].resolve(full()); h.requests[1].reject(new Error('old')); await flush();
    assert.equal(h.history.read().status, 'loading');
    assert.equal(h.requests.length, 3);
    assert.equal(h.requests[2].base, null);
    h.requests[2].resolve(full([1, 3])); await flush();
    assert.deepEqual(h.history.read().data.map(p => p.step), [1, 3]);
  });
}

test('valid deltas preserve unchanged rows, remove points, and replace recorded annotations', async () => {
  const h = chartHarness(); h.history.setDemand(true);
  h.requests[0].resolve(full([1, 2])); await flush();
  h.update({ lastStep: 3 }); await h.advance(1000);
  h.requests[1].resolve({ ...full([3], 'b'), base: 'a', removed: [1],
    constants: [[['episode'], 1], [['realized_return'], 7], [['return_estimate_step'], 3]] }); await flush();
  assert.deepEqual(h.history.read().data, [
    { episode: 1, realized_return: 7, return_estimate_step: 3, step: 2, sequence: 2, reward_shaped: 2 },
    { episode: 1, realized_return: 7, return_estimate_step: 3, step: 3, sequence: 3, reward_shaped: 3 },
  ]);
});

test('permanent failures do not retry and new selections restart recovery', async () => {
  const h = chartHarness(); h.history.setDemand(true);
  h.requests[0].reject(new Error('No permission')); await flush();
  assert.equal(h.history.read().status, 'error');
  await h.advance(30000); assert.equal(h.requests.length, 1);
  h.history.selectRange({ first: 1, last: 3 });
  assert.equal(h.history.read().status, 'loading');
  h.requests[1].resolve(full([1, 3])); await flush();
  assert.equal(h.history.read().status, 'ready');
});

test('updates during successful recovery are coalesced into the next recorded refresh', async () => {
  const h = chartHarness(); h.history.setDemand(true);
  h.requests[0].reject(new TypeError('offline')); await flush();
  await h.advance(1000);
  h.update({ lastStep: 12 });
  h.requests[1].resolve(full()); await flush();
  await h.advance(1000);
  assert.equal(h.requests.length, 3);
});

test('same-selection recovery retains recorded annotations and live-tail episode, range and cursor limits', async () => {
  const h = chartHarness(); h.history.setDemand(true);
  h.history.selectRange({ first: 5, last: 15 });
  h.requests[1].resolve({ ...full([5, 10]), constants: [[['episode'], 1], [['realized_return'], 7]] }); await flush();
  h.update({ lastStep: 16, throughStep: 12,
    liveHistory: [{ step: 4, episode: 1 }, { step: 9, episode: 1 }, { step: 11, episode: 2 },
      { step: 11, episode: 1 }, { step: 12, episode: 1 }, { step: 13, episode: 1 }, { step: 16, episode: 1 }] });
  await h.advance(1000);
  h.requests[2].reject(new TypeError('offline')); await flush();
  assert.equal(h.history.read().status, 'recovering');
  assert.deepEqual(h.history.read().data.map(p => p.step), [5, 10, 11, 12]);
  assert.equal(h.history.read().data[1].realized_return, 7);
  h.update({ throughStep: 99 });
  assert.deepEqual(h.history.read().data.map(p => p.step), [5, 10, 11, 12, 13]);
  h.update({ throughStep: 1 });
  assert.deepEqual(h.history.read().data.map(p => p.step), [5, 10]);
  assert.deepEqual(h.history.read().range, { first: 5, last: 15 });
});

test('live-history changes notify readers without needing a playback tick', async () => {
  let changes = 0;
  const h = chartHarness(() => changes++); h.history.setDemand(true);
  h.requests[0].resolve(full()); await flush();
  const before = changes;
  h.update({ lastStep: 11, liveHistory: [{ step: 11, episode: 1 }], throughStep: 11 });
  assert.ok(changes > before);
  assert.deepEqual(h.history.read().data.map(p => p.step), [1, 10, 11]);
});

test('the browser adapter preserves query identity and distinguishes transient HTTP failures from permanent failures', async t => {
  const { createChartHistory } = await import('../../src/gradlab/web_player/chart-history.js');
  const calls = [];
  const responses = [new Response('gateway down', { status: 503 }),
    new Response(JSON.stringify(full()), { status: 200 }), new Response('forbidden', { status: 403 })];
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    calls.push({ url: new URL(url, 'http://localhost'), options });
    return responses.shift();
  });
  const timers = [];
  const history = createChartHistory({ token: 'fixture-token', clock: {
    now: () => 0, setTimeout: (callback, delay) => { timers.push({ callback, delay }); return 1; }, clearTimeout() {},
  } });
  history.updateContext({ epoch: 2, episodeId: 'episode-a', episode: 1, lastStep: 10 });
  history.selectRange({ first: 2, last: 9 }); history.setDemand(true); await flush();
  assert.equal(history.read().status, 'recovering');
  assert.equal(timers[0].delay, 1000); timers.shift().callback(); await flush();
  assert.equal(history.read().status, 'ready');
  assert.deepEqual(Object.fromEntries(calls[0].url.searchParams), {
    epoch: '2', episode_id: 'episode-a', format: 'chart-columns-v1', first: '2', last: '9',
  });
  assert.equal(calls[0].options.headers.Authorization, 'Bearer fixture-token');
  assert.ok(calls[0].options.signal instanceof AbortSignal);
  history.retry(); await flush();
  assert.equal(calls[2].url.searchParams.get('base'), 'a');
  assert.equal(history.read().status, 'error');
  assert.equal(timers.length, 0);
  history.dispose();
});
