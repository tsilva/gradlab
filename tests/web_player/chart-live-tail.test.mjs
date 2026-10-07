import test from 'node:test';
import assert from 'node:assert/strict';
import { chartWithLiveTail } from '../../src/gradlab/web_player/chart-live-tail.js';
const point = (step, episode = 1) => ({ step, episode, return: step });

test('streamed transitions extend charts before the next recorded-history response', () => {
  const recorded = [point(1), point(10)];
  const live = [point(9), point(10), point(11), point(12)];
  assert.deepEqual(chartWithLiveTail(recorded, live, { episode: 1, throughStep: 11 }).map(p => p.step), [1, 10, 11]);
  assert.equal(recorded.length, 2);
  assert.deepEqual(chartWithLiveTail(recorded, live, { episode: 1 }).map(p => p.step), [1, 10, 11, 12]);
});
test('history refresh absorbs tail without duplicates or losing annotations', () => {
  const recorded = [point(1), { ...point(12), realized_return: 7 }];
  assert.equal(chartWithLiveTail(recorded, [point(11), point(12)], { episode: 1 }), recorded);
});
test('respects episode, zoom window, and empty initial history', () => {
  const live = [point(1, 0), point(1), point(2), point(3)];
  assert.deepEqual(chartWithLiveTail(null, live, { episode: 1, range: { first: 2, last: 2 } }), [point(2)]);
});
