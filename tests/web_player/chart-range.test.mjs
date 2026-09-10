import test from 'node:test';
import assert from 'node:assert/strict';
import { selectedRange } from '../../src/gradlab/web_player/chart-range.js';

test('zoom translates both drag directions to inclusive episode steps', () => {
  const plot = {left: 20, right: 120};
  assert.deepEqual(selectedRange(plot, 40, 80, 1, 1001), {first: 201, last: 601});
  assert.deepEqual(selectedRange(plot, 80, 40, 1, 1001), {first: 201, last: 601});
  assert.deepEqual(selectedRange(plot, -20, 200, 1, 1001), {first: 1, last: 1001});
  assert.equal(selectedRange(plot, 40, 43, 1, 1001), null);
});
