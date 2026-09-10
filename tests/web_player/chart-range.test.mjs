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

import { resizeChartRange, bindTimelineRange } from '../../src/gradlab/web_player/chart-range.js';

test('resizing clamps to episode bounds and prevents crossed edges', () => {
  const range = {first: 194, last: 847}, episode = {first: 1, last: 1022};
  assert.deepEqual(resizeChartRange(range, episode, 'first', -100), {first: 1, last: 847});
  assert.deepEqual(resizeChartRange(range, episode, 'first', 900), {first: 846, last: 847});
  assert.deepEqual(resizeChartRange(range, episode, 'last', 100), {first: 194, last: 195});
  assert.deepEqual(resizeChartRange(range, episode, 'last', 2000), {first: 194, last: 1022});
});

test('handles publish shared ranges during drag and support keyboard resizing', () => {
  const listeners = {};
  const handle = {dataset: {edge: 'first'}, addEventListener: (name, fn) => {listeners[name] = fn;},
    focus() {}, setPointerCapture() {}, releasePointerCapture() {}};
  const band = {querySelectorAll: () => [handle], parentElement: {getBoundingClientRect: () => ({width: 1000})}};
  let range = {first: 200, last: 800};
  const updates = [];
  bindTimelineRange(band, () => ({first: 0, last: 1000}), () => range, next => {range = next; updates.push(next);});
  const event = extra => ({button: 0, pointerId: 1, clientX: 200, preventDefault() {}, stopPropagation() {}, ...extra});
  listeners.pointerdown(event());
  listeners.pointermove(event({clientX: 250}));
  assert.deepEqual(range, {first: 250, last: 800});
  listeners.pointerup(event({clientX: 300}));
  assert.deepEqual(range, {first: 300, last: 800});
  listeners.pointermove(event({clientX: 400}));
  assert.equal(updates.length, 2);
  listeners.keydown(event({key: 'ArrowLeft', shiftKey: true}));
  assert.deepEqual(range, {first: 290, last: 800});
  listeners.pointerdown(event());
  listeners.pointercancel(event());
  listeners.pointermove(event({clientX: 500}));
  assert.deepEqual(range, {first: 290, last: 800});
});
