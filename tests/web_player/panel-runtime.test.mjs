import assert from 'node:assert/strict';
import test from 'node:test';
import { PanelRuntime } from '../../src/gradlab/web_player/panels/runtime.js';
import { deferred } from './helpers/playback-inspection.mjs';

for (const method of ['renderFrame', 'prepareFrame']) {
  for (const superseded of [false, true]) test(`${method} reports only current asynchronous errors (${superseded})`, async () => {
    const held = deferred(), errors = [];
    let current = true;
    const runtime = new PanelRuntime({ onError: (id, error) => errors.push(error.message) });
    runtime.instances.set('game', { definition: { enabled: true, frameKinds: [1] }, [method]: () => held.promise });
    const pending = runtime[method](1, new Blob(['frame']), { isCurrent: () => current });
    current = !superseded;
    held.reject(new Error('decode error'));
    await pending;
    assert.deepEqual(errors, superseded ? [] : ['decode error']);
  });
}
