import test from 'node:test';
import assert from 'node:assert/strict';
import { snapshotFailsCheckpointSelection } from '../../src/gradlab/web_player/playback-transition.js';

test('asynchronous checkpoint errors release the blocking load mask', () => {
  const load = { checkpointId: 'checkpoint-new', commandId: 'accepted-command' };
  assert.equal(snapshotFailsCheckpointSelection(load, { app: { phase: 'error', error: 'missing paddle velocity' } }), true);
  assert.equal(snapshotFailsCheckpointSelection(load, { app: { phase: 'active', error: 'replacement failed', route: { checkpoint_id: 'checkpoint-old' } } }), true);
  assert.equal(snapshotFailsCheckpointSelection(load, { app: { phase: 'loading', error: '' } }), false);
  assert.equal(snapshotFailsCheckpointSelection(null, { app: { phase: 'error' } }), false);
});

import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { snapshotActivatesCheckpointSelection } from '../../src/gradlab/web_player/playback-transition.js';

test('accepted load followed by an error snapshot removes the overlay and exposes the error', () => {
  const source = readFileSync(new URL('../../src/gradlab/web_player/app.js', import.meta.url), 'utf8');
  const handler = source.slice(source.indexOf('function handleMessage(message)'), source.indexOf('\nfunction updatePublicationButton'));
  const load = { commandId: 'accepted', checkpointId: 'new' };
  const state = { checkpointLoad: load, sessionEpoch: 1 };
  const errors = [];
  let sourceSnapshot;
  const context = vm.createContext({ state, snapshotFailsCheckpointSelection,
    snapshotActivatesCheckpointSelection,
    finishCheckpointLoad: () => { state.checkpointLoad = null; },
    showToast: (error) => errors.push(error),
    updatePublicationButton() {}, updateControlState() {},
    setSourceMode: (active, snapshot) => { sourceSnapshot = snapshot; },
  });
  vm.runInContext(handler, context);
  context.handleMessage({ type: 'command_result', id: 'accepted', ok: true });
  assert.equal(state.checkpointLoad, load);
  context.handleMessage({ type: 'snapshot', session_epoch: 1,
    app: { phase: 'error', error: 'missing paddle velocity' } });
  assert.equal(state.checkpointLoad, null);
  assert.deepEqual(errors, ['missing paddle velocity']);
  assert.equal(sourceSnapshot.app.phase, 'error');
});
