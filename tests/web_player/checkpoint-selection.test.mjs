import test from 'node:test';
import assert from 'node:assert/strict';
import { CheckpointSelection } from '../../src/gradlab/web_player/checkpoint-selection.js';

const route = { level: 'runs', environment_id: 'Game-v0', goal_id: 'goal', goal_variant_id: 'variant', run_id: 'run', checkpoint_id: 'a' };
const active = (checkpoint = 'a', epoch = 1, sequence = 1) => ({ type: 'snapshot', session_epoch: epoch, sequence, app: { phase: 'active', has_active_runner: true, route: { ...route, checkpoint_id: checkpoint } } });

test('selection stays busy after acknowledgement and activation until its presentation completes', () => {
  const commands = [];
  const selection = new CheckpointSelection((...args) => { commands.push(args); return 'command'; });
  const result = selection.select({ kind: 'public_run', value: 'manifest' }, route);
  assert.equal(result.commandId, 'command');
  assert.deepEqual(selection.view.route, route);
  assert.equal(selection.view.navigationPending, true);
  assert.equal(selection.view.loading, true);
  selection.receive({ type: 'command_result', id: 'command', ok: true });
  assert.equal(selection.view.loading, true);
  const snapshot = active();
  selection.receive(snapshot);
  assert.equal(selection.view.navigationPending, false);
  assert.equal(selection.view.loading, true);
  selection.presented(selection.presentationFor(snapshot), snapshot);
  assert.equal(selection.view.loading, false);
  assert.equal(commands.length, 1);
});

test('local browsing retains background playback until the selected checkpoint activates', () => {
  const commands = [];
  const selection = new CheckpointSelection((...args) => { commands.push(args); return 'command'; });
  const previous = active();
  selection.receive(previous);
  const browse = { ...route, checkpoint_id: '' };
  assert.equal(selection.browse(browse, previous).route.checkpoint_id, '');
  assert.equal(selection.view.sourceMode, true);
  assert.equal(commands.length, 0);
  assert.equal(selection.receive(active()).background, true);
  selection.select({}, { ...route, checkpoint_id: 'b' });
  assert.equal(selection.receive(active()).background, true);
  selection.receive(active('b', 2));
  assert.equal(selection.view.sourceMode, false);
  assert.equal(selection.view.backgroundSnapshot, null);
});

test('authoritative errors release loading once without rolling back the selected route', () => {
  const selection = new CheckpointSelection(() => 'command');
  selection.select({}, route);
  const error = { type: 'snapshot', app: { phase: 'error', error: 'preparation failed' } };
  assert.equal(selection.receive(error).error, 'preparation failed');
  assert.equal(selection.view.loading, false);
  assert.equal(selection.view.navigationPending, true);
  assert.deepEqual(selection.view.route, route);
  assert.equal(selection.receive(error).error, undefined);
});

test('termination and disposal invalidate outstanding presentations', () => {
  const selection = new CheckpointSelection(() => 'command');
  selection.select({}, route);
  const snapshot = active();
  selection.receive(snapshot);
  const ticket = selection.presentationFor(snapshot);
  selection.terminate();
  assert.equal(selection.view.loading, false);
  selection.select({}, route);
  selection.presented(ticket, snapshot);
  assert.equal(selection.view.loading, true);
  selection.dispose();
  assert.equal(selection.view.loading, false);
  assert.equal(selection.select({}, route), false);
});

test('a deferred presentation cannot complete a later selection of the same checkpoint', async () => {
  const selection = new CheckpointSelection(() => 'command');
  selection.select({}, route);
  const old = active();
  selection.receive(old);
  const ticket = selection.presentationFor(old);
  let release;
  const completion = new Promise(resolve => { release = resolve; }).then(() => selection.presented(ticket, old));
  selection.select({}, route);
  const current = active('a', 2);
  selection.receive(current);
  release();
  await completion;
  assert.equal(selection.view.loading, true);
  selection.presented(selection.presentationFor(current), old);
  assert.equal(selection.view.loading, true);
  selection.presented(selection.presentationFor(current), current);
  assert.equal(selection.view.loading, false);
});

test('dispatch refusal leaves route and loading unchanged; rejection preserves pending navigation', () => {
  let refusal = true;
  const selection = new CheckpointSelection(() => refusal ? null : 'command');
  assert.equal(selection.select({}, route), false);
  assert.equal(selection.view.route, null);
  assert.equal(selection.view.loading, false);
  assert.equal(selection.browse(route, null), false);
  refusal = false;
  selection.select({}, route);
  selection.receive({ type: 'command_result', id: 'other', ok: false });
  assert.equal(selection.view.loading, true);
  selection.receive({ type: 'command_result', id: 'command', ok: false });
  assert.equal(selection.view.loading, false);
  assert.equal(selection.view.navigationPending, true);
});

test('another window and imported playback can activate without a local command', () => {
  const selection = new CheckpointSelection(() => { throw Error('must not execute attachment'); });
  selection.receive(active());
  assert.equal(selection.view.sourceMode, false);
  assert.equal(selection.view.route.checkpoint_id, 'a');
  selection.receive({ ...active(), mode: 'trajectory' });
  assert.equal(selection.view.loading, false);
  assert.equal(Object.isFrozen(selection.view), true);
  assert.equal(Object.isFrozen(selection.view.route), true);
});


import { SynchronizedPresentation } from '../../src/gradlab/web_player/synchronized-presentation.js';
for (const framesFirst of [false, true]) {
  test(`selection waits for exact frames with ${framesFirst ? 'frames' : 'snapshot'} arriving first`, async () => {
    const selection = new CheckpointSelection(() => 'command');
    selection.select({}, route);
    const frames = new Set();
    const presented = [];
    const presenter = new SynchronizedPresentation({
      isReady: snapshot => frames.has(snapshot.sequence),
      prepare: async () => {},
      present: snapshot => {
        presented.push(snapshot.sequence);
        selection.presented(selection.presentationFor(snapshot), snapshot);
      },
    });
    const snapshot = active('a', 1, 12);
    if (framesFirst) frames.add(12);
    selection.receive(snapshot);
    await presenter.offer(snapshot);
    if (!framesFirst) {
      assert.equal(selection.view.loading, true);
      frames.add(11);
      await presenter.notifyReady();
      assert.equal(selection.view.loading, true);
      frames.add(12);
      await presenter.notifyReady();
    }
    assert.deepEqual(presented, [12]);
    assert.equal(selection.view.loading, false);
  });
}

test('session replacement invalidates completion before its first snapshot arrives', () => {
  const selection = new CheckpointSelection(() => 'command');
  selection.select({}, route);
  const snapshot = active();
  selection.receive(snapshot);
  const ticket = selection.presentationFor(snapshot);
  selection.receive({ type: 'session_changed', session_epoch: 2 });
  selection.presented(ticket, snapshot);
  assert.equal(selection.view.loading, true);
});

test('the selection view owns initial discovery and authoritative preparation routes', () => {
  const selection = new CheckpointSelection(() => 'command');
  selection.receive({ type: 'snapshot', app: { phase: 'selecting', route } });
  assert.deepEqual(selection.view.route, route);
  selection.select({}, route);
  const preparingRoute = { ...route, goal_variant_id: 'resolved-variant' };
  selection.receive({ type: 'snapshot', app: { phase: 'loading', route: preparingRoute } });
  assert.deepEqual(selection.view.route, preparingRoute);
  selection.receive({ type: 'snapshot', app: { phase: 'error', route: preparingRoute } });
  assert.deepEqual(selection.view.route, preparingRoute);
});


test('authoritative cancellation can present the previous session at the same epoch', () => {
  const selection = new CheckpointSelection(() => 'command');
  selection.receive(active());
  selection.select({}, { ...route, checkpoint_id: 'b' });
  selection.receive({ type: 'snapshot', session_epoch: 1, app: { phase: 'loading', route: { ...route, checkpoint_id: 'b' } } });
  // Existing cancellation keeps the runner and the requested source route.
  const cancelled = active('b', 1);
  selection.receive(cancelled);
  selection.presented(selection.presentationFor(cancelled), cancelled);
  assert.equal(selection.view.loading, false);
});

test('history decisions are returned once and reading the view never dispatches work', () => {
  const commands = [];
  const selection = new CheckpointSelection((...command) => { commands.push(command); return 'command'; });
  assert.equal(selection.select({}, route, { historyMode: 'replace' }).historyMode, 'replace');
  assert.equal(selection.browse(route, active(), { historyMode: null }).historyMode, null);
  const view = selection.view;
  for (let read = 0; read < 5; read += 1) assert.deepEqual(selection.view, view);
  assert.equal(commands.length, 1);
});

test('replacement errors remain authoritative without inferred command correlation', () => {
  const selection = new CheckpointSelection(() => 'command');
  selection.receive(active());
  selection.browse({ ...route, checkpoint_id: '' }, active());
  selection.select({}, { ...route, checkpoint_id: 'b' });
  const failure = { ...active(), app: { ...active().app, error: 'replacement failed' } };
  assert.deepEqual(selection.receive(failure), { background: true, error: 'replacement failed' });
  assert.equal(selection.view.loading, false);
  assert.equal(selection.view.sourceMode, true);
  assert.equal(selection.receive(failure).error, undefined);
});
