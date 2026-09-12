import assert from 'node:assert/strict';
import test from 'node:test';
import { harness, snapshot, recorded, flush, deferred } from './helpers/playback-inspection.mjs';
import { transportPresentation } from '../../src/gradlab/web_player/player-presentation.js';

async function player(options = {}) {
  const h = harness(options);
  await h.inspection.admitSnapshot(snapshot(100));
  return h;
}
const names = h => h.commands.map(command => command.name);

test('pause stops both clocks and play behind the head resumes unfinished inference', async () => {
  const h = await player();
  await h.inspection.admitSnapshot(snapshot(100, { run_state: 'playing' }));
  h.inspection.pause();
  assert.equal(h.inspection.view.inspectionSequence, 100);
  assert.equal(h.inspection.view.running, false);
  await h.inspection.admitSnapshot(snapshot(100));
  await h.inspection.selectStep(80);
  assert.equal(h.inspection.view.canReplay, true);
  h.inspection.play();
  assert.equal(h.inspection.view.running, true);
  await h.inspection.admitSnapshot(snapshot(100, { run_state: 'playing' }));
  await h.tick();
  assert.equal(h.inspection.view.snapshot.transition.step, 81);
  await h.inspection.admitSnapshot(snapshot(140, { run_state: 'playing', trajectory: { episode_id: 'episode-a', transitions: 141, first_step: 0, last_step: 140 } }));
  await h.tick();
  assert.equal(h.inspection.view.snapshot.transition.step, 82);
  assert.equal(h.inspection.view.liveSnapshot.transition.step, 140);
  assert.deepEqual(names(h), ['pause', 'play']);
  h.inspection.pause();
  assert.equal(h.inspection.view.snapshot.transition.step, 82);
  assert.equal(h.timers.size, 0);
  assert.equal(h.inspection.view.running, false);
  assert.deepEqual(names(h), ['pause', 'play', 'pause']);
});

test('replay discards suspended timer debt and rebases for FPS, RGB and unlimited playback', async () => {
  const h = await player();
  await h.inspection.selectStep(10); h.inspection.play();
  const delay = () => h.timers.values().next().value.delay;
  assert.equal(delay(), 1000 / 30);
  await h.tick(5000);
  assert.ok(Math.abs(delay() - 1000 / 30) < 1e-8);
  h.inspection.pause(); h.elapse(10000); h.inspection.play();
  assert.ok(Math.abs(delay() - 1000 / 30) < 1e-8);
  for (const fps of [60, 0, 30]) {
    await h.inspection.admitSnapshot(snapshot(100, { session: { episode: 1, target_fps: fps } }));
    await h.tick();
    assert.ok(Math.abs(delay() - (fps ? 1000 / fps : 0)) < 1e-8);
  }
  await h.inspection.setFrameDemand({ rgbEnabled: false }); await h.tick();
  assert.equal(delay(), 0);
  assert.equal(h.inspection.view.liveSnapshot.session.target_fps, 30);
  await h.inspection.setFrameDemand({ rgbEnabled: true });
  h.inspection.play(); await h.tick();
  assert.ok(Math.abs(delay() - 1000 / 30) < 1e-8);
  h.inspection.dispose();
});

test('replay catches the head without an extra timer or inference restart', async () => {
  const h = await player();
  await h.inspection.selectStep(99); h.inspection.play();
  await h.inspection.admitSnapshot(snapshot(100, { run_state: 'playing' }));
  await h.tick();
  assert.equal(h.inspection.view.inspectionSequence, null);
  assert.equal(h.inspection.view.running, true);
  assert.equal(h.timers.size, 0);
  assert.deepEqual(names(h), ['play']);
});

for (const halted of ['boundary', 'storage']) test(`replaying a ${halted}-halted episode does not restart inference`, async () => {
  const h = await player();
  const live = snapshot(100);
  if (halted === 'boundary') live.session.awaiting_next_episode = true;
  else live.trajectory.error = 'storage limit reached';
  await h.inspection.admitSnapshot(live);
  await h.inspection.selectStep(99); h.inspection.play(); await h.tick();
  assert.equal(h.inspection.view.running, false);
  assert.equal(h.timers.size, 0);
  assert.deepEqual(names(h), []);
});

test('human control keeps inference paused and imported Playback uses only recorded commands', async () => {
  const human = await player();
  await human.inspection.admitSnapshot(snapshot(100, { driver: 'human', run_state: 'playing' }));
  human.inspection.pause();
  await human.inspection.admitSnapshot(snapshot(100, { driver: 'human' }));
  await human.inspection.selectStep(50); human.inspection.play();
  assert.deepEqual(names(human), ['pause']);
  human.inspection.dispose();
  const imported = await player();
  await imported.inspection.admitSnapshot(snapshot(100, { driver: 'recorded', mode: 'trajectory',
    trajectory: { imported: true, current_step: 100, last_step: 100, first_step: 0 } }));
  imported.inspection.pause(); imported.inspection.play(); await imported.inspection.selectStep(20);
  assert.deepEqual(names(imported), ['pause', 'replay', 'seek']);
  assert.equal(imported.timers.size, 0);
  assert.equal(imported.inspection.view.inspectionSequence, null);
});

test('pause tooltip explains that both clocks stop', () => {
  const presentation = transportPresentation({ running: true, hasControl: true, independentInference: true });
  assert.equal(presentation.reason, 'Pause playback and policy inference');
  assert.equal(transportPresentation({ running: true, replaying: true, independentInference: true }).label, 'Pause');
});

test('pause and restart during a demand read leave one replay clock and suppress the old result', async () => {
  const gate = deferred(); let hold = false;
  const h = await player({ fetchStep: ({ step }) => hold ? gate.promise : Promise.resolve(recorded(step)) });
  await h.inspection.selectStep(50); hold = true; h.inspection.play();
  const oldTick = h.tick(); await flush();
  h.inspection.pause(); h.inspection.play();
  assert.equal(h.timers.size, 1);
  gate.resolve(recorded(51)); await oldTick;
  assert.equal(h.inspection.view.snapshot.transition.step, 50);
  assert.equal(h.timers.size, 1);
  hold = false; await h.tick();
  assert.equal(h.inspection.view.snapshot.transition.step, 51);
  assert.equal(h.timers.size, 1);
  h.inspection.dispose();
});

test('final explicit Pause wins before previous Pause and Play acknowledgements', async () => {
  const h = await player();
  await h.inspection.admitSnapshot(snapshot(100, { run_state: 'playing' }));
  h.inspection.pause(); await h.inspection.selectStep(50); h.inspection.play(); h.inspection.pause();
  assert.deepEqual(names(h), ['pause', 'play', 'pause']);
  assert.equal(h.inspection.view.running, false);
  assert.equal(h.timers.size, 0);
  h.inspection.commandResult({ id: 'command-1', ok: false });
  h.inspection.play();
  assert.deepEqual(names(h), ['pause', 'play', 'pause', 'play']);
  h.inspection.dispose();
});

test('observer scrubbing remains local and control loss preserves an already running replay', async () => {
  const h = await player();
  await h.inspection.selectStep(20); h.inspection.play();
  h.inspection.updateConnection({ connected: false, hasControl: false });
  await h.tick();
  assert.equal(h.inspection.view.snapshot.transition.step, 21);
  assert.equal(h.inspection.view.replayingInspection, true);
  await h.inspection.selectStep(30);
  assert.equal(h.inspection.view.snapshot.transition.step, 30);
  assert.equal(h.timers.size, 0);
  assert.deepEqual(names(h), ['play']);
});

for (const readMs of [0, 5, 16.666667]) test(`real reader and lookahead sustain 30 FPS with ${readMs} ms read latency`, async () => {
  const timers = new Map(); let now = 0, id = 0;
  const schedule = (callback, delay) => { timers.set(++id, { callback, at: now + delay }); return id; };
  const h = await player({ now: () => now, setTimeout: schedule, clearTimeout: timer => timers.delete(timer),
    fetchStep: ({ step }) => step === 0 ? Promise.resolve(recorded(0)) : new Promise(resolve => schedule(() => resolve(recorded(step)), readMs)),
  });
  await h.inspection.admitSnapshot(snapshot(10000, { trajectory: { episode_id: 'episode-a', transitions: 10001, first_step: 0, last_step: 10000 } }));
  await h.inspection.selectStep(0); h.inspection.play();
  const selected = [];
  h.inspection.subscribe(view => { const step = view.snapshot?.transition.step; if (step !== selected.at(-1)) selected.push(step); });
  const end = 10000 + 1;
  while (timers.size) {
    const [timerId, timer] = [...timers].sort(([, a], [, b]) => a.at - b.at)[0];
    if (timer.at > end) break;
    now = timer.at; timers.delete(timerId); void timer.callback(); await flush();
  }
  assert.equal(h.inspection.view.snapshot.transition.step, 300);
  assert.deepEqual(selected.filter(step => step > 0), Array.from({ length: 300 }, (_, index) => index + 1));
  h.inspection.dispose();
});
