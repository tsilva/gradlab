import assert from "node:assert/strict";
import test from "node:test";
import { createPlaybackTransport, shouldPauseForInspection } from "../../src/gradlab/web_player/playback-transport.js";
import { transportPresentation } from "../../src/gradlab/web_player/player-presentation.js";

function harness({ driver = "policy", mode, imported = false, acknowledgeCommands = true } = {}) {
  const snapshot = step => ({ sequence: step, transition: { step }, driver, mode,
    run_state: "playing", session: { target_fps: 30 },
    trajectory: { imported, transitions: 100, first_step: 0, last_step: 100, current_step: step },
  });
  const state = { liveSnapshot: snapshot(100), snapshot: snapshot(100), hasControl: true,
    inspectionSequence: null, inspectionPauseCommandId: null, replayingInspection: false,
    inspectionReplayTimer: null, seekingStep: null, timelineSequences: [] };
  const commands = [], timers = new Map(), requests = [];
  let timerId = 0, invalidations = 0, transport, now = 0;
  const select = (sequence, { preserveReplay = false } = {}) => {
    if (!preserveReplay) transport.stopInspectionReplay({ render: false });
    if (shouldPauseForInspection(state)) command("pause");
    state.inspectionSequence = sequence;
    state.snapshot = snapshot(sequence);
  };
  const command = name => {
    commands.push(name);
    if (acknowledgeCommands) state.liveSnapshot.run_state = name === "pause" ? "paused" : "playing";
    return `command-${commands.length}`;
  };
  const returnToLive = () => {
    transport.stopInspectionReplay({ render: false });
    state.inspectionSequence = null;
    state.snapshot = state.liveSnapshot;
  };
  const services = { state, command, renderSnapshot() {}, setInspectionCursor: select, returnToLive,
    now: () => now,
    inspectionEpisodeSequences: () => [], invalidateRead() { invalidations++; },
    async inspectStep(step, options) {
      requests.push(step);
      const version = invalidations;
      await services.beforeRead?.();
      if (version !== invalidations) return;
      if (step === state.liveSnapshot.trajectory.last_step) returnToLive();
      else select(step, options);
    },
    setTimeout(callback, delay) { timers.set(++timerId, { callback, delay }); return timerId; },
    clearTimeout(id) { timers.delete(id); },
  };
  transport = createPlaybackTransport(services);
  return { state, commands, timers, requests, select, transport, services,
    get now() { return now; },
    elapse(ms) { now += ms; },
    get invalidations() { return invalidations; },
    advanceInference(step) {
      if (state.liveSnapshot.run_state !== "playing") return;
      state.liveSnapshot = snapshot(step);
      state.liveSnapshot.trajectory.last_step = step;
      state.liveSnapshot.trajectory.transitions = step;
      if (state.inspectionSequence === null) state.snapshot = state.liveSnapshot;
    },
    async tick(lateMs = 0) {
      const [id, timer] = timers.entries().next().value;
      timers.delete(id);
      now += timer.delay + lateMs;
      await timer.callback();
    },
  };
}

test("pause stops both clocks and play behind the head resumes unfinished inference", async () => {
  const h = harness();
  h.transport.pauseCurrentPlayback();
  assert.equal(h.state.inspectionSequence, 100);
  assert.equal(h.transport.playbackIsRunning(), false);
  h.advanceInference(120);
  assert.equal(h.state.snapshot.transition.step, 100);
  assert.equal(h.state.liveSnapshot.transition.step, 100);
  h.select(80);
  assert.equal(h.transport.canReplayInspection(), true);
  h.transport.playFromCurrentPosition();
  assert.equal(h.transport.playbackIsRunning(), true);
  assert.equal(h.timers.values().next().value.delay, 1000 / 30);
  await h.tick();
  assert.equal(h.state.snapshot.transition.step, 81);
  h.advanceInference(140);
  await h.tick();
  assert.equal(h.state.snapshot.transition.step, 82);
  assert.equal(h.state.liveSnapshot.transition.step, 140);
  assert.deepEqual(h.commands, ["pause", "play"]);
  h.transport.pauseCurrentPlayback();
  h.advanceInference(160);
  assert.equal(h.state.snapshot.transition.step, 82);
  assert.equal(h.timers.size, 0);
  assert.equal(h.state.liveSnapshot.transition.step, 140);
  assert.equal(h.transport.playbackIsRunning(), false);
  assert.deepEqual(h.commands, ["pause", "play", "pause"]);
});

for (const readMs of [0, 5, 16.666667]) {
  test(`replay sustains 30 FPS with ${readMs} ms reads and timer jitter`, async () => {
    const h = harness();
    h.advanceInference(10000);
    h.select(0);
    h.services.beforeRead = () => h.elapse(readMs);
    h.transport.playFromCurrentPosition();
    for (let frame = 0; frame < 300; frame++) await h.tick(frame % 3);
    assert.ok(Math.abs(h.now - 10000 - readMs) < 3, `300 frames took ${h.now} ms`);
    assert.deepEqual(h.requests, Array.from({ length: 300 }, (_, index) => index + 1));
    assert.equal(h.timers.size, 1);
  });
}

test("replay discards timing debt after a stalled timer and slow reads", async () => {
  const h = harness();
  h.select(0);
  h.transport.playFromCurrentPosition();
  await h.tick(5000);
  const nextDelay = () => h.timers.values().next().value.delay;
  assert.ok(nextDelay() >= 33 && nextDelay() <= 34);
  h.services.beforeRead = () => h.elapse(100);
  for (let frame = 0; frame < 5; frame++) await h.tick();
  h.services.beforeRead = null;
  await h.tick();
  assert.ok(nextDelay() >= 33 && nextDelay() <= 34);
  assert.deepEqual(h.requests, [1, 2, 3, 4, 5, 6, 7]);
  assert.equal(h.timers.size, 1);
});

test("replay rebases its clock on restart and FPS changes including unlimited", async () => {
  const h = harness();
  h.select(0);
  h.transport.playFromCurrentPosition();
  await h.tick();
  h.transport.pauseCurrentPlayback();
  h.elapse(10000);
  h.transport.playFromCurrentPosition();
  const nextDelay = () => h.timers.values().next().value.delay;
  assert.ok(Math.abs(nextDelay() - 1000 / 30) < 1e-8);
  h.state.liveSnapshot.session.target_fps = 60;
  await h.tick();
  assert.ok(Math.abs(nextDelay() - 1000 / 60) < 1e-8);
  h.state.liveSnapshot.session.target_fps = 0;
  await h.tick();
  assert.equal(nextDelay(), 0);
  await h.tick();
  assert.equal(nextDelay(), 0);
  h.state.liveSnapshot.session.target_fps = 30;
  await h.tick();
  assert.ok(Math.abs(nextDelay() - 1000 / 30) < 1e-8);
});

test("playing an older step resumes a paused producer without seeking to the live head", () => {
  const h = harness();
  h.state.liveSnapshot.run_state = "paused";
  h.select(50);
  h.transport.playFromCurrentPosition();
  assert.deepEqual(h.commands, ["play"]);
  assert.equal(h.state.snapshot.transition.step, 50);
  assert.equal(h.state.replayingInspection, true);
});

test("replay catches the live head without an extra timer or inference restart", async () => {
  const h = harness();
  h.select(99);
  h.transport.playFromCurrentPosition();
  await h.tick();
  assert.equal(h.state.inspectionSequence, null);
  assert.equal(h.transport.playbackIsRunning(), true);
  assert.equal(h.timers.size, 0);
  assert.deepEqual(h.commands, ["pause", "play"]);
});

test("replaying a completed or storage-failed episode does not restart inference", async () => {
  for (const halted of ["boundary", "storage"]) {
    const h = harness();
    h.state.liveSnapshot.run_state = "paused";
    if (halted === "boundary") h.state.liveSnapshot.session.awaiting_next_episode = true;
    else h.state.liveSnapshot.trajectory.error = "storage limit reached";
    h.select(99);
    h.transport.playFromCurrentPosition();
    await h.tick();
    assert.equal(h.transport.playbackIsRunning(), false);
    assert.equal(h.timers.size, 0);
    assert.deepEqual(h.commands, []);
  }
});

test("human control retains execution pauses and imported playback uses recorded transport", () => {
  const human = harness({ driver: "human" });
  human.transport.pauseCurrentPlayback();
  assert.deepEqual(human.commands, ["pause"]);
  human.select(50);
  assert.deepEqual(human.commands, ["pause"]);
  const imported = harness({ driver: "recorded", mode: "trajectory", imported: true });
  imported.transport.pauseCurrentPlayback();
  imported.transport.playFromCurrentPosition();
  assert.deepEqual(imported.commands, ["pause", "replay"]);
});

test("pause tooltip explains that both clocks stop", () => {
  const presentation = transportPresentation({ running: true, hasControl: true, independentInference: true });
  assert.equal(presentation.reason, "Pause playback and policy inference");
  assert.equal(transportPresentation({ running: true, replaying: true, independentInference: true }).label, "Pause");
});


test("pausing and restarting during a pending read does not create two replay clocks", async () => {
  const h = harness();
  h.select(50);
  let release;
  h.services.beforeRead = () => new Promise(resolve => { release = resolve; });
  h.transport.playFromCurrentPosition();
  const oldTick = h.tick();
  h.transport.pauseCurrentPlayback();
  h.transport.playFromCurrentPosition();
  assert.equal(h.timers.size, 1);
  release();
  await oldTick;
  assert.equal(h.state.snapshot.transition.step, 50);
  assert.equal(h.timers.size, 1);
  h.services.beforeRead = null;
  await h.tick();
  assert.equal(h.state.snapshot.transition.step, 51);
  assert.equal(h.timers.size, 1);
});


test("play behind the cursor supersedes an inference pause still awaiting acknowledgement", () => {
  const h = harness();
  h.state.inspectionSequence = 50;
  h.state.snapshot = { sequence: 50, transition: { step: 50 } };
  h.state.inspectionPauseCommandId = "pending-pause";
  h.transport.playFromCurrentPosition();
  assert.deepEqual(h.commands, ["play"]);
  assert.equal(h.state.replayingInspection, true);
});


test("rapid pause, play, pause sends the final pause even before the server acknowledges", () => {
  const h = harness({ acknowledgeCommands: false });
  h.transport.pauseCurrentPlayback();
  h.select(50);
  h.transport.playFromCurrentPosition();
  h.transport.pauseCurrentPlayback();
  assert.deepEqual(h.commands, ["pause", "play", "pause"]);
  assert.equal(h.transport.playbackIsRunning(), false);
  assert.equal(h.timers.size, 0);
});

test("hiding RGB bypasses configured replay FPS and restoring RGB restores pacing", async () => {
  const h = harness();
  h.select(10);
  h.state.rgbEnabled = false;
  h.transport.playFromCurrentPosition();
  assert.equal([...h.timers.values()][0].delay, 0);
  await h.tick();
  assert.equal(h.state.liveSnapshot.session.target_fps, 30);
  h.state.rgbEnabled = true;
  await h.tick();
  assert.equal([...h.timers.values()][0].delay, 1000 / 30);
});
