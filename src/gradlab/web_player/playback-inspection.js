import { FRAME_GAME, FRAME_OBSERVATION, FRAME_ATTRIBUTION, FRAME_CNN_INSPECTION } from './panels/catalog.js';
import { RecordedStepReader, EventOverview, episodeStepRange } from './episode-timeline.js';
import { RecordedStepPrefetch } from './recorded-step-prefetch.js';
import { SynchronizedPresentation } from './synchronized-presentation.js';

const RUNNING_STATES = new Set(["playing", "stepping", "continuing"]);

export function hasIndependentInference(snapshot) {
  return snapshot?.driver === "policy"
    && !snapshot?.trajectory?.imported
    && !["recording", "dataset", "trajectory"].includes(snapshot?.mode);
}

function shouldPauseForInspection(state) {
  return state.hasControl
    && (!hasIndependentInference(state.liveSnapshot) || !state.replayingInspection)
    && state.inspectionPauseCommandId === null
    && state.liveSnapshot?.mode !== "recording"
    && RUNNING_STATES.has(state.liveSnapshot?.run_state);
}

// External effects are adapters. Cursor, read validity, replay and frame eligibility
// belong to this closure; no caller can mutate its caches or generations.
export function createPlaybackInspection({
  windowId = 'main', fetchStep, command,
  peer = () => {}, send = () => {},
  prepareFrame = async () => {}, renderFrame = async () => {}, resetFrames = () => {},
  presented = () => {}, onError = () => {},
  setTimeout: schedule = globalThis.setTimeout, clearTimeout: cancel = globalThis.clearTimeout,
  now = () => performance.now(),
}) {
  const state = {
    sessionEpoch: 0, snapshot: null, liveSnapshot: null, snapshots: new Map(),
    frameBlobs: new Map([FRAME_GAME, FRAME_OBSERVATION, FRAME_ATTRIBUTION, FRAME_CNN_INSPECTION].map(kind => [kind, new Map()])),
    inspectionSequence: null, replayingInspection: false, inspectionReplayTimer: null,
    inspectionFrameRequestTimer: null, inspectionPauseCommandId: null,
    inspectionHistory: null, seekingStep: null,
    history: [], historyLimit: 4096, hasControl: false, connected: false, rgbEnabled: true,
    retainedEpisode: null, recordedEpisodeId: null, eventOverview: new EventOverview(),
    windowId,
  };
  const INSPECTION_FRAME_REQUEST_DELAY_MS = 50;
  const listeners = new Set();
  function freeze(value) {
    if (!value || typeof value !== 'object' || Object.isFrozen(value) || value instanceof Blob) return value;
    Object.values(value).forEach(freeze);
    return Object.freeze(value);
  }
  let frameKinds = [], disposed = false, view = null;
  let lifetime = 0, cursorGeneration = 0, frameDemandGeneration = 0;
  function reportError(error) { if (!disposed) onError(error instanceof Error ? error : new Error(error)); }
  function publish() {
    view = null;
    if (!disposed) listeners.forEach(listener => listener(read()));
  }
  function read() {
    if (!view) {
      const retainedSnapshots = [...state.snapshots.values()].filter(snapshot => episodeForSnapshot(snapshot) === state.retainedEpisode);
      view = Object.freeze({
        snapshot: state.snapshot, liveSnapshot: state.liveSnapshot,
        sessionEpoch: state.sessionEpoch, rgbEnabled: state.rgbEnabled,
        inspectionSequence: state.inspectionSequence, replayingInspection: state.replayingInspection,
        seekingStep: state.seekingStep, history: Object.freeze([...state.history]),
        currentHistory: Object.freeze([...currentEpisodeHistory()]),
        range: freeze(episodeStepRange(state.liveSnapshot?.trajectory, retainedSnapshots)),
        eventPoints: Object.freeze([...state.eventOverview.buckets.values()].map(point => freeze({ ...point }))),
        canReplay: canReplayInspection(), running: playbackIsRunning(),
      });
    }
    return view;
  }
  function frameKey(sequence, generation = 0) {
    return `${Number(sequence)}:${Number(generation)}`;
  }

  function frameKeySequence(key) {
    return Number(String(key).split(":", 1)[0]);
  }

  function rememberFrame(kind, sequence, generation, blob, preserveSequence = null) {
    const frames = state.frameBlobs.get(kind);
    if (!frames) return;
    frames.set(frameKey(sequence, generation), blob);
    while (frames.size > state.historyLimit) {
      const candidates = [...frames.keys()].filter(
        (candidate) => preserveSequence === null
          || frameKeySequence(candidate) !== Number(preserveSequence),
      );
      if (!candidates.length) break;
      const oldest = candidates.sort((left, right) => (
        frameKeySequence(left) - frameKeySequence(right)
      ))[0];
      frames.delete(oldest);
    }
  }

  function requiredFrameKinds(snapshot) {
    const visible = new Set(
      frameKinds.filter(kind => kind !== FRAME_GAME || state.rgbEnabled),
    );
    const required = [];
    if (visible.has(FRAME_GAME) && snapshot.transition?.after?.game_frame) {
      required.push(FRAME_GAME);
    }
    if (
      visible.has(FRAME_OBSERVATION)
      && Number(snapshot.transition?.before?.observation_frames || 0) > 0
    ) required.push(FRAME_OBSERVATION);
    return required.filter((kind) => !isGeneratedFrame(kind));
  }

  function requiredFramesAvailable(snapshot) {
    const sequence = Number(snapshot?.sequence);
    return requiredFrameKinds(snapshot).every(
      (kind) => exactFrameBlob(kind, sequence) !== null,
    );
  }

  function episodeForSnapshot(snapshot) {
    const episode = snapshot?.transition?.episode ?? snapshot?.session?.episode;
    return episode === undefined || episode === null ? null : Number(episode);
  }

  function historyKey(point) {
    return `${Number(point?.episode)}:${Number(point?.sequence)}`;
  }

  function normalizedHistory(points) {
    const byTransition = new Map();
    (Array.isArray(points) ? points : []).forEach((point) => {
      if (!point || !Number.isFinite(Number(point.sequence))) return;
      byTransition.set(historyKey(point), point);
    });
    return [...byTransition.values()]
      .sort((a, b) => Number(a.sequence) - Number(b.sequence))
      .slice(-state.historyLimit);
  }

  function ingestHistoryPoint(point) {
    if (!point || !Number.isFinite(Number(point.sequence))) return false;
    if (Number(point.episode) === state.retainedEpisode) state.eventOverview.append(point);
    const key = historyKey(point);
    const index = state.history.findIndex((candidate) => historyKey(candidate) === key);
    if (index >= 0) {
      state.history[index] = freeze({ ...state.history[index], ...point });
      return true;
    }
    state.history.push(freeze(point));
    state.history = normalizedHistory(state.history);
    return true;
  }

  function currentEpisodeHistory() {
    if (state.inspectionSequence !== null && state.inspectionHistory) return state.inspectionHistory;
    const episode = episodeForSnapshot(state.liveSnapshot) ?? state.retainedEpisode;
    if (episode === null) return state.history;
    return state.history.filter((point) => Number(point.episode) === episode);
  }

  function pruneRetainedTrace(preserveSequence = state.inspectionSequence) {
    const sequences = [...state.snapshots.keys()].sort((a, b) => a - b);
    const remove = sequences
      .filter(
        (sequence) => preserveSequence === null
          || Number(sequence) !== Number(preserveSequence),
      )
      .slice(0, Math.max(0, sequences.length - state.historyLimit));
    remove.forEach((sequence) => {
      state.snapshots.delete(sequence);
      state.frameBlobs.forEach((frames) => {
        [...frames.keys()]
          .filter((key) => frameKeySequence(key) === Number(sequence))
          .forEach((key) => frames.delete(key));
      });
    });
    if (
      state.inspectionSequence !== null
      && !state.snapshots.has(Number(state.inspectionSequence))
    ) {
      stopInspectionReplay({ render: false });
      state.inspectionSequence = null;
      state.snapshot = state.liveSnapshot;
      reportError("The selected transition expired from the bounded history.", true);
      broadcastInspection(null);
    }
  }

  function clearRetainedEpisode() {
    state.eventOverview.reset(null);
    recordedStepReader.invalidate();
    state.inspectionHistory = null;
    state.seekingStep = null;
    lifetime += 1;
    cursorGeneration += 1;
    cancelInspectionFrameRequest();
    livePresentation.reset();
    resetFrames();
    state.snapshots.clear();
    state.frameBlobs.forEach((frames) => frames.clear());
  }

  function prepareRetainedEpisode(snapshot) {
    const episode = episodeForSnapshot(snapshot);
    if (episode === null) return;
    const episodeId = snapshot.trajectory?.episode_id ?? null;
    if ((state.retainedEpisode !== null && state.retainedEpisode !== episode)
        || (state.recordedEpisodeId !== null && state.recordedEpisodeId !== episodeId)) {
      clearRetainedEpisode();
      state.history = [];
      state.snapshot = null;
      state.liveSnapshot = null;
      stopInspectionReplay({ render: false });
      state.inspectionSequence = null;
    }
    state.retainedEpisode = episode;
    state.recordedEpisodeId = episodeId;
    const overviewId = episodeId ?? `episode-${episode}`;
    if (state.eventOverview.episodeId !== overviewId) state.eventOverview.reset(overviewId);
  }

  function inspectionEpisodeSequences() {
    if (state.inspectionSequence === null) return [];
    const selected = state.snapshots.get(Number(state.inspectionSequence));
    const episode = episodeForSnapshot(selected);
    if (episode === null) return [];
    return [...state.snapshots.keys()].sort((a, b) => a - b).filter(
      (sequence) => episodeForSnapshot(state.snapshots.get(Number(sequence))) === episode,
    );
  }

  function attributionGeneration(snapshot) {
    const attribution = snapshot?.transition?.attribution;
    return attribution?.status === "available" ? Number(attribution.generation || 0) : 0;
  }

  function cnnInspectionGeneration(snapshot) {
    const cnn = snapshot?.transition?.cnn;
    return cnn?.status === "available" ? Number(cnn.generation || 0) : 0;
  }

  function isGeneratedFrame(kind) {
    return [FRAME_ATTRIBUTION, FRAME_CNN_INSPECTION].includes(Number(kind));
  }

  function frameGeneration(kind, snapshot) {
    if (Number(kind) === FRAME_ATTRIBUTION) return attributionGeneration(snapshot);
    if (Number(kind) === FRAME_CNN_INSPECTION) return cnnInspectionGeneration(snapshot);
    return 0;
  }

  function frameExpected(kind, snapshot) {
    if (isGeneratedFrame(kind)) return frameGeneration(kind, snapshot) > 0;
    if (!snapshot?.transition) return true;
    if (Number(kind) === FRAME_GAME) {
      return Boolean(snapshot.transition.after?.game_frame);
    }
    if (Number(kind) === FRAME_OBSERVATION) {
      return Number(snapshot.transition.before?.observation_frames || 0) > 0;
    }
    return false;
  }

  function exactFrameBlob(kind, sequence, generation = 0) {
    return state.frameBlobs.get(kind)?.get(frameKey(sequence, generation)) || null;
  }

  function frameMetadata(sequence, generation, { preparing = false, kind } = {}) {
    const episodeLifetime = lifetime, selection = cursorGeneration, demand = frameDemandGeneration;
    return { sequence, generation, isCurrent: () => !disposed && episodeLifetime === lifetime
      && selection === cursorGeneration && demand === frameDemandGeneration
      && (preparing ? state.inspectionSequence === null : Number(state.snapshot?.sequence) === Number(sequence))
      && (!isGeneratedFrame(kind) || (generation > 0 && frameGeneration(kind, state.snapshot) === generation)) };
  }

  async function renderEligibleFrame(kind, blob, metadata) {
    try {
      await renderFrame(kind, blob, metadata);
    } catch (error) {
      if (metadata.isCurrent()) reportError(error);
    }
  }

  async function showFramesForSequence(sequence) {
    const snapshot = state.snapshots.get(Number(sequence)) || state.snapshot;
    const retainMissing = (
      state.inspectionSequence !== null
      && Number(state.inspectionSequence) === Number(sequence)
    );
    const kinds = frameKinds.filter(kind => kind !== FRAME_GAME || state.rgbEnabled);
    const missing = [];
    await Promise.all(kinds.map(async (kind) => {
      const generation = frameGeneration(kind, snapshot);
      const expected = frameExpected(kind, snapshot);
      const blob = expected ? exactFrameBlob(kind, sequence, generation) : null;
      if (expected && !blob) missing.push(kind);
      if (blob) {
        await renderEligibleFrame(kind, blob, frameMetadata(sequence, generation, { kind }));
      } else if (!expected || !retainMissing) {
        await renderEligibleFrame(kind, null, frameMetadata(sequence, generation, { kind }));
      }
    }));
    return missing;
  }

  function inspectionFrames(sequence) {
    const snapshot = state.snapshots.get(Number(sequence)) || state.snapshot;
    return [FRAME_GAME, FRAME_OBSERVATION, FRAME_ATTRIBUTION, FRAME_CNN_INSPECTION]
      .map((kind) => {
        const generation = frameGeneration(kind, snapshot);
        return {
          kind,
          generation,
          blob: generation || !isGeneratedFrame(kind)
            ? exactFrameBlob(kind, sequence, generation)
            : null,
        };
      })
      .filter((item) => item.blob);
  }

  function broadcastInspection(sequence) {
    if (sequence === null) {
      peer({
        type: "inspection-cursor",
        session_epoch: state.sessionEpoch,
        episode: episodeForSnapshot(state.liveSnapshot),
        sequence: null,
        source: state.windowId,
      });
      return;
    }
    const snapshot = state.snapshots.get(Number(sequence)) || state.snapshot;
    peer({
      type: "inspection-cursor",
      session_epoch: state.sessionEpoch,
      episode: episodeForSnapshot(snapshot),
      sequence: Number(sequence),
      snapshot,
      frames: inspectionFrames(sequence),
      points: state.inspectionHistory,
      source: state.windowId,
    });
  }

  function requestInspectionFrames(sequence, kinds) {
    if (!kinds.length) return;
    const request = {
      session_epoch: state.sessionEpoch,
      sequence: Number(sequence),
      kinds,
      source: state.windowId,
    };
    peer({
      type: "inspection-frame-request",
      ...request,
    });
    send({
      type: "inspection_frames",
      ...request,
    });
  }

  function cancelInspectionFrameRequest() {
    cancel(state.inspectionFrameRequestTimer);
    state.inspectionFrameRequestTimer = null;
  }

  function scheduleInspectionFrameRequest(sequence, kinds) {
    cancelInspectionFrameRequest();
    if (!kinds.length) return;
    state.inspectionFrameRequestTimer = schedule(() => {
      state.inspectionFrameRequestTimer = null;
      if (Number(state.inspectionSequence) !== Number(sequence)) return;
      requestInspectionFrames(sequence, kinds);
    }, INSPECTION_FRAME_REQUEST_DELAY_MS);
  }

  function maybePauseForInspection() {
    if (!shouldPauseForInspection(state)) return;
    state.inspectionPauseCommandId = command("pause");
  }

  function setInspectionCursor(
    sequence,
    {
      announce = true,
      snapshot: suppliedSnapshot = null,
      frames = [],
      preserveReplay = false,
    } = {},
  ) {
    if (sequence === null) {
      returnToLive({ announce });
      return;
    }
    const numericSequence = Number(sequence);
    const snapshot = suppliedSnapshot || state.snapshots.get(numericSequence);
    if (!snapshot) {
      reportError("That transition is not retained in this window.", true);
      return;
    }
    if (
      Number(snapshot.session_epoch || 0) !== state.sessionEpoch
      || (snapshot.trajectory?.episode_id && state.liveSnapshot?.trajectory?.episode_id
        && snapshot.trajectory.episode_id !== state.liveSnapshot.trajectory.episode_id)
      || (
        state.retainedEpisode !== null
        && episodeForSnapshot(snapshot) !== state.retainedEpisode
      )
    ) return;
    frames.forEach(({ kind, generation = 0, blob }) => {
      if ([FRAME_GAME, FRAME_OBSERVATION, FRAME_ATTRIBUTION, FRAME_CNN_INSPECTION].includes(Number(kind)) && blob instanceof Blob) {
        rememberFrame(
          Number(kind),
          numericSequence,
          Number(generation),
          blob,
          numericSequence,
        );
      }
    });
    if (!preserveReplay) stopInspectionReplay({ render: false });
    const selection = ++cursorGeneration;
    cancelInspectionFrameRequest();
    if (announce) maybePauseForInspection();
    freeze(snapshot);
    state.snapshots.set(numericSequence, snapshot);
    state.inspectionSequence = numericSequence;
    pruneRetainedTrace(numericSequence);
    state.snapshot = snapshot;
    publish();
    void showFramesForSequence(numericSequence).then((missing) => {
      if (disposed || selection !== cursorGeneration || Number(state.inspectionSequence) !== numericSequence) return;
      scheduleInspectionFrameRequest(numericSequence, missing);
    }).catch(error => { if (selection === cursorGeneration) reportError(error); });
    if (announce) broadcastInspection(numericSequence);
  }

  function inspectSequence(sequence) {
    if (state.liveSnapshot?.mode === "trajectory") {
      const snapshot = state.snapshots.get(Number(sequence));
      if (snapshot?.transition) command("seek", { step: snapshot.transition.step });
      return;
    }
    if (state.liveSnapshot?.trajectory?.transitions > 0) {
      const point = currentEpisodeHistory().find((item) => Number(item.sequence) === Number(sequence));
      const snapshot = state.snapshots.get(Number(sequence));
      const step = point?.step ?? snapshot?.transition?.step;
      if (Number.isInteger(step)) void inspectStep(step);
      return;
    }
    recordedStepReader.invalidate();
    state.inspectionHistory = null;
    setInspectionCursor(sequence);
  }

  const recordedStepPrefetch = new RecordedStepPrefetch((request, options) => fetchStep({ ...request, rgbEnabled: state.rgbEnabled }, options));
  const recordedStepReader = new RecordedStepReader(request => recordedStepPrefetch.read(request), {
    onInvalidate: () => recordedStepPrefetch.clear(),
  });

  function recordedFrames(result) {
    return result.frames.map(({ kind, generation, png }) => ({
      kind, generation,
      blob: new Blob([Uint8Array.from(atob(png), (character) => character.charCodeAt(0))], { type: "image/png" }),
    }));
  }

  async function inspectStep(step, { preserveReplay = false } = {}) {
    const trajectory = state.liveSnapshot?.trajectory;
    if (!Number.isInteger(step)) return;
    if (trajectory?.imported) { command("seek", { step }); return; }
    if (!preserveReplay) stopInspectionReplay({ render: false });
    const selection = ++cursorGeneration;
    cancelInspectionFrameRequest();
    if (!trajectory?.episode_id || !trajectory?.transitions) {
      const entry = [...state.snapshots.entries()].find(([, snapshot]) =>
        Number(snapshot.transition?.step ?? snapshot.session?.step) === step);
      if (entry?.[0] === Number(state.liveSnapshot?.sequence)) returnToLive();
      else if (entry) inspectSequence(entry[0]);
      return;
    }
    if (step < trajectory.first_step || step > trajectory.last_step) return;
    if (step === trajectory.last_step && step === Number(state.liveSnapshot?.transition?.step)) {
      returnToLive();
      return;
    }
    maybePauseForInspection();
    const epoch = state.sessionEpoch;
    const episodeId = trajectory.episode_id;
    state.seekingStep = step;
    publish();
    try {
      const result = await recordedStepReader.read({ epoch, episode_id: episodeId, step });
      if (!result || disposed || selection !== cursorGeneration || epoch !== state.sessionEpoch || episodeId !== state.liveSnapshot?.trajectory?.episode_id) return;
      state.seekingStep = null;
      state.inspectionHistory = freeze(result.points);
      const frames = recordedFrames(result);
      setInspectionCursor(result.snapshot.sequence, { snapshot: result.snapshot, frames, preserveReplay });
      if (preserveReplay && state.replayingInspection) {
        void recordedStepPrefetch.ahead({ epoch, episode_id: episodeId, step: step + 1 }, trajectory.last_step);
      }
    } catch (error) {
      if (disposed || selection !== cursorGeneration) return;
      state.seekingStep = null;
      stopInspectionReplay({ render: false });
      reportError(error.message, true);
      publish();
    }
  }

  function returnToLive({ announce = true } = {}) {
    cursorGeneration += 1;
    recordedStepReader.invalidate();
    state.seekingStep = null;
    state.inspectionHistory = null;
    stopInspectionReplay({ render: false });
    cancelInspectionFrameRequest();
    state.inspectionSequence = null;
    state.snapshot = state.liveSnapshot;
    if (state.snapshot) {
      publish();
      void showFramesForSequence(Number(state.snapshot.sequence));
    }
    if (announce) broadcastInspection(null);
    void restoreLatestRecordedPresentation();
  }

  async function restoreLatestRecordedPresentation() {
    const selection = cursorGeneration;
    const live = state.liveSnapshot;
    const trajectory = live?.trajectory;
    if (!trajectory?.transitions || trajectory.imported || live.run_state !== "paused"
        || live.transition?.step !== trajectory.last_step) return;
    const epoch = state.sessionEpoch;
    try {
      const result = await recordedStepReader.read({ epoch, episode_id: trajectory.episode_id, step: trajectory.last_step });
      if (!result || disposed || selection !== cursorGeneration || state.inspectionSequence !== null || epoch !== state.sessionEpoch
          || state.liveSnapshot?.sequence !== live.sequence
          || state.liveSnapshot?.trajectory?.episode_id !== trajectory.episode_id) return;
      // Reconnection or cache eviction can lose the latest frame too. Restore only
      // its captured presentation; live transport and configuration stay current.
      const snapshot = { ...state.liveSnapshot, transition: result.snapshot.transition };
      recordedFrames(result).forEach(({ kind, generation, blob }) =>
        rememberFrame(kind, snapshot.sequence, generation, blob, snapshot.sequence));
      state.snapshots.set(Number(snapshot.sequence), snapshot);
      state.snapshot = freeze(snapshot);
      publish();
      void showFramesForSequence(Number(snapshot.sequence));
    } catch (error) {
      if (disposed || selection !== cursorGeneration) return;
      reportError(error);
    }
  }

  let replayGeneration = 0;
  let nextReplayAt = null;
  let replayFps = null;

  function canReplayInspection() {
    if (state.liveSnapshot?.trajectory?.imported) return false;
    if (state.liveSnapshot?.trajectory?.transitions > 0 && state.inspectionSequence !== null) {
      return Number(state.snapshot?.transition?.step) < state.liveSnapshot.trajectory.last_step;
    }
    const sequences = inspectionEpisodeSequences();
    const selectedIndex = sequences.indexOf(Number(state.inspectionSequence));
    return selectedIndex >= 0 && selectedIndex < sequences.length - 1;
  }

  function stopInspectionReplay({ render = true } = {}) {
    replayGeneration += 1;
    nextReplayAt = null;
    replayFps = null;
    if (state.replayingInspection) {
      recordedStepReader.invalidate();
      state.seekingStep = null;
    }
    if (state.inspectionReplayTimer !== null) {
      cancel(state.inspectionReplayTimer);
      state.inspectionReplayTimer = null;
    }
    const wasReplaying = state.replayingInspection;
    state.replayingInspection = false;
    if (render && wasReplaying && state.snapshot) publish();
  }

  function scheduleInspectionReplay() {
    const generation = replayGeneration;
    const fps = state.rgbEnabled === false ? 0 : Number(state.liveSnapshot?.session?.target_fps || 0);
    const interval = fps > 0 ? 1000 / fps : 0;
    const scheduledAt = now();
    nextReplayAt = nextReplayAt === null || replayFps !== fps
      ? scheduledAt + interval
      : Math.max(nextReplayAt + interval, scheduledAt);
    replayFps = fps;
    state.inspectionReplayTimer = schedule(async () => {
      state.inspectionReplayTimer = null;
      if (!state.replayingInspection || generation !== replayGeneration) return;
      // A suspended tab must not accumulate replay debt. Short timer delays
      // and read work still consume the current interval instead of adding one.
      const startedAt = now();
      if (startedAt - nextReplayAt >= interval) nextReplayAt = startedAt;
      if (state.liveSnapshot?.trajectory?.transitions > 0) {
        const nextStep = Number(state.snapshot.transition.step) + 1;
        if (nextStep > state.liveSnapshot.trajectory.last_step) {
          returnToLive();
          return;
        }
        await inspectStep(nextStep, { preserveReplay: true });
        if (state.replayingInspection && generation === replayGeneration) scheduleInspectionReplay();
        return;
      }
      const sequences = inspectionEpisodeSequences();
      const selectedIndex = sequences.indexOf(Number(state.inspectionSequence));
      const nextSequence = sequences[selectedIndex + 1];
      if (selectedIndex < 0 || nextSequence === undefined) {
        returnToLive();
        return;
      }
      if (nextSequence === Number(state.liveSnapshot?.sequence)) returnToLive();
      else setInspectionCursor(nextSequence, { preserveReplay: true });
      if (state.replayingInspection) scheduleInspectionReplay();
    }, Math.max(0, nextReplayAt - scheduledAt));
  }

  function playFromCurrentPosition() {
    if (state.liveSnapshot?.trajectory?.imported) {
      state.inspectionSequence = null;
      const trajectory = state.liveSnapshot.trajectory;
      command(trajectory.current_step >= trajectory.last_step ? "replay" : "play");
      return;
    }
    if (state.replayingInspection) return;
    if (canReplayInspection()) {
      // Resume an explicitly paused producer too, without jumping to its head.
      if (hasIndependentInference(state.liveSnapshot)
          && (state.liveSnapshot.run_state === "paused" || state.inspectionPauseCommandId !== null)
          && !state.liveSnapshot.session?.awaiting_next_episode
          && !state.liveSnapshot.trajectory?.error) command("play");
      replayGeneration += 1;
      state.replayingInspection = true;
      publish();
      scheduleInspectionReplay();
      return;
    }
    if (state.inspectionSequence !== null) returnToLive();
    if (!RUNNING_STATES.has(state.liveSnapshot?.run_state)
        || state.inspectionPauseCommandId !== null) command("play");
  }

  function pauseCurrentPlayback() {
    recordedStepReader.invalidate();
    state.seekingStep = null;
    if (state.replayingInspection) {
      stopInspectionReplay();
    }
    // Always enqueue the explicit pause, even if an earlier pause/play is unacknowledged.
    state.inspectionPauseCommandId = command("pause");
    if (hasIndependentInference(state.liveSnapshot) && state.snapshot?.sequence != null) {
      setInspectionCursor(Number(state.snapshot.sequence));
    }
  }

  function playbackIsRunning() {
    if (state.inspectionSequence !== null) return state.replayingInspection;
    return RUNNING_STATES.has(state.liveSnapshot?.run_state);
  }

  function reset(epoch = state.sessionEpoch) {
    cancelInspectionFrameRequest();
    stopInspectionReplay({ render: false });
    clearRetainedEpisode();
    state.sessionEpoch = Number(epoch) || 0;
    state.retainedEpisode = null;
    state.recordedEpisodeId = null;
    state.inspectionSequence = null;
    state.inspectionPauseCommandId = null;
    state.liveSnapshot = null;
    state.snapshot = null;
    state.history = [];
    publish();
  }

  const livePresentation = new SynchronizedPresentation({
    isReady: ({ snapshot }) => requiredFramesAvailable(snapshot),
    prepare: async ({ snapshot }) => {
      if (state.inspectionSequence !== null) return;
      await Promise.all(requiredFrameKinds(snapshot).map(async kind => {
        const metadata = frameMetadata(Number(snapshot.sequence), 0, { preparing: true, kind });
        try {
          await prepareFrame(kind, exactFrameBlob(kind, snapshot.sequence), metadata);
        } catch (error) {
          if (metadata.isCurrent()) throw error;
        }
      }));
    },
    present: ({ snapshot, ticket }) => {
      state.liveSnapshot = snapshot;
      if (snapshot.run_state === 'paused') state.inspectionPauseCommandId = null;
      if (snapshot.mode === 'trajectory') state.inspectionSequence = null;
      if (state.inspectionSequence === null) state.snapshot = snapshot;
      publish();
      const currentLifetime = lifetime;
      void (state.inspectionSequence === null ? showFramesForSequence(Number(snapshot.sequence)) : Promise.resolve())
        .then(() => { if (!disposed && currentLifetime === lifetime) presented(ticket, snapshot); })
        .catch(error => { if (currentLifetime === lifetime) reportError(error); });
    },
  });

  async function admitSnapshot(snapshot, ticket = null) {
    if (disposed) return;
    if (Number(snapshot.session_epoch || 0) !== state.sessionEpoch) reset(snapshot.session_epoch);
    freeze(snapshot);
    prepareRetainedEpisode(snapshot);
    state.hasControl = Boolean(snapshot.control?.has_control);
    state.snapshots.set(Number(snapshot.sequence), snapshot);
    pruneRetainedTrace();
    if (snapshot.history_point) ingestHistoryPoint(snapshot.history_point);
    const currentLifetime = lifetime;
    await livePresentation.offer({ snapshot, ticket }).catch(error => {
      if (currentLifetime === lifetime) reportError(error);
    });
  }

  async function receiveFrame({ epoch, episodeId, sequence, kind, generation = 0, blob }) {
    if (disposed || Number(epoch) !== state.sessionEpoch
        || (episodeId !== undefined && episodeId !== state.recordedEpisodeId)
        || !state.frameBlobs.has(kind) || !(blob instanceof Blob)
        || (kind === FRAME_GAME && !state.rgbEnabled)) return;
    rememberFrame(kind, sequence, generation, blob, state.inspectionSequence);
    const selected = state.snapshot;
    const exact = Number(selected?.sequence) === sequence
      && (!isGeneratedFrame(kind) || (frameGeneration(kind, selected) > 0 && frameGeneration(kind, selected) === generation));
    const currentLifetime = lifetime;
    try {
      if (exact && frameKinds.includes(kind)) await renderEligibleFrame(kind, blob, frameMetadata(sequence, generation, { kind }));
      if (!disposed && currentLifetime === lifetime) await livePresentation.notifyReady();
    } catch (error) { if (currentLifetime === lifetime) reportError(error); }
  }

  function receiveHistory({ session_epoch, points, timeline }) {
    if (disposed || (session_epoch !== undefined && Number(session_epoch) !== state.sessionEpoch)) return;
    state.history = normalizedHistory(points).map(freeze);
    if (timeline && (!state.recordedEpisodeId || timeline.episode_id === state.recordedEpisodeId)) state.eventOverview.load(timeline);
    else state.history.filter(point => Number(point.episode) === state.retainedEpisode).forEach(point => state.eventOverview.append(point));
    publish();
  }

  async function setFrameDemand({ kinds = frameKinds, rgbEnabled = state.rgbEnabled } = {}) {
    if (disposed) return;
    const rgbChanged = Boolean(rgbEnabled) !== state.rgbEnabled;
    frameKinds = [...new Set(kinds)].filter(kind => state.frameBlobs.has(kind));
    frameDemandGeneration += 1;
    cancelInspectionFrameRequest();
    if (rgbChanged) {
      cursorGeneration += 1;
      state.rgbEnabled = Boolean(rgbEnabled);
      recordedStepReader.invalidate();
      state.seekingStep = null;
      state.frameBlobs.get(FRAME_GAME).clear();
      resetFrames();
    }
    publish();
    const selection = cursorGeneration;
    if (state.snapshot) {
      const sequence = Number(state.snapshot.sequence);
      try {
        const missing = await showFramesForSequence(sequence);
        if (selection === cursorGeneration && state.inspectionSequence !== null) scheduleInspectionFrameRequest(sequence, missing);
      } catch (error) { if (selection === cursorGeneration) reportError(error); }
    }
    if (selection === cursorGeneration) await livePresentation.notifyReady().catch(reportError);
    if (rgbChanged && state.rgbEnabled && state.inspectionSequence !== null && selection === cursorGeneration) {
      await inspectStep(Number(state.snapshot?.transition?.step));
    }
  }

  function receivePeer(message) {
    if (disposed || message.source === state.windowId || Number(message.session_epoch || 0) !== state.sessionEpoch
        || (message.target && message.target !== state.windowId)) return;
    if (message.type === 'inspection-cursor') {
      if (Number(message.episode) !== state.retainedEpisode) return;
      if (message.sequence === null) { returnToLive({ announce: false }); return; }
      if (!message.snapshot || Number(message.snapshot.sequence) !== Number(message.sequence)
          || message.snapshot.trajectory?.episode_id !== state.recordedEpisodeId
          || Number(message.episode) !== episodeForSnapshot(message.snapshot)) return;
      recordedStepReader.invalidate();
      state.seekingStep = null;
      state.inspectionHistory = Array.isArray(message.points) ? freeze(message.points) : null;
      setInspectionCursor(Number(message.sequence), { announce: false, snapshot: message.snapshot,
        frames: Array.isArray(message.frames) ? message.frames : [] });
    } else if (message.type === 'inspection-frame-request') {
      const snapshot = state.snapshots.get(Number(message.sequence));
      if (!snapshot) return;
      for (const kind of Array.isArray(message.kinds) ? message.kinds : []) {
        const generation = frameGeneration(Number(kind), snapshot);
        const blob = exactFrameBlob(Number(kind), Number(message.sequence), generation);
        if (blob) peer({ type: 'inspection-frame', session_epoch: state.sessionEpoch,
          sequence: Number(message.sequence), kind: Number(kind), generation, blob,
          source: state.windowId, target: message.source });
      }
    } else if (message.type === 'inspection-frame' && message.target === state.windowId
        && state.inspectionSequence !== null && Number(message.sequence) === state.inspectionSequence) {
      void receiveFrame({ epoch: message.session_epoch, sequence: Number(message.sequence),
        kind: Number(message.kind), generation: Number(message.generation || 0), blob: message.blob });
    }
  }

  function updateConnection({ connected = state.connected, hasControl = state.hasControl, historyLimit = state.historyLimit }) {
    if (disposed) return;
    state.connected = Boolean(connected);
    state.hasControl = Boolean(hasControl);
    state.historyLimit = Math.max(1, Number(historyLimit) || 4096);
    livePresentation.limit = state.historyLimit;
    state.history = normalizedHistory(state.history);
    pruneRetainedTrace();
    publish();
  }

  const intent = callback => (...args) => { if (!disposed) return callback(...args); };
  return Object.freeze({
    get view() { return read(); },
    subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); },
    admitSnapshot, reset: intent(reset), receiveFrame, receiveHistory, receivePeer, setFrameDemand, updateConnection,
    commandResult({ id, ok }) {
      if (!disposed && id === state.inspectionPauseCommandId && !ok) state.inspectionPauseCommandId = null;
    },
    selectStep: intent(step => inspectStep(step)), selectSequence: intent(inspectSequence),
    play: intent(playFromCurrentPosition), pause: intent(pauseCurrentPlayback), returnToLive: intent(() => returnToLive()),
    dispose() { if (disposed) return; disposed = true; reset(); listeners.clear(); },
  });
}
