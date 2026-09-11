const RUNNING_STATES = new Set(["playing", "stepping", "continuing"]);

export function hasIndependentInference(snapshot) {
  return snapshot?.driver === "policy"
    && !snapshot?.trajectory?.imported
    && !["recording", "dataset", "trajectory"].includes(snapshot?.mode);
}

export function shouldPauseForInspection(state) {
  return state.hasControl
    && (!hasIndependentInference(state.liveSnapshot) || !state.replayingInspection)
    && state.inspectionPauseCommandId === null
    && state.liveSnapshot?.mode !== "recording"
    && RUNNING_STATES.has(state.liveSnapshot?.run_state);
}

// Play advances replay and unfinished inference; Pause stops both clocks.
export function createPlaybackTransport({
  state, command, renderSnapshot, inspectStep, setInspectionCursor, returnToLive,
  inspectionEpisodeSequences, invalidateRead,
  setTimeout: schedule = globalThis.setTimeout,
  clearTimeout: cancel = globalThis.clearTimeout,
  now = () => performance.now(),
}) {
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
      invalidateRead();
      state.seekingStep = null;
    }
    if (state.inspectionReplayTimer !== null) {
      cancel(state.inspectionReplayTimer);
      state.inspectionReplayTimer = null;
    }
    const wasReplaying = state.replayingInspection;
    state.replayingInspection = false;
    if (render && wasReplaying && state.snapshot) renderSnapshot();
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
      if (nextSequence === state.timelineSequences.at(-1)) returnToLive();
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
      renderSnapshot();
      scheduleInspectionReplay();
      return;
    }
    if (state.inspectionSequence !== null) returnToLive();
    if (!RUNNING_STATES.has(state.liveSnapshot?.run_state)
        || state.inspectionPauseCommandId !== null) command("play");
  }

  function pauseCurrentPlayback() {
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

  return { canReplayInspection, stopInspectionReplay, playFromCurrentPosition,
    pauseCurrentPlayback, playbackIsRunning };
}
