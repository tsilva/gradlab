import {
  FRAME_GAME,
  FRAME_OBSERVATION,
  PANEL_TYPES,
  panelDefinition,
  panelLabels,
  panelSubscriptions,
} from "./panels/catalog.js";
import { PanelManager } from "./panels/manager.js";
import { PanelRuntime } from "./panels/runtime.js";
import { text } from "./panels/shared.js";
import {
  bumpWorkspaceRevision,
  compareWorkspaceRevisions,
  createDefaultWorkspace,
  createTelemetryInstance,
  normalizePanelConfig,
  normalizeWorkspace,
} from "./panels/workspace.js";

const FRAME_HEADER_BYTES = 24;
const panelName = location.pathname.startsWith("/panel/")
  ? location.pathname.slice("/panel/".length)
  : null;
const workspaceWindowName = location.pathname.startsWith("/workspace/")
  ? location.pathname.slice("/workspace/".length)
  : null;
const pairedWorkspace = new URLSearchParams(location.search).get("workspace") === "paired";
const token = new URLSearchParams(location.hash.slice(1)).get("token") || "";
const WORKSPACE_ID_KEY = "rlab.player.workspace.id";
const LAYOUT_KEY = pairedWorkspace
  ? "rlab.player.workspace.v4.paired"
  : "rlab.player.workspace.v4.single";
const SAVED_LAYOUTS_KEY = "rlab.player.workspace.saved.v4";
const STATS_WINDOW_ID = "stats";
const workspaceId = localStorage.getItem(WORKSPACE_ID_KEY) || crypto.randomUUID();
localStorage.setItem(WORKSPACE_ID_KEY, workspaceId);
const windowId = panelName ? `panel-${panelName}` : (workspaceWindowName || "main");

function defaultLayout() {
  return createDefaultWorkspace({ paired: pairedWorkspace, writer: windowId });
}

const state = {
  socket: null,
  connected: false,
  clientId: null,
  snapshot: null,
  liveSnapshot: null,
  snapshots: new Map(),
  frameBlobs: new Map([[FRAME_GAME, new Map()], [FRAME_OBSERVATION, new Map()]]),
  inspectionSequence: null,
  replayingInspection: false,
  inspectionReplayTimer: null,
  inspectionPauseCommandId: null,
  timelineSequences: [],
  history: [],
  historyLimit: 4096,
  hasControl: false,
  frameSequence: new Map(),
  receivedFrameSequence: new Map(),
  retainedEpisode: null,
  pendingSnapshot: null,
  mode: null,
  lastStatus: null,
  actionNamesKey: "",
  workspaceId,
  windowId,
  layout: null,
  selectedPanel: null,
  activeWindows: new Map(),
  sessionEpoch: 0,
  sourceMode: false,
  applicationSnapshot: null,
  workspaceReady: false,
};
let panelRuntime = null;
let sourceBrowser = null;
let sourceBrowserPromise = null;
let gridStack = null;
let syncingGrid = false;
let panelManager = null;

const workspaceChannel = "BroadcastChannel" in window
  ? new BroadcastChannel(`rlab-player-${workspaceId}`)
  : null;

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, Number(value) || minimum));
}

function readStoredLayout() {
  try {
    return normalizeWorkspace(
      JSON.parse(localStorage.getItem(LAYOUT_KEY) || "null"),
      { paired: pairedWorkspace, writer: windowId },
    );
  } catch {
    return defaultLayout();
  }
}

function panelsInThisWindow() {
  if (!state.layout) return [];
  return Object.entries(state.layout.panels)
    .filter(([, panel]) => (
      panel.placement.visible && panel.placement.window === state.windowId
    ))
    .map(([name]) => name);
}

function subscriptions() {
  return panelSubscriptions(state.layout, panelsInThisWindow());
}

function setDetachedLayout() {
  const secondary = state.windowId !== "main";
  document.body.classList.toggle("secondary-window", secondary);
  document.body.classList.toggle(
    "stats-window",
    pairedWorkspace && state.windowId === STATS_WINDOW_ID,
  );
}

function showToast(message, error = false) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.style.borderColor = error ? "var(--red)" : "var(--cyan)";
  toast.classList.add("visible");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.remove("visible"), 3200);
}

function updateConnection(label, kind = "") {
  const badge = $("#connection-status");
  badge.textContent = label;
  badge.className = `sync-status ${kind}`.trim();
}

function resetSession(epoch) {
  state.sessionEpoch = Number(epoch) || 0;
  state.retainedEpisode = null;
  state.pendingSnapshot = null;
  state.inspectionSequence = null;
  state.inspectionPauseCommandId = null;
  state.liveSnapshot = null;
  state.snapshot = null;
  state.history = [];
  clearRetainedEpisode();
  stopInspectionReplay({ render: false });
}

async function ensureSourceBrowser() {
  if (sourceBrowser) return sourceBrowser;
  if (!sourceBrowserPromise) {
    sourceBrowserPromise = import("./sources/browser.js").then(({ SourceBrowser }) => {
      sourceBrowser = new SourceBrowser($("#source-browser"), $("#source-breadcrumbs"), {
        token,
        command,
        getState: () => state,
        showToast,
      });
      return sourceBrowser;
    });
  }
  return sourceBrowserPromise;
}

function setSourceMode(active, snapshot = null) {
  state.sourceMode = Boolean(active);
  document.body.classList.toggle("source-selection", state.sourceMode);
  $("#source-browser").hidden = !state.sourceMode;
  $("#page-title").hidden = state.sourceMode;
  $("#change-source").hidden = (
    state.sourceMode
    || !(snapshot?.app?.has_active_runner || state.liveSnapshot?.app?.has_active_runner)
  );
  if (!state.sourceMode) {
    sourceBrowser?.stop();
    if (!state.workspaceReady) {
      state.workspaceReady = true;
      void applyLayout();
    }
    updateLayoutTitle();
    return;
  }
  document.title = "Select checkpoint · rlab player";
  const expected = snapshot;
  void ensureSourceBrowser().then((browser) => {
    if (state.sourceMode && state.applicationSnapshot === expected) browser.render(expected);
  }).catch((error) => showToast(`Source browser failed: ${error.message || error}`, true));
}

function connect() {
  if (!token) {
    updateConnection("Missing session token", "error");
    showToast("Open the complete dashboard URL printed by rlab.", true);
    return;
  }
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${scheme}://${location.host}/ws`);
  state.socket = socket;
  socket.binaryType = "arraybuffer";
  updateConnection("Connecting", "warning");
  socket.addEventListener("open", () => {
    socket.send(JSON.stringify({
      type: "hello",
      token,
      subscriptions: subscriptions(),
      panel: panelName || "workspace",
      workspace_id: state.workspaceId,
      window_id: state.windowId,
    }));
  });
  socket.addEventListener("message", (event) => {
    if (typeof event.data === "string") handleMessage(JSON.parse(event.data));
    else handleFrame(event.data);
  });
  socket.addEventListener("close", () => {
    state.connected = false;
    state.hasControl = false;
    updateConnection("Disconnected", "error");
    updateControlState();
  });
  socket.addEventListener("error", () => updateConnection("Connection error", "error"));
}

function handleMessage(message) {
  if (message.type === "welcome") {
    state.connected = true;
    state.clientId = message.client_id;
    state.historyLimit = Math.max(1, Number(message.history_limit) || 4096);
    updateConnection("Synced", "");
    return;
  }
  if (message.type === "history") {
    if (
      message.session_epoch !== undefined
      && Number(message.session_epoch) !== state.sessionEpoch
    ) return;
    state.history = normalizedHistory(message.points);
    renderHistory();
    return;
  }
  if (message.type === "session_changed") {
    resetSession(message.session_epoch);
    return;
  }
  if (message.type === "snapshot") {
    const epoch = Number(message.session_epoch || 0);
    if (epoch !== state.sessionEpoch) resetSession(epoch);
    state.applicationSnapshot = message;
    state.hasControl = Boolean(message.control?.has_control);
    updateControlState();
    if (message.app && message.app.phase !== "active") {
      state.liveSnapshot = message;
      state.snapshot = message;
      setSourceMode(true, message);
      return;
    }
    setSourceMode(false, message);
    prepareRetainedEpisode(message);
    if (!requiredFramesAvailable(message)) {
      state.pendingSnapshot = message;
    } else {
      applySnapshot(message);
    }
    return;
  }
  if (message.type === "command_result") {
    if (message.id === state.inspectionPauseCommandId && !message.ok) {
      state.inspectionPauseCommandId = null;
    }
    if (!message.ok) showToast(message.error || "Command failed", true);
    return;
  }
  if (message.type === "error") showToast(message.error || "Player error", true);
}

function rememberFrame(kind, sequence, blob, preserveSequence = null) {
  const frames = state.frameBlobs.get(kind);
  frames.set(sequence, blob);
  while (frames.size > state.historyLimit) {
    const candidates = [...frames.keys()].filter(
      (candidate) => preserveSequence === null
        || Number(candidate) !== Number(preserveSequence),
    );
    if (!candidates.length) break;
    const oldest = Math.min(...candidates);
    frames.delete(oldest);
  }
}

async function handleFrame(buffer) {
  const view = new DataView(buffer);
  if (buffer.byteLength <= FRAME_HEADER_BYTES) return;
  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 4));
  if (magic !== "RLP2") return;
  const kind = view.getUint8(4);
  const epoch = Number(view.getBigUint64(8));
  const sequence = Number(view.getBigUint64(16));
  if (epoch !== state.sessionEpoch) return;
  if (sequence < (state.receivedFrameSequence.get(kind) ?? -1)) return;
  state.receivedFrameSequence.set(kind, sequence);
  const blob = new Blob([buffer.slice(FRAME_HEADER_BYTES)], { type: "image/png" });
  rememberFrame(kind, sequence, blob);
  if (
    state.inspectionSequence === sequence
    || (
      state.inspectionSequence === null
      && Number(state.liveSnapshot?.sequence) === sequence
    )
  ) {
    await panelRuntime.renderFrame(kind, blob);
  }
  state.frameSequence.set(kind, sequence);
  flushPendingSnapshot();
}

function requiredFrameKinds(snapshot) {
  const visible = new Set(
    panelsInThisWindow().flatMap(
      (id) => panelDefinition(state.layout, id)?.frameKinds || [],
    ),
  );
  const required = [];
  if (visible.has(FRAME_GAME) && snapshot.transition?.after?.game_frame) {
    required.push(FRAME_GAME);
  }
  if (
    visible.has(FRAME_OBSERVATION)
    && Number(snapshot.transition?.before?.observation_frames || 0) > 0
  ) required.push(FRAME_OBSERVATION);
  return required;
}

function requiredFramesAvailable(snapshot) {
  const sequence = Number(snapshot?.sequence);
  return requiredFrameKinds(snapshot).every(
    (kind) => state.frameBlobs.get(kind)?.has(sequence),
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
  const key = historyKey(point);
  const index = state.history.findIndex((candidate) => historyKey(candidate) === key);
  if (index >= 0) {
    state.history[index] = { ...state.history[index], ...point };
    return true;
  }
  state.history.push(point);
  state.history = normalizedHistory(state.history);
  return true;
}

function currentEpisodeHistory() {
  const episode = episodeForSnapshot(state.liveSnapshot) ?? state.retainedEpisode;
  if (episode === null) return state.history;
  return state.history.filter((point) => Number(point.episode) === episode);
}

function panelView() {
  return {
    history: currentEpisodeHistory(),
    inspection: state.inspectionSequence !== null,
    sessionEpoch: state.sessionEpoch,
    selectedSequence: state.inspectionSequence ?? state.snapshot?.sequence ?? null,
    liveSequence: state.liveSnapshot?.sequence ?? null,
  };
}

function pruneRetainedTrace(preserveSequence = null) {
  const sequences = [...state.snapshots.keys()].sort((a, b) => a - b);
  const remove = sequences
    .filter(
      (sequence) => preserveSequence === null
        || Number(sequence) !== Number(preserveSequence),
    )
    .slice(0, Math.max(0, sequences.length - state.historyLimit));
  remove.forEach((sequence) => {
    state.snapshots.delete(sequence);
    state.frameBlobs.forEach((frames) => frames.delete(sequence));
  });
  if (
    state.inspectionSequence !== null
    && !state.snapshots.has(Number(state.inspectionSequence))
  ) {
    stopInspectionReplay({ render: false });
    state.inspectionSequence = null;
    state.snapshot = state.liveSnapshot;
    showToast("The selected transition expired from the bounded history.", true);
    broadcastInspection(null);
  }
}

function clearRetainedEpisode() {
  state.snapshots.clear();
  state.frameBlobs.forEach((frames) => frames.clear());
  state.frameSequence.clear();
  state.receivedFrameSequence.clear();
  state.timelineSequences = [];
}

function prepareRetainedEpisode(snapshot) {
  const episode = episodeForSnapshot(snapshot);
  if (episode === null) return;
  if (state.retainedEpisode !== null && state.retainedEpisode !== episode) {
    clearRetainedEpisode();
  }
  state.retainedEpisode = episode;
}

function applySnapshot(snapshot) {
  state.pendingSnapshot = null;
  const previousEnvironmentId = state.liveSnapshot?.session?.env_id;
  const previousEpisode = episodeForSnapshot(state.liveSnapshot);
  const nextEpisode = episodeForSnapshot(snapshot);
  const episodeChanged = (
    previousEpisode !== null
    && nextEpisode !== null
    && previousEpisode !== nextEpisode
  );
  state.liveSnapshot = snapshot;
  if (snapshot.run_state === "paused") state.inspectionPauseCommandId = null;
  if (snapshot.session?.env_id !== previousEnvironmentId) updateLayoutTitle();
  if (state.inspectionSequence !== null && episodeChanged) {
    stopInspectionReplay({ render: false });
    state.inspectionSequence = null;
    state.snapshot = snapshot;
  }
  state.snapshots.set(Number(snapshot.sequence), snapshot);
  pruneRetainedTrace();
  state.hasControl = Boolean(snapshot.control?.has_control);
  const historyChanged = snapshot.history_point
    ? ingestHistoryPoint(snapshot.history_point)
    : false;
  if (state.inspectionSequence === null) {
    state.snapshot = snapshot;
    renderSnapshot();
    void showFramesForSequence(Number(snapshot.sequence));
  } else {
    panelRuntime.invoke("controls", "render", snapshot);
    updateControlState();
    renderWorkspaceStatus();
    renderTimeline();
  }
  if (historyChanged) renderHistory();
}

function flushPendingSnapshot() {
  const snapshot = state.pendingSnapshot;
  if (!snapshot) return;
  if (requiredFramesAvailable(snapshot)) applySnapshot(snapshot);
}

function send(value) {
  if (state.socket?.readyState === WebSocket.OPEN) state.socket.send(JSON.stringify(value));
}

function command(name, payload = {}) {
  if (!state.hasControl) {
    showToast("This window is an observer. Choose Control here first.", true);
    return null;
  }
  const id = crypto.randomUUID();
  send({
    type: "command",
    id,
    name,
    payload,
    expected_revision: state.liveSnapshot?.revision ?? null,
  });
  return id;
}

function inspectionEpisodeSequences() {
  if (state.inspectionSequence === null) return [];
  const selected = state.snapshots.get(Number(state.inspectionSequence));
  const episode = episodeForSnapshot(selected);
  if (episode === null) return [];
  return state.timelineSequences.filter(
    (sequence) => episodeForSnapshot(state.snapshots.get(Number(sequence))) === episode,
  );
}

function canReplayInspection() {
  if (state.liveSnapshot?.run_state !== "paused") return false;
  const sequences = inspectionEpisodeSequences();
  const selectedIndex = sequences.indexOf(Number(state.inspectionSequence));
  return selectedIndex >= 0 && selectedIndex < sequences.length - 1;
}

function stopInspectionReplay({ render = true } = {}) {
  if (state.inspectionReplayTimer !== null) {
    window.clearTimeout(state.inspectionReplayTimer);
    state.inspectionReplayTimer = null;
  }
  const wasReplaying = state.replayingInspection;
  state.replayingInspection = false;
  if (render && wasReplaying && state.snapshot) renderSnapshot();
}

function inspectionReplayDelay() {
  const fps = Number(state.liveSnapshot?.session?.target_fps || 0);
  return fps > 0 ? 1000 / fps : 0;
}

function scheduleInspectionReplay() {
  state.inspectionReplayTimer = window.setTimeout(() => {
    state.inspectionReplayTimer = null;
    if (!state.replayingInspection) return;
    const sequences = inspectionEpisodeSequences();
    const selectedIndex = sequences.indexOf(Number(state.inspectionSequence));
    const nextSequence = sequences[selectedIndex + 1];
    if (selectedIndex < 0 || nextSequence === undefined) {
      stopInspectionReplay();
      return;
    }
    const reachedEpisodeEnd = selectedIndex + 1 === sequences.length - 1;
    if (reachedEpisodeEnd) state.replayingInspection = false;
    if (nextSequence === state.timelineSequences.at(-1)) returnToLive();
    else setInspectionCursor(nextSequence, { preserveReplay: true });
    if (!reachedEpisodeEnd) scheduleInspectionReplay();
  }, inspectionReplayDelay());
}

function playFromCurrentPosition() {
  if (canReplayInspection()) {
    state.replayingInspection = true;
    renderSnapshot();
    scheduleInspectionReplay();
    return;
  }
  command("play");
}

function pauseCurrentPlayback() {
  if (state.replayingInspection) {
    stopInspectionReplay();
    return;
  }
  command("pause");
}

function updateControlState() {
  const control = $("#control-status");
  control.textContent = state.hasControl ? "Controller" : "Observer";
  control.className = `badge ${state.hasControl ? "" : "muted"}`.trim();
  panelRuntime?.invoke("controls", "updateControl");
}

function renderWorkspaceStatus() {
  const live = state.liveSnapshot;
  const shown = state.snapshot || live;
  const samplingMode = live?.session?.sampling_mode || "stochastic";
  const samplingStatus = $("#sampling-status");
  samplingStatus.hidden = ["recording", "dataset"].includes(live?.mode);
  samplingStatus.textContent = samplingMode === "deterministic" ? "Deterministic" : "Stochastic";
  samplingStatus.className = `badge ${samplingMode === "deterministic" ? "warning" : "muted"}`;
  const timelineContext = [
    state.inspectionSequence === null ? null : "INSPECTING",
    `STEP ${text(shown?.session?.step)}`,
    `SEQ ${text(shown?.sequence)}`,
    state.inspectionSequence === null ? null : `LIVE ${text(live?.sequence)}`,
  ];
  $("#timeline-label").textContent = timelineContext.filter(Boolean).join(" · ");
}

function renderSnapshot() {
  const snapshot = state.snapshot;
  const session = snapshot.session || {};
  configureMode(snapshot.mode || "playback");
  updateControlState();
  renderWorkspaceStatus();
  const actionNamesKey = JSON.stringify(session.action_names || []);
  if (actionNamesKey !== state.actionNamesKey) {
    state.actionNamesKey = actionNamesKey;
    renderHistory();
  }
  if (state.inspectionSequence === null && snapshot.status_message && snapshot.status_message !== state.lastStatus) {
    state.lastStatus = snapshot.status_message;
    showToast(snapshot.status_message, snapshot.run_state === "paused" && /error|expired|unsupported|no configured/i.test(snapshot.status_message));
  }
  panelRuntime.renderSnapshot(snapshot, panelView());
  renderTimeline();
}

function configureMode(mode) {
  if (state.mode === mode) return;
  state.mode = mode;
  const recording = mode === "recording";
  const dataset = mode === "dataset";
  document.body.classList.toggle("recording", recording);
  document.querySelector(".eyebrow").textContent = recording
    ? "HUMAN RECORDING"
    : (dataset ? "DATASET PLAYBACK" : "RLAB PLAYER");
}

function renderHistory() {
  panelRuntime.renderHistory(currentEpisodeHistory(), state.snapshot, panelView());
  renderTimeline();
}

function exactFrameBlob(kind, sequence) {
  return state.frameBlobs.get(kind)?.get(Number(sequence)) || null;
}

async function showFramesForSequence(sequence) {
  const kinds = [...new Set(
    panelsInThisWindow().flatMap(
      (id) => panelDefinition(state.layout, id)?.frameKinds || [],
    ),
  )];
  const missing = [];
  await Promise.all(kinds.map(async (kind) => {
    const blob = exactFrameBlob(kind, sequence);
    if (!blob) missing.push(kind);
    await panelRuntime.renderFrame(kind, blob);
  }));
  return missing;
}

function inspectionFrames(sequence) {
  return [FRAME_GAME, FRAME_OBSERVATION]
    .map((kind) => ({ kind, blob: exactFrameBlob(kind, sequence) }))
    .filter((item) => item.blob);
}

function broadcastInspection(sequence) {
  if (!workspaceChannel) return;
  if (sequence === null) {
    workspaceChannel.postMessage({
      type: "inspection-cursor",
      session_epoch: state.sessionEpoch,
      episode: episodeForSnapshot(state.liveSnapshot),
      sequence: null,
      source: state.windowId,
    });
    return;
  }
  const snapshot = state.snapshots.get(Number(sequence)) || state.snapshot;
  workspaceChannel.postMessage({
    type: "inspection-cursor",
    session_epoch: state.sessionEpoch,
    episode: episodeForSnapshot(snapshot),
    sequence: Number(sequence),
    snapshot,
    frames: inspectionFrames(sequence),
    source: state.windowId,
  });
}

function requestInspectionFrames(sequence, kinds) {
  if (!workspaceChannel || !kinds.length) return;
  workspaceChannel.postMessage({
    type: "inspection-frame-request",
    session_epoch: state.sessionEpoch,
    sequence: Number(sequence),
    kinds,
    source: state.windowId,
  });
}

function maybePauseForInspection() {
  if (
    !state.hasControl
    || state.inspectionPauseCommandId !== null
    || state.liveSnapshot?.mode === "recording"
    || !["playing", "stepping", "continuing"].includes(state.liveSnapshot?.run_state)
  ) return;
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
    showToast("That transition is not retained in this window.", true);
    return;
  }
  if (
    Number(snapshot.session_epoch || 0) !== state.sessionEpoch
    || (
      state.retainedEpisode !== null
      && episodeForSnapshot(snapshot) !== state.retainedEpisode
    )
  ) return;
  frames.forEach(({ kind, blob }) => {
    if ([FRAME_GAME, FRAME_OBSERVATION].includes(Number(kind)) && blob instanceof Blob) {
      rememberFrame(Number(kind), numericSequence, blob, numericSequence);
    }
  });
  state.snapshots.set(numericSequence, snapshot);
  pruneRetainedTrace(numericSequence);
  if (!preserveReplay) stopInspectionReplay({ render: false });
  if (announce) maybePauseForInspection();
  state.inspectionSequence = numericSequence;
  state.snapshot = snapshot;
  renderSnapshot();
  renderHistory();
  void showFramesForSequence(numericSequence).then((missing) => {
    if (!missing.length) return;
    requestInspectionFrames(numericSequence, missing);
  });
  if (announce) broadcastInspection(numericSequence);
}

function inspectSequence(sequence) {
  setInspectionCursor(sequence);
}

function returnToLive({ announce = true } = {}) {
  stopInspectionReplay({ render: false });
  state.inspectionSequence = null;
  state.snapshot = state.liveSnapshot;
  if (state.snapshot) {
    renderSnapshot();
    renderHistory();
    void showFramesForSequence(Number(state.snapshot.sequence));
  }
  if (announce) broadcastInspection(null);
}

function renderTimeline() {
  const scrubber = $("#timeline-scrubber");
  if (!scrubber) return;
  const currentEpisode = episodeForSnapshot(state.liveSnapshot);
  state.timelineSequences = [...state.snapshots.entries()]
    .filter(([, snapshot]) => (
      currentEpisode === null || episodeForSnapshot(snapshot) === currentEpisode
    ))
    .map(([sequence]) => Number(sequence))
    .sort((a, b) => a - b);
  const sequences = state.timelineSequences;
  scrubber.min = "0";
  scrubber.max = String(Math.max(0, sequences.length - 1));
  scrubber.step = "1";
  scrubber.disabled = sequences.length < 2;
  const selected = state.inspectionSequence ?? sequences.at(-1);
  const selectedIndex = sequences.indexOf(selected);
  scrubber.value = String(selectedIndex < 0 ? Math.max(0, sequences.length - 1) : selectedIndex);
  renderWorkspaceStatus();

  const markers = $("#timeline-markers");
  if (!sequences.length) { markers.replaceChildren(); return; }
  const minimum = sequences[0];
  const maximum = sequences.at(-1);
  const range = Math.max(1, maximum - minimum);
  const interesting = currentEpisodeHistory().filter((point) =>
    Number(point.sequence) >= minimum
    && Number(point.sequence) <= maximum
    && (currentEpisode === null || Number(point.episode) === currentEpisode)
    && (point.boundary || point.events?.length)
  );
  markers.replaceChildren(...interesting.slice(-120).map((point) => {
    const marker = document.createElement("span");
    marker.className = "timeline-marker";
    marker.style.left = `${((Number(point.sequence) - minimum) / range) * 100}%`;
    marker.style.setProperty("--marker-color", point.boundary ? "var(--red)" : "var(--magenta)");
    return marker;
  }));
}

function maxPanelRow(targetWindow = state.windowId) {
  return Math.max(0, ...Object.values(state.layout.panels)
    .filter((panel) => (
      panel.placement.visible && panel.placement.window === targetWindow
    ))
    .map((panel) => panel.placement.y + panel.placement.h));
}

function panelLabel(id) {
  return panelLabels(state.layout)[id] || id;
}

function gridWidgetFor(name, placement = state.layout.panels[name]?.placement) {
  const definition = panelDefinition(state.layout, name);
  const minimum = definition?.minimum || PANEL_TYPES.telemetry.minimum;
  return {
    id: name,
    x: placement.x,
    y: placement.y,
    w: placement.w,
    h: placement.h,
    minW: minimum.w,
    minH: minimum.h,
    maxH: 40,
  };
}

function syncGridNodes(nodes = null) {
  const current = nodes || $$(".grid-stack-item")
    .map((item) => item.gridstackNode)
    .filter(Boolean);
  current.forEach((node) => {
    const name = node.el?.dataset.panel;
    const placement = state.layout.panels[name]?.placement;
    if (!placement) return;
    placement.x = Number(node.x || 0);
    placement.y = Number(node.y || 0);
    placement.w = Number(node.w || placement.w);
    placement.h = Number(node.h || placement.h);
  });
}

function persistLayout({ announce = true } = {}) {
  bumpWorkspaceRevision(state.layout, state.windowId);
  localStorage.setItem(LAYOUT_KEY, JSON.stringify(state.layout));
  if (announce) workspaceChannel?.postMessage({ type: "layout", layout: state.layout, source: state.windowId });
}

function updateLayoutTitle() {
  const environmentId = String(state.liveSnapshot?.session?.env_id || "").trim();
  const environmentTitle = environmentId || "Environment";
  const title = panelName
    ? `${environmentTitle} · ${panelLabel(panelName)}`
    : pairedWorkspace && state.windowId === STATS_WINDOW_ID
      ? `${environmentTitle} · Stats`
      : environmentTitle;
  $("#page-title").textContent = title;
  $("#layout-name-input").value = state.layout.name;
  document.title = `${title} · rlab player`;
}

async function applyLayout() {
  const visibleHere = panelsInThisWindow();
  document.body.classList.toggle("empty-workspace", visibleHere.length === 0);
  updateLayoutTitle();
  panelManager?.renderShelf();
  renderSavedLayouts();
  send({ type: "subscribe", subscriptions: subscriptions() });
  syncingGrid = true;
  gridStack.batchUpdate();
  try {
    await panelRuntime.sync(state.layout, state.windowId);
  } finally {
    gridStack.batchUpdate(false);
    syncingGrid = false;
  }
  syncGridNodes();
  if (state.snapshot) {
    panelRuntime.renderSnapshot(state.snapshot, panelView());
    panelRuntime.renderHistory(currentEpisodeHistory(), state.snapshot, panelView());
    const sequence = Number(state.snapshot.sequence);
    const visibleKinds = new Set(visibleHere.flatMap(
      (id) => panelDefinition(state.layout, id)?.frameKinds || [],
    ));
    if (visibleKinds.has(FRAME_GAME)) {
      panelRuntime.renderFrame(FRAME_GAME, exactFrameBlob(FRAME_GAME, sequence));
    }
    if (visibleKinds.has(FRAME_OBSERVATION)) {
      panelRuntime.renderFrame(
        FRAME_OBSERVATION,
        exactFrameBlob(FRAME_OBSERVATION, sequence),
      );
    }
  }
  requestAnimationFrame(() => panelRuntime.resize());
}

function readSavedLayouts() {
  try {
    const value = JSON.parse(localStorage.getItem(SAVED_LAYOUTS_KEY) || "{}");
    return value && typeof value === "object" ? value : {};
  } catch {
    return {};
  }
}

function renderSavedLayouts() {
  const target = $("#saved-layouts");
  const saved = readSavedLayouts();
  const rows = Object.keys(saved).sort().map((name) => {
    const row = document.createElement("div");
    row.className = "saved-layout-row";
    const load = document.createElement("button");
    load.type = "button";
    load.className = "quiet";
    load.textContent = name;
    load.title = `Load layout ${name}`;
    load.addEventListener("click", () => {
      state.layout = normalizeWorkspace(saved[name], {
        paired: pairedWorkspace,
        writer: state.windowId,
      });
      state.layout.name = name;
      persistLayout();
      applyLayout();
      $("#layout-menu").hidden = true;
      showToast(`Loaded layout “${name}”.`);
    });
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "quiet danger";
    remove.textContent = "Delete";
    remove.title = `Delete layout ${name}`;
    remove.addEventListener("click", () => {
      const next = readSavedLayouts();
      delete next[name];
      localStorage.setItem(SAVED_LAYOUTS_KEY, JSON.stringify(next));
      renderSavedLayouts();
    });
    row.append(load, remove);
    return row;
  });
  if (!rows.length) {
    const empty = document.createElement("span");
    empty.className = "empty-state";
    empty.textContent = "No named layouts saved yet.";
    target.replaceChildren(empty);
  } else target.replaceChildren(...rows);
}

function renderPanelShelf() {
  panelManager?.renderShelf();
}

function bindPanelElement(panel, name) {
  const handle = panel.querySelector("[data-drag-handle]");
  if (handle) {
    handle.draggable = false;
    handle.addEventListener("keydown", (event) => {
      if (!event.altKey || !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) return;
      event.preventDefault();
      const placement = state.layout.panels[name].placement;
      const minimum = panelDefinition(state.layout, name).minimum;
      const amount = event.key === "ArrowLeft" || event.key === "ArrowUp" ? -1 : 1;
      if (event.shiftKey) {
        if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
          placement.w = clamp(placement.w + amount, minimum.w, 12 - placement.x);
        } else placement.h = clamp(placement.h + amount, minimum.h, 40);
      } else if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
        placement.x = clamp(placement.x + amount, 0, 12 - placement.w);
      } else placement.y = clamp(placement.y + amount, 0, 199);
      gridStack.update(panel.closest(".grid-stack-item"), {
        x: placement.x,
        y: placement.y,
        w: placement.w,
        h: placement.h,
      });
      syncGridNodes();
      persistLayout();
      requestAnimationFrame(() => panelRuntime.resize());
    });
  }
  const menu = panel.querySelector("[data-panel-menu]");
  menu?.addEventListener("click", (event) => {
    event.stopPropagation();
    openPanelMenu(name, menu);
  });
}

function bindPanelLayout() {
  gridStack = window.GridStack.init({
    animate: false,
    cellHeight: 32,
    column: 12,
    draggable: { handle: ".panel-drag", scroll: true },
    float: false,
    margin: 5,
    maxRow: 200,
    minRow: 8,
    resizable: { handles: "se" },
  }, $("#dashboard"));
  gridStack.on("change", (_event, nodes) => {
    if (!syncingGrid) syncGridNodes(nodes);
  });
  gridStack.on("resizestop", (_event, item) => {
    syncGridNodes([item.gridstackNode]);
    persistLayout();
    panelRuntime.resize();
    showToast(`${panelLabel(item.dataset.panel)} resized.`);
  });
  gridStack.on("dragstop", (_event, item) => {
    syncGridNodes([item.gridstackNode]);
    persistLayout();
    panelRuntime.resize();
    showToast(`${panelLabel(item.dataset.panel)} moved.`);
  });
}

function positionMenu(menu, anchor) {
  const rect = anchor.getBoundingClientRect();
  menu.hidden = false;
  const width = menu.offsetWidth || 304;
  menu.style.left = `${Math.max(8, Math.min(window.innerWidth - width - 8, rect.right - width))}px`;
  menu.style.top = `${Math.min(window.innerHeight - menu.offsetHeight - 8, rect.bottom + 6)}px`;
}

function openPanelMenu(name, anchor) {
  state.selectedPanel = name;
  const instance = state.layout.panels[name];
  $("#panel-menu-title").textContent = panelLabel(name);
  $("#panel-dock-main").hidden = state.windowId === "main";
  $("#panel-edit").hidden = instance?.type !== "telemetry";
  $("#panel-duplicate").hidden = instance?.type !== "telemetry";
  $("#panel-remove").hidden = Boolean(instance?.builtin);
  positionMenu($("#panel-menu"), anchor);
}

function windowUrl(targetWindow) {
  return `${location.origin}/workspace/${encodeURIComponent(targetWindow)}${location.search}#token=${encodeURIComponent(token)}`;
}

function movePanelToNewWindow(name) {
  const targetWindow = `window-${crypto.randomUUID().slice(0, 8)}`;
  const popup = window.open(windowUrl(targetWindow), `rlab-${targetWindow}`, "popup");
  if (!popup) { showToast("The browser blocked the new workspace window.", true); return; }
  const placement = state.layout.panels[name].placement;
  placement.window = targetWindow;
  placement.visible = true;
  placement.x = 0;
  placement.y = 0;
  persistLayout();
  applyLayout();
  showToast(`${panelLabel(name)} moved to a synchronized window.`);
}

function revealPanel(name) {
  const panel = state.layout.panels[name];
  if (!panel) return;
  const windowHasPanels = Object.entries(state.layout.panels).some(
    ([otherName, candidate]) => (
      otherName !== name
      && candidate.placement.visible
      && candidate.placement.window === state.windowId
    ),
  );
  panel.placement.visible = true;
  panel.placement.window = state.windowId;
  panel.placement.x = 0;
  panel.placement.y = windowHasPanels ? maxPanelRow() : 0;
  persistLayout();
  void applyLayout();
  $("#panel-shelf").hidden = true;
  $("#panels-toggle").setAttribute("aria-expanded", "false");
  showToast(`${panel.title} moved into this window.`);
}

function createTelemetryPanel({ title, config }) {
  const id = `panel-${crypto.randomUUID()}`;
  const panel = createTelemetryInstance({
    id,
    title,
    config,
    window: state.windowId,
    y: maxPanelRow(),
  });
  if (!panel) {
    showToast("The telemetry panel could not be created.", true);
    return;
  }
  state.layout.panels[id] = panel;
  persistLayout();
  void applyLayout();
  showToast(`${panel.title} added.`);
}

function updateTelemetryPanel(name, { title, config }) {
  const panel = state.layout.panels[name];
  if (panel?.type !== "telemetry") return;
  panel.title = title.trim().slice(0, 80);
  panel.config = normalizePanelConfig(config);
  persistLayout();
  void applyLayout();
  showToast(`${panel.title} updated.`);
}

function duplicateTelemetryPanel(name) {
  const source = state.layout.panels[name];
  if (source?.type !== "telemetry") return;
  createTelemetryPanel({
    title: `${source.title} copy`,
    config: structuredClone(source.config),
  });
}

function removeTelemetryPanel(name) {
  const panel = state.layout.panels[name];
  if (!panel || panel.builtin) return;
  const label = panel.title;
  delete state.layout.panels[name];
  state.selectedPanel = null;
  persistLayout();
  void applyLayout();
  showToast(`${label} removed.`);
}

function bindWorkspaceMenus() {
  $("#player-home").addEventListener("click", () => {
    void ensureSourceBrowser()
      .then((browser) => browser.goHome())
      .catch((error) => showToast(`Source browser failed: ${error.message || error}`, true));
  });
  $("#change-source").addEventListener("click", () => {
    void ensureSourceBrowser()
      .then((browser) => browser.browseCurrentSource())
      .catch((error) => showToast(`Source browser failed: ${error.message || error}`, true));
  });
  $("#layouts-toggle").addEventListener("click", (event) => {
    $("#panel-shelf").hidden = true;
    $("#panels-toggle").setAttribute("aria-expanded", "false");
    positionMenu($("#layout-menu"), event.currentTarget);
  });
  $("#save-layout").addEventListener("click", () => {
    const name = $("#layout-name-input").value.trim().slice(0, 48) || "Workspace";
    state.layout.name = name;
    const saved = readSavedLayouts();
    saved[name] = state.layout;
    localStorage.setItem(SAVED_LAYOUTS_KEY, JSON.stringify(saved));
    persistLayout();
    applyLayout();
    $("#layout-menu").hidden = true;
    showToast(`Layout “${name}” saved.`);
  });
  $("#reset-layout").addEventListener("click", () => {
    state.layout = defaultLayout();
    persistLayout();
    applyLayout();
    $("#layout-menu").hidden = true;
    showToast("Default research layout restored.");
  });
  $("#panel-new-window").addEventListener("click", () => {
    if (state.selectedPanel) movePanelToNewWindow(state.selectedPanel);
    $("#panel-menu").hidden = true;
  });
  $("#panel-dock-main").addEventListener("click", () => {
    const name = state.selectedPanel;
    if (!name) return;
    const placement = state.layout.panels[name].placement;
    placement.window = "main";
    placement.visible = true;
    placement.x = 0;
    placement.y = maxPanelRow("main");
    persistLayout();
    applyLayout();
    $("#panel-menu").hidden = true;
    showToast(`${panelLabel(name)} docked to the main window.`);
    if (state.windowId !== "main" && !panelsInThisWindow().length) setTimeout(() => window.close(), 250);
  });
  $("#panel-edit").addEventListener("click", () => {
    if (state.selectedPanel) panelManager.openEditor(state.selectedPanel);
    $("#panel-menu").hidden = true;
  });
  $("#panel-duplicate").addEventListener("click", () => {
    if (state.selectedPanel) panelManager.duplicate(state.selectedPanel);
    $("#panel-menu").hidden = true;
  });
  $("#panel-hide").addEventListener("click", () => {
    const name = state.selectedPanel;
    if (!name) return;
    state.layout.panels[name].placement.visible = false;
    persistLayout();
    applyLayout();
    $("#panel-menu").hidden = true;
    showToast(`${panelLabel(name)} moved to the panel shelf.`);
  });
  $("#panel-reset-size").addEventListener("click", () => {
    const name = state.selectedPanel;
    if (!name) return;
    const placement = state.layout.panels[name].placement;
    const defaults = defaultLayout().panels[name]?.placement || { w: 4, h: 8 };
    Object.assign(placement, { w: defaults.w, h: defaults.h });
    placement.x = clamp(placement.x, 0, 12 - defaults.w);
    persistLayout();
    applyLayout();
    $("#panel-menu").hidden = true;
  });
  $("#panel-remove").addEventListener("click", () => {
    if (state.selectedPanel) panelManager.remove(state.selectedPanel);
    $("#panel-menu").hidden = true;
  });
  $("#panels-toggle").addEventListener("click", (event) => {
    const shelf = $("#panel-shelf");
    const opening = shelf.hidden;
    $("#layout-menu").hidden = true;
    shelf.hidden = true;
    event.currentTarget.setAttribute("aria-expanded", String(opening));
    if (opening) {
      renderPanelShelf();
      positionMenu(shelf, event.currentTarget);
    }
  });
  $("#new-window").addEventListener("click", () => {
    const targetWindow = `window-${crypto.randomUUID().slice(0, 8)}`;
    const popup = window.open(windowUrl(targetWindow), `rlab-${targetWindow}`, "popup");
    if (!popup) showToast("The browser blocked the new workspace window.", true);
  });
  document.addEventListener("click", (event) => {
    if (!$("#panel-menu").contains(event.target) && !event.target.closest("[data-panel-menu]")) $("#panel-menu").hidden = true;
    if (!$("#layout-menu").contains(event.target) && !event.target.closest("#layouts-toggle")) $("#layout-menu").hidden = true;
    if (!$("#panel-shelf").contains(event.target) && !event.target.closest("#panels-toggle")) {
      $("#panel-shelf").hidden = true;
      $("#panels-toggle").setAttribute("aria-expanded", "false");
    }
  });
}

function reclaimWindow(closedWindow) {
  if (state.windowId !== "main") return;
  if (pairedWorkspace && closedWindow === STATS_WINDOW_ID) return;
  let changed = false;
  Object.values(state.layout.panels).forEach((panel) => {
    if (panel.placement.visible && panel.placement.window === closedWindow) {
      panel.placement.window = "main";
      panel.placement.y = maxPanelRow("main");
      changed = true;
    }
  });
  if (changed) {
    persistLayout();
    applyLayout();
    showToast("Panels from a closed window returned to the main workspace.");
  }
}

function bindWorkspaceSync() {
  if (workspaceChannel) {
    workspaceChannel.addEventListener("message", (event) => {
      const message = event.data || {};
      if (message.type === "layout" && message.source !== state.windowId) {
        const next = normalizeWorkspace(message.layout, {
          paired: pairedWorkspace,
          writer: state.windowId,
        });
        if (compareWorkspaceRevisions(next.revision, state.layout.revision) > 0) {
          state.layout = next;
          applyLayout();
        }
      } else if (message.type === "heartbeat") {
        state.activeWindows.set(message.window, Date.now());
      } else if (
        message.type === "inspection-cursor"
        && message.source !== state.windowId
        && Number(message.session_epoch || 0) === state.sessionEpoch
      ) {
        if (message.sequence === null) {
          returnToLive({ announce: false });
        } else if (
          message.snapshot
          && Number(message.episode) === episodeForSnapshot(message.snapshot)
        ) {
          setInspectionCursor(Number(message.sequence), {
            announce: false,
            snapshot: message.snapshot,
            frames: Array.isArray(message.frames) ? message.frames : [],
          });
        }
      } else if (
        message.type === "inspection-frame-request"
        && message.source !== state.windowId
        && Number(message.session_epoch || 0) === state.sessionEpoch
      ) {
        (Array.isArray(message.kinds) ? message.kinds : []).forEach((kind) => {
          const numericKind = Number(kind);
          const blob = exactFrameBlob(numericKind, Number(message.sequence));
          if (!blob) return;
          workspaceChannel.postMessage({
            type: "inspection-frame",
            session_epoch: state.sessionEpoch,
            sequence: Number(message.sequence),
            kind: numericKind,
            blob,
            source: state.windowId,
            target: message.source,
          });
        });
      } else if (
        message.type === "inspection-frame"
        && message.target === state.windowId
        && Number(message.session_epoch || 0) === state.sessionEpoch
        && Number(message.sequence) === Number(state.inspectionSequence)
        && [FRAME_GAME, FRAME_OBSERVATION].includes(Number(message.kind))
        && message.blob instanceof Blob
      ) {
        rememberFrame(
          Number(message.kind),
          Number(message.sequence),
          message.blob,
          Number(message.sequence),
        );
        void panelRuntime.renderFrame(Number(message.kind), message.blob);
      } else if (message.type === "window-closing" && state.windowId === "main") {
        setTimeout(() => {
          const lastSeen = state.activeWindows.get(message.window) || 0;
          if (Date.now() - lastSeen > 1800) reclaimWindow(message.window);
        }, 2000);
      }
    });
  }
  window.addEventListener("storage", (event) => {
    if (event.key !== LAYOUT_KEY || !event.newValue) return;
    try {
      const next = normalizeWorkspace(JSON.parse(event.newValue), {
        paired: pairedWorkspace,
        writer: state.windowId,
      });
      if (compareWorkspaceRevisions(next.revision, state.layout.revision) > 0) {
        state.layout = next;
        applyLayout();
      }
    } catch { /* Ignore malformed local data. */ }
  });
  const heartbeat = () => workspaceChannel?.postMessage({ type: "heartbeat", window: state.windowId });
  heartbeat();
  setInterval(heartbeat, 1000);
  window.addEventListener("beforeunload", () => {
    workspaceChannel?.postMessage({ type: "window-closing", window: state.windowId });
  });
}

function bindTimeline() {
  const scrubber = $("#timeline-scrubber");
  const selectIndex = (index) => {
    stopInspectionReplay({ render: false });
    const sequence = state.timelineSequences[index];
    if (sequence === undefined) return;
    if (index === state.timelineSequences.length - 1) returnToLive();
    else inspectSequence(sequence);
  };
  scrubber.addEventListener("input", (event) => {
    selectIndex(Number(event.target.value));
  });
  scrubber.addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      event.preventDefault();
      const direction = event.key === "ArrowLeft" ? -1 : 1;
      const index = clamp(
        Number(scrubber.value) + direction,
        0,
        Math.max(0, state.timelineSequences.length - 1),
      );
      scrubber.value = String(index);
      selectIndex(index);
      return;
    }
    if (event.code !== "Space" || event.repeat) return;
    event.preventDefault();
    const running = state.replayingInspection
      || ["playing", "stepping", "continuing"].includes(
        state.liveSnapshot?.run_state,
      );
    if (running) pauseCurrentPlayback();
    else playFromCurrentPosition();
  });
}

function initWorkspace() {
  state.layout = readStoredLayout();
  if (panelName && state.layout.panels[panelName]) {
    state.layout.panels[panelName].placement.visible = true;
    state.layout.panels[panelName].placement.window = state.windowId;
    persistLayout();
  }
  panelManager = new PanelManager({
    getWorkspace: () => state.layout,
    getContext: () => ({
      snapshot: state.snapshot,
      history: currentEpisodeHistory(),
    }),
    getWindowId: () => state.windowId,
    onReveal: revealPanel,
    onCreate: createTelemetryPanel,
    onUpdate: updateTelemetryPanel,
    onDuplicate: duplicateTelemetryPanel,
    onRemove: removeTelemetryPanel,
    showToast,
  });
  setDetachedLayout();
  bindPanelLayout();
  bindWorkspaceMenus();
  bindWorkspaceSync();
  bindTimeline();
}

panelRuntime = new PanelRuntime({
  definitionFor: panelDefinition,
  container: $("#dashboard"),
  services: {
    getState: () => state,
    send,
    command,
    canReplayInspection,
    playFromCurrentPosition,
    pauseCurrentPlayback,
    showToast,
    updatePanelConfig: (name, config) => {
      const panel = state.layout.panels[name];
      if (panel?.type === "telemetry") {
        updateTelemetryPanel(name, { title: panel.title, config });
      }
    },
  },
  onMount: (panel, name, _definition, gridItem) => {
    gridStack.makeWidget(gridItem, gridWidgetFor(name));
    const resizeHandle = gridItem.querySelector(".ui-resizable-se");
    if (resizeHandle) {
      resizeHandle.setAttribute("aria-label", `Resize ${panelLabel(name)}`);
      resizeHandle.title = resizeHandle.getAttribute("aria-label");
    }
    bindPanelElement(panel, name);
  },
  onLayout: (_panel, name, placement, gridItem) => {
    gridStack.update(gridItem, gridWidgetFor(name, placement));
  },
  onUnmount: (_panel, _name, gridItem) => {
    gridStack.removeWidget(gridItem, false, false);
  },
  onError: (name, error) => {
    console.error(`Panel ${name} failed`, error);
    showToast(`${panelLabel(name)} panel failed to load.`, true);
  },
});

window.addEventListener("resize", () => panelRuntime.resize());
initWorkspace();
updateControlState();
connect();
