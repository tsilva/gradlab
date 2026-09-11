import { frameScheduler } from "./chart-transport.js";
import { createPlaybackTransport, hasIndependentInference, shouldPauseForInspection } from "./playback-transport.js";
import { rewardReferenceStore } from "./panels/reward-reference.js";
import { createChartHistory } from "./chart-history.js";
import { usesChartHistory } from "./panels/chart-status.js";
import { bindTimelineRange } from "./chart-range.js";
import {
  FRAME_ATTRIBUTION,
  FRAME_CNN_INSPECTION,
  FRAME_GAME,
  FRAME_OBSERVATION,
  PANEL_TYPES,
  panelDefinition,
  panelLabels,
  panelProcessing,
  panelSubscriptions,
} from "./panels/catalog.js";
import { episodeReport } from "./episode-report.js";
import { episodeStepRange, RecordedStepReader, EventOverview, timelineEventMarkers } from "./episode-timeline.js";
import { RecordedStepPrefetch } from "./recorded-step-prefetch.js";
import { eventColorFill, eventLabels } from "./event-colors.js";
import { mountPlaybackSettings } from "./playback-settings.js";
import { snapshotActivatesCheckpointSelection, snapshotFailsCheckpointSelection } from "./playback-transition.js";
import { SynchronizedPresentation } from "./synchronized-presentation.js";
import {
  playbackSourceTitle,
  statusMessageShouldToast,
  timelineProgress,
  transportPresentation,
  workspaceIsEditable,
} from "./player-presentation.js";
import { PanelManager } from "./panels/manager.js";
import { PanelRuntime } from "./panels/runtime.js";
import {
  DEFAULT_GRID_CELL_HEIGHT,
  viewportGridCellHeight,
} from "./panels/layout-sizing.js";
import { mountTrajectoryControls } from "./trajectory-controls.js";
import { setSvgUseHref, text, timelineLabel } from "./panels/shared.js";
import {
  bumpWorkspaceRevision,
  compareWorkspaceRevisions,
  createDefaultWorkspace,
  createTelemetryInstance,
  normalizePanelConfig,
  normalizeWorkspace,
} from "./panels/workspace.js";

const FRAME_HEADER_BYTES = 32;
const panelName = location.pathname.startsWith("/panel/")
  ? location.pathname.slice("/panel/".length)
  : null;
const workspaceWindowName = location.pathname.startsWith("/workspace/")
  ? location.pathname.slice("/workspace/".length)
  : null;
const pairedWorkspace = new URLSearchParams(location.search).get("workspace") === "paired";
const token = new URLSearchParams(location.hash.slice(1)).get("token") || "";
const WORKSPACE_ID_KEY = "gradlab.player.workspace.v7.id";
const LAYOUT_KEY = pairedWorkspace
  ? "gradlab.player.workspace.v8.paired"
  : "gradlab.player.workspace.v7.single";
const SAVED_LAYOUTS_KEY = "gradlab.player.workspace.saved.v7";
const STATS_WINDOW_ID = "stats";
const workspaceId = localStorage.getItem(WORKSPACE_ID_KEY) || crypto.randomUUID();
localStorage.setItem(WORKSPACE_ID_KEY, workspaceId);
const rewardReferences = rewardReferenceStore(localStorage, workspaceId);
const windowId = panelName ? `panel-${panelName}` : (workspaceWindowName || "main");

function defaultLayout() {
  return createDefaultWorkspace({ paired: pairedWorkspace, writer: windowId });
}

const state = {
  rgbEnabled: true,
  socket: null,
  connected: false,
  clientId: null,
  snapshot: null,
  liveSnapshot: null,
  snapshots: new Map(),
  frameBlobs: new Map([
    [FRAME_GAME, new Map()],
    [FRAME_OBSERVATION, new Map()],
    [FRAME_ATTRIBUTION, new Map()],
    [FRAME_CNN_INSPECTION, new Map()],
  ]),
  inspectionSequence: null,
  replayingInspection: false,
  inspectionReplayTimer: null,
  inspectionFrameRequestTimer: null,
  inspectionPauseCommandId: null,
  attributionCommand: null,
  attributionPreference: { mode: "gradcam", interval: 1 },
  cnnCaptureCommand: null,
  timelineSequences: [],
  inspectionHistory: null,
  eventOverview: new EventOverview(),
  seekingStep: null,
  history: [],
  historyLimit: 4096,
  hasControl: false,
  publicationAuthority: false,
  publicationCapability: null,
  controlEpoch: 0,
  publicationCurrent: null,
  publicationJob: null,
  publicationPoll: null,
  frameSequence: new Map(),
  receivedFrameSequence: new Map(),
  retainedEpisode: null,
  recordedEpisodeId: null,
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
  backgroundPlaybackSnapshot: null,
  applicationSnapshot: null,
  workspaceReady: false,
  checkpointLoad: null,
};
let panelRuntime = null;
let sourceBrowser = null;
let sourceBrowserPromise = null;
let contractViewer = null;
let contractViewerPromise = null;
let gridStack = null;
let gridCellHeight = DEFAULT_GRID_CELL_HEIGHT;
let syncingGrid = false;
let panelManager = null;
let playbackSettings = null;
const INSPECTION_FRAME_REQUEST_DELAY_MS = 50;
let youtubeOAuthPopup = null;

const workspaceChannel = "BroadcastChannel" in window
  ? new BroadcastChannel(`gradlab-player-${workspaceId}`)
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

function panelSuspended(id) {
  const fullscreen = document.fullscreenElement;
  if (!fullscreen?.matches(".game-stage")) return false;
  return fullscreen.closest("[data-panel]")?.dataset.panel !== id;
}

function processingPanels() {
  return panelsInThisWindow().filter((id) => !panelSuspended(id));
}

function subscriptions() {
  return panelSubscriptions(state.layout, processingPanels()).filter(
    (name) => name !== "game" || state.rgbEnabled !== false,
  );
}

function processing() {
  const features = new Set(panelProcessing(state.layout, processingPanels()));
  if (state.windowId === "main") features.add("rewards");
  return [...features];
}

function enabledPanelDefinitions() {
  return processingPanels()
    .map((id) => panelDefinition(state.layout, id))
    .filter((definition) => definition?.enabled)
    .map((definition) => state.rgbEnabled === false
      ? { ...definition, frameKinds: definition.frameKinds.filter((kind) => kind !== FRAME_GAME) }
      : definition);
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

function beginCheckpointLoad({ commandId, checkpointId }) {
  state.checkpointLoad = {
    commandId: String(commandId || ""),
    checkpointId: String(checkpointId || ""),
  };
  const mask = $("#checkpoint-loading-mask");
  mask.hidden = false;
  document.activeElement?.blur?.();
  document.body.classList.add("checkpoint-loading");
  document.body.setAttribute("aria-busy", "true");
}

function finishCheckpointLoad() {
  if (!state.checkpointLoad) return;
  state.checkpointLoad = null;
  $("#checkpoint-loading-mask").hidden = true;
  document.body.classList.remove("checkpoint-loading");
  document.body.removeAttribute("aria-busy");
}

function snapshotCompletesCheckpointLoad(snapshot) {
  return Boolean(
    state.checkpointLoad
    && snapshot?.app?.phase === "active"
    && String(snapshot?.app?.route?.checkpoint_id || "")
      === state.checkpointLoad.checkpointId
  );
}

function updateConnection(label, kind = "") {
  const badge = $("#connection-status");
  badge.textContent = label;
  badge.className = `sync-status ${kind}`.trim();
  badge.hidden = label === "Synced" && !kind;
}

function resetSession(epoch) {
  cancelInspectionFrameRequest();
  state.sessionEpoch = Number(epoch) || 0;
  state.backgroundPlaybackSnapshot = null;
  state.retainedEpisode = null;
  state.recordedEpisodeId = null;
  state.inspectionSequence = null;
  state.inspectionPauseCommandId = null;
  state.attributionCommand = null;
  state.cnnCaptureCommand = null;
  state.liveSnapshot = null;
  chartHistory.updateContext({ epoch: state.sessionEpoch });
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
        checkpointNavigationRoot: $("#checkpoint-navigation"),
        beginCheckpointLoad,
        openInspection: (endpoint, options) => openContractInspection(endpoint, options),
        openSourceRoute: (route) => openSourceRoute(route),
      });
      return sourceBrowser;
    });
  }
  return sourceBrowserPromise;
}

async function ensureContractViewer() {
  if (contractViewer) return contractViewer;
  if (!contractViewerPromise) {
    contractViewerPromise = import("./documents/viewer.js").then(({ ContractViewer }) => {
      contractViewer = new ContractViewer($("#contract-viewer"), { token, showToast });
      return contractViewer;
    });
  }
  return contractViewerPromise;
}

async function openContractInspection(endpoint, options = {}) {
  const viewer = await ensureContractViewer();
  return viewer.open(endpoint, options);
}

function openSourceRoute(route) {
  const current = state.applicationSnapshot || state.liveSnapshot || {};
  if (!state.sourceMode && current?.app?.has_active_runner) {
    state.backgroundPlaybackSnapshot = current;
  }
  const snapshot = {
    ...current,
    app: {
      ...(current.app || {}),
      phase: "selecting",
      message: "",
      error: "",
      route: { ...route },
      has_active_runner: Boolean(
        current?.app?.has_active_runner || state.backgroundPlaybackSnapshot,
      ),
    },
  };
  state.applicationSnapshot = snapshot;
  state.liveSnapshot = snapshot;
  state.snapshot = snapshot;
  setSourceMode(true, snapshot);
}

function setSourceMode(active, snapshot = null) {
  state.sourceMode = Boolean(active);
  const activeCheckpointRoute = (
    !state.sourceMode
    && snapshot?.app?.route?.checkpoint_id
  );
  const activeRecordingRoute = (
    !state.sourceMode
    && snapshot?.mode === "trajectory"
    && snapshot?.app?.route?.environment_id
  );
  document.body.classList.toggle("source-selection", state.sourceMode);
  $("#source-browser").hidden = !state.sourceMode;
  $("#checkpoint-navigation").hidden = Boolean(state.sourceMode || !activeCheckpointRoute);
  $("#page-title").hidden = Boolean(state.sourceMode || activeRecordingRoute);
  $("#source-back").hidden = Boolean(
    state.sourceMode
    || activeRecordingRoute
    || !(snapshot?.app?.has_active_runner || state.liveSnapshot?.app?.has_active_runner)
  );
  $("#more-toggle").hidden = state.sourceMode;
  $("#inspect-active").hidden = !(
    snapshot?.app?.has_active_runner || state.liveSnapshot?.app?.has_active_runner
  );
  if (!state.sourceMode) {
    const expected = snapshot;
    if (activeCheckpointRoute || activeRecordingRoute) {
      if (sourceBrowser) {
        sourceBrowser.renderActiveBreadcrumbs(expected);
        $("#source-breadcrumbs").hidden = Boolean(activeCheckpointRoute);
      } else {
        void ensureSourceBrowser().then((browser) => {
          if (!state.sourceMode && state.applicationSnapshot === expected) {
            browser.renderActiveBreadcrumbs(expected);
            $("#source-breadcrumbs").hidden = Boolean(activeCheckpointRoute);
          }
        }).catch((error) => showToast(`Source breadcrumbs failed: ${error.message || error}`, true));
      }
    } else {
      sourceBrowser?.stop();
    }
    if (!state.workspaceReady) {
      state.workspaceReady = true;
      void applyLayout();
    }
    updateLayoutTitle();
    return;
  }
  document.title = "Select playback source · gradlab";
  const expected = snapshot;
  void ensureSourceBrowser().then((browser) => {
    if (state.sourceMode && state.applicationSnapshot === expected) browser.render(expected);
  }).catch((error) => showToast(`Source browser failed: ${error.message || error}`, true));
}

function connect() {
  if (!token) {
    updateConnection("Missing session token", "error");
    showToast("Open the complete dashboard URL printed by gradlab.", true);
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
      processing: processing(),
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
    state.publicationAuthority = false;
    state.publicationCapability = null;
    finishCheckpointLoad();
    updateConnection("Disconnected", "error");
    updateControlState();
  });
  socket.addEventListener("error", () => {
    finishCheckpointLoad();
    updateConnection("Connection error", "error");
  });
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
    if (message.timeline && (!state.recordedEpisodeId || message.timeline.episode_id === state.recordedEpisodeId)) {
      state.eventOverview.load(message.timeline);
    } else {
      state.history.filter((point) => Number(point.episode) === state.retainedEpisode)
        .forEach((point) => state.eventOverview.append(point));
    }
    renderHistory();
    return;
  }
  if (message.type === "publication_authority") {
    state.publicationAuthority = Boolean(message.has_authority);
    state.publicationCapability = message.capability || null;
    state.controlEpoch = Number(message.control_epoch || 0);
    updatePublicationButton();
    return;
  }
  if (message.type === "session_changed") {
    resetSession(message.session_epoch);
    return;
  }
  if (message.type === "snapshot") {
    const epoch = Number(message.session_epoch || 0);
    if (epoch !== state.sessionEpoch) resetSession(epoch);
    if (snapshotFailsCheckpointSelection(state.checkpointLoad, message)) {
      finishCheckpointLoad();
      showToast(message.app.error || "Could not open checkpoint", true);
    }
    if (
      state.sourceMode
      && state.backgroundPlaybackSnapshot
      && message.app?.phase === "active"
      && !snapshotActivatesCheckpointSelection(state.checkpointLoad, message)
    ) {
      state.backgroundPlaybackSnapshot = message;
      state.hasControl = Boolean(message.control?.has_control);
      state.controlEpoch = Number(message.control_epoch || 0);
      trajectoryControls.render();
      return;
    }
    if (snapshotActivatesCheckpointSelection(state.checkpointLoad, message)) {
      state.backgroundPlaybackSnapshot = null;
    }
    state.applicationSnapshot = message;
    state.hasControl = Boolean(message.control?.has_control);
    state.controlEpoch = Number(message.control_epoch || 0);
    updatePublicationButton();
    if (typeof message.session?.rgb_enabled === "boolean" && state.rgbEnabled !== message.session.rgb_enabled) {
      state.rgbEnabled = message.session.rgb_enabled;
      recordedStepReader.invalidate();
      state.frameBlobs.get(FRAME_GAME).clear();
      panelRuntime.resetFrames();
      send({ type: "subscribe", subscriptions: subscriptions(), processing: processing() });
      refreshPanels();
      if (state.rgbEnabled && state.inspectionSequence !== null) {
        void inspectStep(Number(state.snapshot?.transition?.step));
      }
    }
    if (message.app && message.app.phase !== "active") {
      state.liveSnapshot = message;
      state.snapshot = message;
      setSourceMode(true, message);
      updateControlState();
      return;
    }
    if (message.mode === "trajectory") finishCheckpointLoad();
    setSourceMode(false, message);
    prepareRetainedEpisode(message);
    state.snapshots.set(Number(message.sequence), message);
    pruneRetainedTrace();
    if (message.history_point) ingestHistoryPoint(message.history_point);
    void livePresentation.offer(message).catch(reportPresentationError);
    return;
  }
  if (message.type === "command_result") {
    if (message.id === state.inspectionPauseCommandId && !message.ok) {
      state.inspectionPauseCommandId = null;
    }
    if (message.id === state.attributionCommand?.id) state.attributionCommand = null;
    if (message.id === state.cnnCaptureCommand?.id) state.cnnCaptureCommand = null;
    if (message.id === state.checkpointLoad?.commandId && !message.ok) {
      finishCheckpointLoad();
    }
    if (!message.ok) showToast(message.error || "Command failed", true);
    return;
  }
  if (message.type === "error") showToast(message.error || "Player error", true);
}

function updatePublicationButton() {
  const button = $("#publish-episode");
  if (!button) return;
  const snapshot = state.applicationSnapshot || state.liveSnapshot;
  const configured = Boolean(snapshot?.publication?.configured);
  const complete = snapshot?.publication_capture?.ready === true;
  button.hidden = !(configured && complete && state.publicationAuthority);
}

async function publicationApi(path, { method = "GET", body } = {}) {
  if (!state.publicationAuthority || !state.publicationCapability || !state.clientId) {
    throw new Error("This tab no longer has publication authority.");
  }
  const response = await fetch(path, {
    method,
    headers: {
      Authorization: `Bearer ${token}`,
      "X-Gradlab-Client": state.clientId,
      "X-Gradlab-Control-Epoch": String(state.controlEpoch),
      "X-Gradlab-Publication-Capability": state.publicationCapability,
      ...(body === undefined ? {} : { "Content-Type": "application/json" }),
    },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  const payload = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
  if (!response.ok) throw new Error(payload.error || payload.message || `HTTP ${response.status}`);
  return payload;
}

function publicationFact(term, value, selector = "#publication-capture") {
  const facts = $(selector);
  const dt = document.createElement("dt");
  const dd = document.createElement("dd");
  dt.textContent = term;
  dd.textContent = String(value ?? "—");
  facts.append(dt, dd);
}

function publicationSettings() {
  return {
    privacy: $("#publication-privacy").value,
    thumbnail_time: Number($("#publication-thumbnail-time").value),
    tags: $("#publication-tags").value.split(",").map((value) => value.trim()).filter(Boolean),
    operator_note: $("#publication-note").value,
    feature: $("#publication-feature").checked,
  };
}

function renderPublicationPreview(preview) {
  const facts = $("#publication-generated");
  facts.replaceChildren();
  if (!preview) return;
  publicationFact("Generated title", preview.title, "#publication-generated");
  publicationFact("Generated description", preview.description, "#publication-generated");
  publicationFact("Repository", `${preview.repo_id}@${preview.release_tag}`, "#publication-generated");
  publicationFact("Release tier", preview.release_tier, "#publication-generated");
  publicationFact("Acceptance", preview.acceptance?.passed ? "Accepted" : "Not accepted", "#publication-generated");
  publicationFact("Replay", `${preview.replay?.status}: ${preview.replay?.outcome}`, "#publication-generated");
  publicationFact("Comparison", preview.comparison?.reason, "#publication-generated");
  publicationFact("Environment container", preview.containers?.environment, "#publication-generated");
  publicationFact("Featured container", preview.feature ? preview.containers?.featured : "Not requested", "#publication-generated");
  publicationFact("Operator note", preview.operator_note || "None", "#publication-generated");
}

async function refreshPublicationPreview() {
  const preview = await publicationApi("/api/publication/preview", {
    method: "POST",
    body: publicationSettings(),
  });
  renderPublicationPreview(preview);
  return preview;
}

function renderPublicationCurrent(current) {
  state.publicationCurrent = current;
  const capture = current?.capture;
  const facts = $("#publication-capture");
  facts.replaceChildren();
  if (!current?.available || !capture) {
    $("#publication-status").textContent = current?.message || "No publishable episode is ready.";
    $("#publication-submit").disabled = true;
    return;
  }
  publicationFact("Outcome", capture.outcome);
  publicationFact("Episode seed", capture.seed);
  publicationFact("Steps", capture.steps);
  publicationFact("Return", capture.return);
  publicationFact("Action selection", capture.sampling_mode);
  publicationFact("Capture", capture.capture_id);
  $("#publication-status").textContent = "The exact completed episode will be uploaded to both destinations.";
  renderPublicationPreview(current.preview);
  if (current.job) renderPublicationJob(current.job);
}

function renderPublicationCredentials(result) {
  const panel = $("#publication-credentials");
  const hf = result?.huggingface || {};
  const yt = result?.youtube || {};
  panel.textContent = [
    `Hugging Face: ${hf.ready ? `${hf.username} → ${hf.namespace}` : (hf.message || "not ready")}`,
    `YouTube: ${yt.ready ? `${yt.channel_title} (${yt.channel_id})` : (yt.message || "not ready")}`,
  ].join("\n");
  panel.style.whiteSpace = "pre-line";
  $("#publication-authorize-youtube").hidden = Boolean(yt.ready);
  $("#publication-submit").disabled = !(result?.ready && state.publicationCurrent?.available && !state.publicationJob);
}

function renderPublicationJob(job) {
  if (!job) return;
  state.publicationJob = job;
  const panel = $("#publication-job");
  panel.hidden = false;
  panel.replaceChildren();
  const summary = document.createElement("div");
  summary.textContent = `${job.state || "queued"} · ${job.progress?.phase || "queued"}${job.message ? ` · ${job.message}` : ""}`;
  panel.append(summary);
  Object.entries(job.urls || {}).forEach(([label, url]) => {
    if (!String(url).startsWith("https://")) return;
    const row = document.createElement("div");
    const link = document.createElement("a");
    link.href = String(url);
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = `${label}: ${url}`;
    row.append(link);
    panel.append(row);
  });
  const terminal = ["succeeded", "failed", "blocked", "canceled"].includes(job.state);
  $("#publication-submit").disabled = true;
  $("#publication-retry").hidden = !["failed", "blocked", "canceled"].includes(job.state);
  $("#publication-cancel").hidden = terminal;
  $("#publication-resolve").hidden = !(job.state === "blocked" && job.progress?.phase === "youtube_uncertain");
  $("#publication-cleanup").hidden = !terminal;
  clearInterval(state.publicationPoll);
  state.publicationPoll = null;
  if (!terminal && job.job_id) {
    state.publicationPoll = setInterval(() => {
      void publicationApi(`/api/publication/jobs/${encodeURIComponent(job.job_id)}`)
        .then(renderPublicationJob)
        .catch((error) => {
          clearInterval(state.publicationPoll);
          state.publicationPoll = null;
          showToast(error.message, true);
        });
    }, 2000);
  }
}

async function checkPublicationCredentials() {
  $("#publication-credentials").textContent = "Checking both accounts…";
  const result = await publicationApi("/api/publication/preflight", { method: "POST" });
  renderPublicationCredentials(result);
  return result;
}

async function openPublicationDialog() {
  const dialog = $("#publication-dialog");
  dialog.showModal();
  try {
    $("#publication-status").textContent = "Rendering the completed episode…";
    await publicationApi("/api/publication/render", { method: "POST" });
    const [current, ticket] = await Promise.all([
      publicationApi("/api/publication/current"),
      publicationApi("/api/publication/replay-ticket", { method: "POST" }),
    ]);
    renderPublicationCurrent(current);
    await refreshPublicationPreview();
    $("#publication-video").src = ticket.url;
    if (current.job) renderPublicationJob(current.job);
    else await checkPublicationCredentials();
  } catch (error) {
    $("#publication-status").textContent = error.message || String(error);
    $("#publication-submit").disabled = true;
  }
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

async function handleFrame(buffer) {
  const view = new DataView(buffer);
  if (buffer.byteLength <= FRAME_HEADER_BYTES) return;
  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 4));
  if (magic !== "RLP3") return;
  const kind = view.getUint8(4);
  if (kind === FRAME_GAME && state.rgbEnabled === false) return;
  const epoch = Number(view.getBigUint64(8));
  const sequence = Number(view.getBigUint64(16));
  const generation = Number(view.getBigUint64(24));
  if (epoch !== state.sessionEpoch) return;
  state.receivedFrameSequence.set(
    kind,
    Math.max(sequence, state.receivedFrameSequence.get(kind) ?? -1),
  );
  const blob = new Blob([buffer.slice(FRAME_HEADER_BYTES)], { type: "image/png" });
  rememberFrame(kind, sequence, generation, blob, state.inspectionSequence);
  const selectedSnapshot = state.inspectionSequence === null
    ? state.liveSnapshot
    : state.snapshot;
  const expectedGeneration = frameGeneration(kind, selectedSnapshot);
  const exactGeneration = !isGeneratedFrame(kind)
    || (expectedGeneration > 0 && expectedGeneration === generation);
  if (
    exactGeneration
    && (
      state.inspectionSequence === sequence
      || (
        state.inspectionSequence === null
        && Number(state.liveSnapshot?.sequence) === sequence
      )
    )
  ) {
    await panelRuntime.renderFrame(kind, blob, { sequence, generation });
  }
  state.frameSequence.set(kind, sequence);
  await livePresentation.notifyReady().catch(reportPresentationError);
}

function requiredFrameKinds(snapshot) {
  const visible = new Set(
    enabledPanelDefinitions().flatMap((definition) => definition.frameKinds),
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
    state.history[index] = { ...state.history[index], ...point };
    return true;
  }
  state.history.push(point);
  state.history = normalizedHistory(state.history);
  return true;
}

function currentEpisodeHistory() {
  if (state.inspectionSequence !== null && state.inspectionHistory) return state.inspectionHistory;
  const episode = episodeForSnapshot(state.liveSnapshot) ?? state.retainedEpisode;
  if (episode === null) return state.history;
  return state.history.filter((point) => Number(point.episode) === episode);
}

function setRewardReference(step) {
  if (!Number.isInteger(step) || step !== state.snapshot?.transition?.step) return;
  rewardReferences.set(state.snapshot, state.sessionEpoch);
  panelRuntime?.renderHistory(currentEpisodeHistory(), state.snapshot, panelView());
}

function updateChartContext() {
  chartHistory.updateContext({
    epoch: state.sessionEpoch,
    episodeId: state.liveSnapshot?.trajectory?.episode_id,
    episode: episodeForSnapshot(state.liveSnapshot),
    lastStep: state.liveSnapshot?.trajectory?.last_step,
    liveHistory: state.history,
    throughStep: state.snapshot?.transition?.step ?? 0,
  });
}

function updateChartDemand() {
  chartHistory.setDemand(enabledPanelDefinitions().some(usesChartHistory));
}

function panelView() {
  updateChartContext();
  const chart = chartHistory.read();
  return {
    history: currentEpisodeHistory(),
    chartHistory: chart.data,
    chartStatus: chart,
    rewardReference: rewardReferences.get(state.snapshot, state.sessionEpoch),
    chartRange: chart.range,
    inspection: state.inspectionSequence !== null,
    sessionEpoch: state.sessionEpoch,
    selectedSequence: state.inspectionSequence ?? state.snapshot?.sequence ?? null,
    liveSequence: state.liveSnapshot?.sequence ?? null,
  };
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
    showToast("The selected transition expired from the bounded history.", true);
    broadcastInspection(null);
  }
}

function clearRetainedEpisode() {
  state.eventOverview.reset(null);
  recordedStepReader.invalidate();
  state.inspectionHistory = null;
  state.seekingStep = null;
  livePresentation.reset();
  panelRuntime?.resetFrames();
  state.snapshots.clear();
  state.frameBlobs.forEach((frames) => frames.clear());
  state.frameSequence.clear();
  state.receivedFrameSequence.clear();
  state.timelineSequences = [];
}

function prepareRetainedEpisode(snapshot) {
  const episode = episodeForSnapshot(snapshot);
  if (episode === null) return;
  const episodeId = snapshot.trajectory?.episode_id ?? null;
  if ((state.retainedEpisode !== null && state.retainedEpisode !== episode)
      || (state.recordedEpisodeId !== null && state.recordedEpisodeId !== episodeId)) {
    clearRetainedEpisode();
    stopInspectionReplay({ render: false });
    state.inspectionSequence = null;
  }
  state.retainedEpisode = episode;
  state.recordedEpisodeId = episodeId;
  const overviewId = episodeId ?? `episode-${episode}`;
  if (state.eventOverview.episodeId !== overviewId) state.eventOverview.reset(overviewId);
}

function hideGoExploreValuePanel(snapshot) {
  if (snapshot?.policy?.provenance?.search_algorithm_id !== "go-explore") return false;
  const panel = state.layout?.panels?.value;
  if (!panel?.builtin || !panel.placement.visible) return false;
  panel.placement.visible = false;
  persistLayout();
  return true;
}

function applySnapshot(snapshot) {
  if (snapshot.mode === "trajectory") state.inspectionSequence = null;
  const previousEnvironmentId = state.liveSnapshot?.session?.env_id;
  const previousEpisode = episodeForSnapshot(state.liveSnapshot);
  const nextEpisode = episodeForSnapshot(snapshot);
  const episodeChanged = (
    previousEpisode !== null
    && nextEpisode !== null
    && previousEpisode !== nextEpisode
  );
  state.liveSnapshot = snapshot;
  if (hideGoExploreValuePanel(snapshot)) void applyLayout();
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
    void showFramesForSequence(Number(snapshot.sequence)).then(() => {
      if (snapshotCompletesCheckpointLoad(snapshot)) finishCheckpointLoad();
    });
  } else {
    panelRuntime.invoke("controls", "render", snapshot);
    updateControlState();
    renderWorkspaceStatus();
    renderTimeline();
    if (snapshotCompletesCheckpointLoad(snapshot)) finishCheckpointLoad();
  }
  if (historyChanged) {
    if (state.inspectionSequence === null || !state.replayingInspection) renderHistory();
    // Replay already renders its recorded inspection page at each cursor step.
    // The recorded chart refresh still redraws whenever its data arrives.
    else updateChartContext();
  }
  syncAttributionToPanel();
  syncCnnCaptureToPanel();
}

async function prepareSnapshotFrames(snapshot) {
  // Background inference must not replace a replay frame's pending decode.
  if (state.inspectionSequence !== null) return;
  const sequence = Number(snapshot.sequence);
  await Promise.all(requiredFrameKinds(snapshot).map((kind) => (
    panelRuntime.prepareFrame(
      kind,
      exactFrameBlob(kind, sequence),
      { sequence, generation: 0 },
    )
  )));
}

function reportPresentationError(error) {
  console.error("Synchronized playback presentation failed", error);
  showToast("A synchronized playback frame could not be displayed.", true);
}

const livePresentation = new SynchronizedPresentation({
  isReady: requiredFramesAvailable,
  prepare: prepareSnapshotFrames,
  present: applySnapshot,
});

function send(value) {
  if (state.socket?.readyState === WebSocket.OPEN) state.socket.send(JSON.stringify(value));
}

function command(name, payload = {}) {
  if (!state.hasControl) {
    showToast("This window is an observer. Choose Control here first.", true);
    return null;
  }
  if (name === "set_fps") payload = { rgb_enabled: state.rgbEnabled !== false, ...payload };
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

function syncAttributionToPanel() {
  if (state.liveSnapshot?.mode === "trajectory") return;
  const panel = state.layout?.panels?.attribution;
  const attribution = state.liveSnapshot?.session?.attribution;
  if (
    !panel
    || panel.placement?.window !== state.windowId
    || !attribution
  ) return;
  const supported = state.liveSnapshot?.policy?.attribution?.supported_modes;
  if (!Array.isArray(supported)) return;
  if (supported.includes(attribution.mode) && attribution.mode !== "none") {
    state.attributionPreference = {
      mode: attribution.mode,
      interval: Math.max(1, Number(attribution.interval) || 1),
    };
  }
  const desired = Boolean(panel.enabled && panel.placement.visible);
  if (desired && !supported.length) return;
  const active = supported.includes(attribution.mode) && attribution.mode !== "none";
  if (active === desired) {
    if (state.attributionCommand?.desired === desired) state.attributionCommand = null;
    return;
  }
  if (state.attributionCommand?.desired === desired || !state.hasControl) return;
  const preferredMode = supported.includes(state.attributionPreference?.mode)
    ? state.attributionPreference.mode
    : supported[0];
  const payload = desired
    ? {
      mode: preferredMode,
      interval: Math.max(1, Number(state.attributionPreference?.interval) || 1),
    }
    : { mode: "none" };
  const id = command("set_attribution", payload);
  if (id) state.attributionCommand = { id, desired };
}

function syncCnnCaptureToPanel() {
  if (state.liveSnapshot?.mode === "trajectory") return;
  const panel = state.layout?.panels?.cnn;
  const cnn = state.liveSnapshot?.session?.cnn;
  if (
    !panel
    || panel.placement?.window !== state.windowId
    || !cnn
  ) return;
  const desired = Boolean(panel.enabled && panel.placement.visible);
  const layers = state.liveSnapshot?.policy?.cnn?.layers;
  if (desired && (!Array.isArray(layers) || !layers.length)) return;
  if (Boolean(cnn.enabled) === desired) {
    if (state.cnnCaptureCommand?.desired === desired) state.cnnCaptureCommand = null;
    return;
  }
  if (state.cnnCaptureCommand?.desired === desired || !state.hasControl) return;
  const id = command("set_cnn_inspection", { enabled: desired });
  if (id) state.cnnCaptureCommand = { id, desired };
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

const {
  canReplayInspection, stopInspectionReplay, playFromCurrentPosition,
  pauseCurrentPlayback, playbackIsRunning,
} = createPlaybackTransport({
  state, command, renderSnapshot, inspectStep, setInspectionCursor, returnToLive,
  inspectionEpisodeSequences,
  invalidateRead: () => recordedStepReader.invalidate(),
});

function updateTimelinePlaybackControl() {
  const playbackToggle = $("#timeline-playback-toggle");
  const playbackIcon = $("#timeline-playback-icon");
  if (!playbackToggle || !playbackIcon) return;
  const session = state.liveSnapshot?.session || state.snapshot?.session || {};
  const presentation = transportPresentation({
    running: playbackIsRunning(),
    replaying: state.replayingInspection,
    independentInference: hasIndependentInference(state.liveSnapshot),
    hasControl: state.hasControl,
    canReplay: canReplayInspection() || Boolean(state.liveSnapshot?.trajectory?.imported && session.awaiting_next_episode),
    session,
    recording: (state.liveSnapshot?.mode || state.snapshot?.mode) === "recording",
  });
  playbackToggle.dataset.action = presentation.action;
  playbackToggle.disabled = presentation.disabled;
  playbackToggle.title = presentation.reason;
  playbackToggle.classList.toggle("primary", presentation.action !== "pause");
  playbackToggle.setAttribute("aria-label", presentation.label);
  setSvgUseHref(playbackIcon, `/assets/tabler-icons.svg#ti-${presentation.icon}`);
  const reset = $("#timeline-reset");
  if (reset) {
    const mode = state.liveSnapshot?.mode || state.snapshot?.mode;
    const canReset = (
      state.hasControl
      && !["recording", "dataset", "trajectory"].includes(mode)
      && (!session.awaiting_next_episode || session.can_start_next_episode)
    );
    reset.disabled = !canReset;
    reset.title = !state.hasControl
      ? "Another window has control"
      : canReset
        ? "Reset to the selected seed and pause"
        : "The configured episode limit has been reached";
  }
}

function updateControlState() {
  trajectoryControls?.render();
  updateTimelinePlaybackControl();
  playbackSettings?.updateControl();
  panelRuntime?.invoke("controls", "updateControl");
}

function renderWorkspaceStatus() {
  $("#timeline-label").textContent = timelineLabel(
    state.snapshot || state.liveSnapshot,
  );
}

function renderPlaybackEvidenceStatus(snapshot) {
  const status = $("#playback-evidence-status");
  if (!status || !snapshot || state.sourceMode || state.windowId !== "main") {
    if (status) status.hidden = true;
    return;
  }
  const report = episodeReport(snapshot);
  const evidenceWarning = /not evidence|differ|incomparable/i.test(
    `${report.semantics} ${report.disclaimer}`,
  );
  status.hidden = !evidenceWarning;
  status.textContent = evidenceWarning ? report.semantics : "";
  status.title = evidenceWarning ? report.disclaimer : "";
}

function renderSnapshot() {
  const snapshot = state.snapshot;
  const session = snapshot.session || {};
  configureMode(snapshot.mode || "playback");
  updateControlState();
  renderWorkspaceStatus();
  renderPlaybackEvidenceStatus(snapshot);
  const actionNamesKey = JSON.stringify([
    session.action_contract || null,
    session.action_names || [],
  ]);
  if (actionNamesKey !== state.actionNamesKey) {
    state.actionNamesKey = actionNamesKey;
    renderHistory();
  }
  if (state.inspectionSequence === null && snapshot.status_message && snapshot.status_message !== state.lastStatus) {
    state.lastStatus = snapshot.status_message;
    if (statusMessageShouldToast(snapshot)) {
      showToast(snapshot.status_message, snapshot.run_state === "paused" && /error|expired|unsupported|no configured/i.test(snapshot.status_message));
    }
  }
  panelRuntime.renderSnapshot(snapshot, panelView());
  playbackSettings?.render(snapshot, panelView());
  fitGridToViewport();
  renderTimeline();
}

function configureMode(mode) {
  if (state.mode === mode) return;
  state.mode = mode;
  const recording = mode === "recording";
  document.body.classList.toggle("recording", recording);
}

function setChartRange(range, { broadcast = true } = {}) {
  chartHistory.selectRange(range);
  if (broadcast) workspaceChannel?.postMessage({ type: "chart-range", source: state.windowId, epoch: state.sessionEpoch, episode: state.liveSnapshot?.trajectory?.episode_id, range });
  renderTimeline();
}

const chartHistory = createChartHistory({ token, onChange: () => scheduleHistoryRender() });

const scheduleHistoryRender = frameScheduler(() => {
  if (!state.snapshot || !panelRuntime) return;
  panelRuntime.renderHistory(currentEpisodeHistory(), state.snapshot, panelView());
  fitGridToViewport();
  renderTimeline();
});

function renderHistory() {
  updateChartContext();
  scheduleHistoryRender();
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

async function showFramesForSequence(sequence) {
  const snapshot = state.snapshots.get(Number(sequence)) || state.snapshot;
  const retainMissing = (
    state.inspectionSequence !== null
    && Number(state.inspectionSequence) === Number(sequence)
  );
  const kinds = [...new Set(
    enabledPanelDefinitions().flatMap((definition) => definition.frameKinds),
  )];
  const missing = [];
  await Promise.all(kinds.map(async (kind) => {
    const generation = frameGeneration(kind, snapshot);
    const expected = frameExpected(kind, snapshot);
    const blob = expected ? exactFrameBlob(kind, sequence, generation) : null;
    if (expected && !blob) missing.push(kind);
    if (blob) {
      await panelRuntime.renderFrame(kind, blob, { sequence, generation });
    } else if (!expected || !retainMissing) {
      await panelRuntime.renderFrame(kind, null, { sequence, generation });
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
  workspaceChannel?.postMessage({
    type: "inspection-frame-request",
    ...request,
  });
  send({
    type: "inspection_frames",
    ...request,
  });
}

function cancelInspectionFrameRequest() {
  clearTimeout(state.inspectionFrameRequestTimer);
  state.inspectionFrameRequestTimer = null;
}

function scheduleInspectionFrameRequest(sequence, kinds) {
  cancelInspectionFrameRequest();
  if (!kinds.length) return;
  state.inspectionFrameRequestTimer = window.setTimeout(() => {
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
    showToast("That transition is not retained in this window.", true);
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
  cancelInspectionFrameRequest();
  if (announce) maybePauseForInspection();
  state.snapshots.set(numericSequence, snapshot);
  state.inspectionSequence = numericSequence;
  pruneRetainedTrace(numericSequence);
  state.snapshot = snapshot;
  renderSnapshot();
  renderHistory();
  void showFramesForSequence(numericSequence).then((missing) => {
    if (Number(state.inspectionSequence) !== numericSequence) return;
    scheduleInspectionFrameRequest(numericSequence, missing);
  });
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

const recordedStepPrefetch = new RecordedStepPrefetch(async ({ epoch, episode_id, step }, { signal } = {}) => {
  const query = new URLSearchParams({ epoch, episode_id, step });
  if (state.rgbEnabled === false) query.set("rgb", "off");
  const response = await fetch(`/api/playback/recorded-step?${query}`, {
    signal,
    headers: { Authorization: `Bearer ${token}` },
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Unable to load the recorded step");
  return payload;
});
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
  renderTimeline();
  try {
    const result = await recordedStepReader.read({ epoch, episode_id: episodeId, step });
    if (!result || epoch !== state.sessionEpoch || episodeId !== state.liveSnapshot?.trajectory?.episode_id) return;
    state.seekingStep = null;
    state.inspectionHistory = result.points;
    const frames = recordedFrames(result);
    setInspectionCursor(result.snapshot.sequence, { snapshot: result.snapshot, frames, preserveReplay });
    if (preserveReplay && state.replayingInspection) {
      void recordedStepPrefetch.ahead({ epoch, episode_id: episodeId, step: step + 1 }, trajectory.last_step);
    }
  } catch (error) {
    state.seekingStep = null;
    stopInspectionReplay({ render: false });
    showToast(error.message, true);
    renderTimeline();
  }
}

function returnToLive({ announce = true } = {}) {
  recordedStepReader.invalidate();
  state.seekingStep = null;
  state.inspectionHistory = null;
  stopInspectionReplay({ render: false });
  cancelInspectionFrameRequest();
  state.inspectionSequence = null;
  state.snapshot = state.liveSnapshot;
  if (state.snapshot) {
    renderSnapshot();
    renderHistory();
    void showFramesForSequence(Number(state.snapshot.sequence));
  }
  if (announce) broadcastInspection(null);
  void restoreLatestRecordedPresentation();
}

async function restoreLatestRecordedPresentation() {
  const live = state.liveSnapshot;
  const trajectory = live?.trajectory;
  if (!trajectory?.transitions || trajectory.imported || live.run_state !== "paused"
      || live.transition?.step !== trajectory.last_step) return;
  const epoch = state.sessionEpoch;
  try {
    const result = await recordedStepReader.read({ epoch, episode_id: trajectory.episode_id, step: trajectory.last_step });
    if (!result || state.inspectionSequence !== null || epoch !== state.sessionEpoch
        || state.liveSnapshot?.sequence !== live.sequence
        || state.liveSnapshot?.trajectory?.episode_id !== trajectory.episode_id) return;
    // Reconnection or cache eviction can lose the latest frame too. Restore only
    // its captured presentation; live transport and configuration stay current.
    const snapshot = { ...state.liveSnapshot, transition: result.snapshot.transition };
    recordedFrames(result).forEach(({ kind, generation, blob }) =>
      rememberFrame(kind, snapshot.sequence, generation, blob, snapshot.sequence));
    state.snapshots.set(Number(snapshot.sequence), snapshot);
    state.snapshot = snapshot;
    renderSnapshot();
    void showFramesForSequence(Number(snapshot.sequence));
  } catch (error) {
    showToast(error.message, true);
  }
}

function renderTimeline() {
  const scrubber = $("#timeline-scrubber");
  if (!scrubber) return;
  const trajectory = state.liveSnapshot?.trajectory;
  const currentEpisode = episodeForSnapshot(state.liveSnapshot);
  const snapshots = [...state.snapshots.values()].filter((snapshot) =>
    currentEpisode === null || episodeForSnapshot(snapshot) === currentEpisode);
  state.timelineSequences = snapshots.map((snapshot) => Number(snapshot.sequence)).sort((a, b) => a - b);
  const range = episodeStepRange(trajectory, snapshots);
  const selected = state.seekingStep ?? Number(trajectory?.imported ? trajectory.current_step
    : state.snapshot?.transition?.step ?? state.snapshot?.session?.step ?? range?.first ?? 0);
  scrubber.min = String(range?.first ?? 0);
  scrubber.max = String(range?.last ?? 0);
  scrubber.step = "1";
  scrubber.disabled = !range || range.first === range.last || Boolean(trajectory?.imported && !state.hasControl);
  scrubber.value = String(selected);
  scrubber.setAttribute("aria-label", "Inspect an episode step");
  scrubber.setAttribute("aria-valuetext", `Step ${selected}`);
  scrubber.style.setProperty("--timeline-progress", `${timelineProgress(
    selected - (range?.first ?? 0), (range?.last ?? 0) - (range?.first ?? 0) + 1,
  )}%`);
  $("#timeline").setAttribute("aria-busy", String(state.seekingStep !== null));
  renderWorkspaceStatus();
  const zoomLabel = $("#timeline-zoom");
  const chartRange = chartHistory.read().range;
  zoomLabel.hidden = !chartRange;
  zoomLabel.textContent = chartRange ? `Steps ${chartRange.first}–${chartRange.last} · Reset zoom` : "";
  const zoomBand = $("#timeline-zoom-band");
  zoomBand.hidden = !chartRange || !range;
  if (chartRange && range) {
    const span = Math.max(1, range.last - range.first);
    zoomBand.style.left = `${100 * (chartRange.first - range.first) / span}%`;
    for (const handle of zoomBand.querySelectorAll(".timeline-range-handle")) {
      const start = handle.dataset.edge === "first";
      handle.setAttribute("aria-valuemin", String(start ? range.first : chartRange.first + 1));
      handle.setAttribute("aria-valuemax", String(start ? chartRange.last - 1 : range.last));
      handle.setAttribute("aria-valuenow", String(chartRange[handle.dataset.edge]));
      handle.setAttribute("aria-valuetext", `Step ${chartRange[handle.dataset.edge]}`);
    }
    zoomBand.style.width = `${100 * (chartRange.last - chartRange.first) / span}%`;
  }
  const markers = $("#timeline-markers");
  if (!range) { markers.replaceChildren(); return; }
  const markerSlots = Math.max(1, Math.min(120, Math.floor(scrubber.clientWidth / 9)));
  const interesting = timelineEventMarkers([...state.eventOverview.buckets.values()], range, markerSlots);
  markers.replaceChildren(...interesting.map((point) => {
    const marker = document.createElement("span");
    marker.className = "timeline-marker";
    marker.style.left = `${point.position * 100}%`;
    marker.dataset.step = String(point.step);
    marker.dataset.count = String(point.count);
    marker.style.setProperty("--event-colors", eventColorFill(eventLabels(point)));
    return marker;
  }));
}

function restoreTimelineHome() {
  const timeline = $("#timeline");
  const home = $("#timeline-home");
  timeline?.classList.remove("game-timeline-docked", "visible");
  if (timeline && home && timeline.previousElementSibling !== home) home.after(timeline);
}

function syncTimelineDock() {
  restoreTimelineHome();
  const timeline = $("#timeline");
  const stage = $(".game-panel .game-stage");
  if (!timeline || !stage) return;
  stage.append(timeline);
  timeline.classList.add("game-timeline-docked");
}

function maxPanelRow(targetWindow = state.windowId) {
  return Math.max(0, ...Object.values(state.layout.panels)
    .filter((panel) => (
      panel.placement.visible
      && panel.placement.window === targetWindow
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

function fitGridToViewport() {
  if (!gridStack) return;
  const dashboard = $("#dashboard");
  const timeline = $("#timeline");
  const nextCellHeight = viewportGridCellHeight({
    viewportHeight: window.innerHeight,
    dashboardTop: dashboard.getBoundingClientRect().top,
    timelineHeight: timeline.hidden || timeline.classList.contains("game-timeline-docked")
      ? 0
      : timeline.getBoundingClientRect().height,
    rows: maxPanelRow(),
    maxFillRows: state.windowId === STATS_WINDOW_ID ? 23 : undefined,
  });
  if (nextCellHeight !== gridCellHeight) {
    gridCellHeight = nextCellHeight;
    gridStack.cellHeight(gridCellHeight);
  }
  dashboard.style.height = `${maxPanelRow() * gridCellHeight}px`;
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
  state.layout.preset = "all";
  bumpWorkspaceRevision(state.layout, state.windowId);
  localStorage.setItem(LAYOUT_KEY, JSON.stringify(state.layout));
  if (announce) workspaceChannel?.postMessage({ type: "layout", layout: state.layout, source: state.windowId });
}

function updateLayoutTitle() {
  const route = (state.applicationSnapshot || state.liveSnapshot)?.app?.route || {};
  const environmentId = String(
    route.environment_id || state.liveSnapshot?.session?.env_id || "",
  ).trim();
  const environmentTitle = playbackSourceTitle({
    ...route,
    environment_id: environmentId || "Environment",
  });
  const title = panelName
    ? `${environmentTitle} · ${panelLabel(panelName)}`
    : pairedWorkspace && state.windowId === STATS_WINDOW_ID
      ? `${environmentTitle} · Stats`
      : environmentTitle;
  $("#page-title").textContent = title;
  $("#layout-name-input").value = state.layout.name;
  document.title = `${title} · gradlab`;
}

function updateWorkspaceEditing() {
  const editable = workspaceIsEditable(state.layout?.preset);
  document.body.dataset.workspaceView = "all";
  document.body.classList.toggle("workspace-editing", editable);
  $$('[data-customize-actions]').forEach((element) => { element.hidden = !editable; });
  gridStack?.enableMove(editable);
  gridStack?.enableResize(editable);
  $$("[data-drag-handle], [data-panel-menu]").forEach((control) => {
    control.hidden = !editable;
    control.tabIndex = editable ? 0 : -1;
  });
  $$(".grid-stack-item > .ui-resizable-se").forEach((handle) => {
    handle.hidden = !editable;
  });
  if (!editable) {
    $("#layout-menu").hidden = true;
    $("#panel-menu").hidden = true;
    $("#panel-shelf").hidden = true;
    $("#panels-toggle").setAttribute("aria-expanded", "false");
  }
}

async function applyLayout() {
  updateChartDemand();
  restoreTimelineHome();
  const visibleHere = panelsInThisWindow();
  document.body.classList.toggle("empty-workspace", visibleHere.length === 0);
  updateLayoutTitle();
  updateWorkspaceEditing();
  panelManager?.renderShelf();
  renderSavedLayouts();
  send({
    type: "subscribe",
    subscriptions: subscriptions(),
    processing: processing(),
  });
  syncingGrid = true;
  gridStack.batchUpdate();
  try {
    await panelRuntime.sync(state.layout, state.windowId);
  } finally {
    gridStack.batchUpdate(false);
    syncingGrid = false;
  }
  updateWorkspaceEditing();
  syncTimelineDock();
  fitGridToViewport();
  syncGridNodes();
  refreshPanels();
}

function refreshPanels() {
  if (state.snapshot) {
    panelRuntime.renderSnapshot(state.snapshot, panelView());
    panelRuntime.renderHistory(currentEpisodeHistory(), state.snapshot, panelView());
    const sequence = Number(state.snapshot.sequence);
    const visibleKinds = new Set(
      enabledPanelDefinitions().flatMap((definition) => definition.frameKinds),
    );
    if (visibleKinds.has(FRAME_GAME)) {
      panelRuntime.renderFrame(
        FRAME_GAME,
        exactFrameBlob(FRAME_GAME, sequence),
        { sequence, generation: 0 },
      );
    }
    if (visibleKinds.has(FRAME_OBSERVATION)) {
      panelRuntime.renderFrame(
        FRAME_OBSERVATION,
        exactFrameBlob(FRAME_OBSERVATION, sequence),
        { sequence, generation: 0 },
      );
    }
    if (visibleKinds.has(FRAME_ATTRIBUTION)) {
      const generation = attributionGeneration(state.snapshot);
      panelRuntime.renderFrame(
        FRAME_ATTRIBUTION,
        generation ? exactFrameBlob(FRAME_ATTRIBUTION, sequence, generation) : null,
        { sequence, generation },
      );
    }
    if (visibleKinds.has(FRAME_CNN_INSPECTION)) {
      const generation = cnnInspectionGeneration(state.snapshot);
      panelRuntime.renderFrame(
        FRAME_CNN_INSPECTION,
        generation ? exactFrameBlob(FRAME_CNN_INSPECTION, sequence, generation) : null,
        { sequence, generation },
      );
    }
    fitGridToViewport();
  }
  requestAnimationFrame(() => panelRuntime.resize());
  syncAttributionToPanel();
  syncCnnCaptureToPanel();
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
  const definition = panelDefinition(state.layout, name);
  if (definition?.switchable) {
    const captureLabel = name === "cnn"
      ? "CNN features"
      : name === "attribution"
        ? "attribution"
        : null;
    const toggle = document.createElement("label");
    toggle.className = "panel-processing-toggle";
    toggle.title = captureLabel
      ? `Enable or disable ${captureLabel} capture`
      : `Enable or disable ${panelLabel(name)} data processing`;
    const input = document.createElement("input");
    input.type = "checkbox";
    input.role = "switch";
    input.checked = definition.enabled;
    input.dataset.panelEnabled = name;
    input.setAttribute(
      "aria-label",
      captureLabel ? `${captureLabel} capture` : `${panelLabel(name)} data processing`,
    );
    input.title = toggle.title;
    input.addEventListener("change", () => {
      const instance = state.layout.panels[name];
      if (!instance) return;
      instance.enabled = input.checked;
      persistLayout();
      void applyLayout();
      showToast(
        captureLabel
          ? `${panelLabel(name)} capture ${input.checked ? "enabled" : "disabled"}.`
          : `${panelLabel(name)} processing ${input.checked ? "enabled" : "disabled"}.`,
      );
    });
    toggle.append(input);
    const menu = panel.querySelector("[data-panel-menu]");
    menu?.before(toggle);
  }
  const handle = panel.querySelector("[data-drag-handle]");
  if (handle) {
    handle.draggable = false;
    handle.addEventListener("keydown", (event) => {
      if (!workspaceIsEditable(state.layout.preset)) return;
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
    if (!workspaceIsEditable(state.layout.preset)) return;
    event.stopPropagation();
    openPanelMenu(name, menu);
  });
}

function bindPanelLayout() {
  gridStack = window.GridStack.init({
    alwaysShowResizeHandle: true,
    animate: false,
    cellHeight: DEFAULT_GRID_CELL_HEIGHT,
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
    if (!workspaceIsEditable(state.layout.preset)) return;
    syncGridNodes([item.gridstackNode]);
    persistLayout();
    panelRuntime.resize();
    showToast(`${panelLabel(item.dataset.panel)} resized.`);
  });
  gridStack.on("dragstop", (_event, item) => {
    if (!workspaceIsEditable(state.layout.preset)) return;
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
  if (!workspaceIsEditable(state.layout.preset)) return;
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
  const popup = window.open(windowUrl(targetWindow), `gradlab-${targetWindow}`, "popup");
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
  const closePlayerMenu = () => {
    $("#player-menu").hidden = true;
    $("#more-toggle").setAttribute("aria-expanded", "false");
  };
  $("#source-back").addEventListener("click", () => {
    void ensureSourceBrowser()
      .then((browser) => browser.browseCurrentSource())
      .catch((error) => showToast(`Source browser failed: ${error.message || error}`, true));
  });
  $("#more-toggle").addEventListener("click", (event) => {
    const menu = $("#player-menu");
    const opening = menu.hidden;
    menu.hidden = true;
    event.currentTarget.setAttribute("aria-expanded", String(opening));
    if (opening) positionMenu(menu, event.currentTarget);
  });
  $("#layouts-toggle").addEventListener("click", (event) => {
    if (!workspaceIsEditable(state.layout.preset)) return;
    $("#panel-shelf").hidden = true;
    $("#panels-toggle").setAttribute("aria-expanded", "false");
    positionMenu($("#layout-menu"), event.currentTarget);
  });
  $("#save-layout").addEventListener("click", () => {
    if (!workspaceIsEditable(state.layout.preset)) return;
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
    showToast("All panels restored.");
  });
  $("#panel-new-window").addEventListener("click", () => {
    if (!workspaceIsEditable(state.layout.preset)) return;
    if (state.selectedPanel) movePanelToNewWindow(state.selectedPanel);
    $("#panel-menu").hidden = true;
  });
  $("#panel-dock-main").addEventListener("click", () => {
    if (!workspaceIsEditable(state.layout.preset)) return;
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
    if (!workspaceIsEditable(state.layout.preset)) return;
    if (state.selectedPanel) panelManager.openEditor(state.selectedPanel);
    $("#panel-menu").hidden = true;
  });
  $("#panel-duplicate").addEventListener("click", () => {
    if (!workspaceIsEditable(state.layout.preset)) return;
    if (state.selectedPanel) panelManager.duplicate(state.selectedPanel);
    $("#panel-menu").hidden = true;
  });
  $("#panel-hide").addEventListener("click", () => {
    if (!workspaceIsEditable(state.layout.preset)) return;
    const name = state.selectedPanel;
    if (!name) return;
    state.layout.panels[name].placement.visible = false;
    persistLayout();
    applyLayout();
    $("#panel-menu").hidden = true;
    showToast(`${panelLabel(name)} moved to the panel shelf.`);
  });
  $("#panel-reset-size").addEventListener("click", () => {
    if (!workspaceIsEditable(state.layout.preset)) return;
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
    if (!workspaceIsEditable(state.layout.preset)) return;
    if (state.selectedPanel) panelManager.remove(state.selectedPanel);
    $("#panel-menu").hidden = true;
  });
  $("#panels-toggle").addEventListener("click", (event) => {
    if (!workspaceIsEditable(state.layout.preset)) return;
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
    if (!workspaceIsEditable(state.layout.preset)) return;
    const targetWindow = `window-${crypto.randomUUID().slice(0, 8)}`;
    const popup = window.open(windowUrl(targetWindow), `gradlab-${targetWindow}`, "popup");
    if (!popup) showToast("The browser blocked the new workspace window.", true);
  });
  document.addEventListener("click", (event) => {
    if (!$("#player-menu").contains(event.target) && !event.target.closest("#more-toggle")) {
      closePlayerMenu();
    }
    if (!$("#panel-menu").contains(event.target) && !event.target.closest("[data-panel-menu]")) $("#panel-menu").hidden = true;
    if (!$("#layout-menu").contains(event.target) && !event.target.closest("#layouts-toggle")) $("#layout-menu").hidden = true;
    if (!$("#panel-shelf").contains(event.target) && !event.target.closest("#panels-toggle")) {
      $("#panel-shelf").hidden = true;
      $("#panels-toggle").setAttribute("aria-expanded", "false");
    }
    if (!$("#playback-settings-menu").contains(event.target) && !event.target.closest("#playback-settings-toggle")) {
      $("#playback-settings-menu").hidden = true;
      $("#playback-settings-toggle").setAttribute("aria-expanded", "false");
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    closePlayerMenu();
    $("#layout-menu").hidden = true;
    $("#panel-menu").hidden = true;
    $("#panel-shelf").hidden = true;
    $("#panels-toggle").setAttribute("aria-expanded", "false");
    $("#playback-settings-menu").hidden = true;
    $("#playback-settings-toggle").setAttribute("aria-expanded", "false");
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
  window.addEventListener("storage", (event) => {
    if (event.key === `gradlab-reward-reference-${workspaceId}`) {
      panelRuntime?.renderHistory(currentEpisodeHistory(), state.snapshot, panelView());
    }
  });
  if (workspaceChannel) {
    workspaceChannel.addEventListener("message", (event) => {
      const message = event.data || {};
      if (message.type === "chart-range" && message.source !== state.windowId) {
        if (message.epoch === state.sessionEpoch && message.episode === state.liveSnapshot?.trajectory?.episode_id) setChartRange(message.range, { broadcast: false });
      } else if (message.type === "layout" && message.source !== state.windowId) {
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
          && message.snapshot.trajectory?.episode_id === state.liveSnapshot?.trajectory?.episode_id
        ) {
          recordedStepReader.invalidate();
          state.seekingStep = null;
          state.inspectionHistory = Array.isArray(message.points) ? message.points : null;
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
          const snapshot = state.snapshots.get(Number(message.sequence));
          const generation = frameGeneration(numericKind, snapshot);
          const blob = exactFrameBlob(
            numericKind,
            Number(message.sequence),
            generation,
          );
          if (!blob) return;
          workspaceChannel.postMessage({
            type: "inspection-frame",
            session_epoch: state.sessionEpoch,
            sequence: Number(message.sequence),
            kind: numericKind,
            generation,
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
        && [FRAME_GAME, FRAME_OBSERVATION, FRAME_ATTRIBUTION, FRAME_CNN_INSPECTION].includes(Number(message.kind))
        && message.blob instanceof Blob
      ) {
        rememberFrame(
          Number(message.kind),
          Number(message.sequence),
          Number(message.generation || 0),
          message.blob,
          Number(message.sequence),
        );
        void panelRuntime.renderFrame(
          Number(message.kind),
          message.blob,
          {
            sequence: Number(message.sequence),
            generation: Number(message.generation || 0),
          },
        );
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
    chartHistory.dispose();
    workspaceChannel?.postMessage({ type: "window-closing", window: state.windowId });
  });
}

function bindTimeline() {
  bindTimelineRange($("#timeline-zoom-band"), () => {
    const scrubber = $("#timeline-scrubber");
    return { first: Number(scrubber.min), last: Number(scrubber.max) };
  }, () => chartHistory.read().range, setChartRange);
  $("#timeline-zoom").addEventListener("click", () => setChartRange(null));
  const scrubber = $("#timeline-scrubber");
  let trackWidth = 0;
  const markerResizeObserver = new ResizeObserver(([entry]) => {
    if (entry.contentRect.width === trackWidth) return;
    trackWidth = entry.contentRect.width;
    if (state.liveSnapshot) renderTimeline();
  });
  markerResizeObserver.observe(scrubber);
  $("#timeline-playback-toggle").addEventListener("click", (event) => {
    const action = event.currentTarget.dataset.action;
    if (action === "pause") {
      pauseCurrentPlayback();
    } else if (action === "next_episode") {
      const options = playbackSettings?.episodeOptions() || {};
      command("next_episode", {
        sampling_mode: options.sampling_mode,
        driver: "policy",
        enabled_termination_conditions: options.enabled_termination_conditions,
      });
    } else {
      playFromCurrentPosition();
    }
  });
  $("#timeline-reset").addEventListener("click", () => {
    const options = playbackSettings?.episodeOptions() || {};
    command("reset_episode", {
      seed: options.seed,
      enabled_termination_conditions: options.enabled_termination_conditions,
    });
  });
  $("#playback-settings-toggle").addEventListener("click", (event) => {
    const menu = $("#playback-settings-menu");
    const opening = menu.hidden;
    menu.hidden = true;
    event.currentTarget.setAttribute("aria-expanded", String(opening));
    if (opening) {
      positionMenu(menu, event.currentTarget);
    }
  });
  $("#playback-settings-close").addEventListener("click", () => {
    $("#playback-settings-menu").hidden = true;
    $("#playback-settings-toggle").setAttribute("aria-expanded", "false");
  });
  const selectStep = (step) => { void inspectStep(step); };
  scrubber.addEventListener("input", (event) => selectStep(Number(event.target.value)));
  scrubber.addEventListener("keydown", (event) => {
    if (event.code !== "Space" || event.repeat) return;
    event.preventDefault();
    $("#timeline-playback-toggle").click();
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
  playbackSettings = mountPlaybackSettings({
    services: {
      getState: () => state,
      command,
    },
    idPrefix: "player-playback",
  });
  $("#playback-settings-content").append(playbackSettings.element);
  setDetachedLayout();
  bindPanelLayout();
  bindWorkspaceMenus();
  bindWorkspaceSync();
  bindTimeline();
  $("#inspect-active").addEventListener("click", () => {
    $("#player-menu").hidden = true;
    $("#more-toggle").setAttribute("aria-expanded", "false");
    void openContractInspection("/api/playback/inspection", {
      preferredDocument: "goal",
    }).catch((error) => showToast(`Contract viewer failed: ${error.message || error}`, true));
  });
  $("#publish-episode").addEventListener("click", () => {
    $("#player-menu").hidden = true;
    $("#more-toggle").setAttribute("aria-expanded", "false");
    void openPublicationDialog();
  });
  $("#publication-close").addEventListener("click", () => {
    $("#publication-dialog").close();
  });
  $("#publication-dialog").addEventListener("close", () => {
    clearInterval(state.publicationPoll);
    state.publicationPoll = null;
    const video = $("#publication-video");
    video.pause();
    video.removeAttribute("src");
    video.load();
  });
  $("#publication-check").addEventListener("click", () => {
    void checkPublicationCredentials().catch((error) => showToast(error.message, true));
  });
  $("#publication-authorize-youtube").addEventListener("click", async () => {
    try {
      const result = await publicationApi("/api/publication/oauth/start", { method: "POST" });
      youtubeOAuthPopup = window.open(
        result.authorization_url,
        "gradlab-youtube-oauth",
        "popup,width=620,height=760",
      );
      if (!youtubeOAuthPopup) {
        throw new Error("The browser blocked the YouTube authorization window.");
      }
      showToast("Finish YouTube authorization in the popup.");
    } catch (error) {
      showToast(error.message || String(error), true);
    }
  });
  window.addEventListener("message", (event) => {
    if (
      event.origin !== location.origin
      || event.source !== youtubeOAuthPopup
      || event.data?.type !== "gradlab-youtube-oauth-complete"
    ) {
      return;
    }
    youtubeOAuthPopup = null;
    void checkPublicationCredentials().then((result) => {
      const youtube = result?.youtube || {};
      showToast(
        youtube.ready
          ? `YouTube authorized as ${youtube.channel_title}.`
          : (youtube.message || "YouTube authorization needs attention."),
        !youtube.ready,
      );
    }).catch((error) => showToast(error.message || String(error), true));
  });
  $("#publication-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const submit = $("#publication-submit");
    submit.disabled = true;
    try {
      const result = await publicationApi("/api/publication/admit", {
        method: "POST",
        body: {
          ...publicationSettings(),
        },
      });
      renderPublicationJob(result.job);
      showToast(result.created ? "Combined publication queued." : "Publication is already queued.");
    } catch (error) {
      submit.disabled = false;
      showToast(error.message || String(error), true);
    }
  });
  ["#publication-privacy", "#publication-thumbnail-time", "#publication-tags", "#publication-note", "#publication-feature"].forEach((selector) => {
    $(selector).addEventListener("change", () => {
      void refreshPublicationPreview().catch((error) => showToast(error.message || String(error), true));
    });
  });
  const publicationAction = async (action, body) => {
    const jobId = state.publicationJob?.job_id;
    if (!jobId) return;
    try {
      const job = await publicationApi(
        `/api/publication/jobs/${encodeURIComponent(jobId)}/${action}`,
        { method: "POST", ...(body === undefined ? {} : { body }) },
      );
      renderPublicationJob(job);
    } catch (error) {
      showToast(error.message || String(error), true);
    }
  };
  $("#publication-retry").addEventListener("click", () => void publicationAction("retry"));
  $("#publication-cancel").addEventListener("click", () => void publicationAction("cancel"));
  $("#publication-cleanup").addEventListener("click", () => void publicationAction("cleanup"));
  $("#publication-resolve").addEventListener("click", () => {
    const videoId = window.prompt("YouTube video id from the admitted channel:", "");
    if (videoId) void publicationAction("resolve", { video_id: videoId.trim() });
  });
}

panelRuntime = new PanelRuntime({
  definitionFor: panelDefinition,
  isSuspended: panelSuspended,
  container: $("#dashboard"),
  services: {
    getState: () => state,
    setRgbEnabled(enabled) {
      command("set_fps", {
        fps: Number(state.liveSnapshot?.session?.target_fps || 0),
        rgb_enabled: enabled,
      });
    },
    send,
    command,
    inspectSequence,
    async loadRewardHistory(episodeId, first, last) {
      const query = new URLSearchParams({ epoch: state.sessionEpoch, episode_id: episodeId });
      query.set("first", first);
      if (last !== null) query.set("last", last);
      const response = await fetch(`/api/playback/reward-history?${query}`, { headers: { Authorization: `Bearer ${token}` } });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "Unable to load rewards");
      return result;
    },
    async loadEvents(episodeId, last) {
      const query = new URLSearchParams({ epoch: state.sessionEpoch, episode_id: episodeId });
      if (last !== null) query.set("last", last);
      const response = await fetch(`/api/playback/event-history?${query}`, { headers: { Authorization: `Bearer ${token}` } });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "Unable to load events");
      return result;
    },
    inspectStep,
    setRewardReference,
    setChartRange,
    retryChartHistory: () => chartHistory.retry(),
    showToast,
    setAttributionPreference: (config) => {
      state.attributionPreference = {
        mode: String(config?.mode || "gradcam"),
        interval: Math.max(1, Number(config?.interval) || 1),
      };
    },
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
  onLayout: (panel, name, placement, gridItem, definition) => {
    gridStack.update(gridItem, gridWidgetFor(name, placement));
    panel.classList.toggle("panel-disabled", !definition.enabled);
    const enabled = panel.querySelector("[data-panel-enabled]");
    if (enabled) {
      enabled.checked = definition.enabled;
    }
  },
  onUnmount: (_panel, _name, gridItem) => {
    gridStack.removeWidget(gridItem, false, false);
  },
  onError: (name, error) => {
    console.error(`Panel ${name} failed`, error);
    showToast(`${panelLabel(name)} panel failed to load.`, true);
  },
});

window.addEventListener("resize", () => {
  fitGridToViewport();
  panelRuntime.resize();
});
const trajectoryControls = mountTrajectoryControls({
  command,
  getState: () => state,
  toast: showToast,
  request: async (path, body) => {
    const response = await fetch(path, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "X-Gradlab-Client": state.clientId,
        "X-Gradlab-Control-Epoch": String(state.controlEpoch),
        "Content-Type": "application/octet-stream",
      },
      body,
    });
    const result = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
    if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
    return result;
  },
});
initWorkspace();
updateControlState();
connect();

// Fullscreen is a local processing override, never a persisted layout edit.
document.addEventListener("fullscreenchange", () => {
  updateChartDemand();
  panelRuntime.resetFrames();
  send({ type: "subscribe", subscriptions: subscriptions(), processing: processing() });
  refreshPanels();
});
