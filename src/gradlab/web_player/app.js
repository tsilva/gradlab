import { frameScheduler } from "./chart-transport.js";
import { createPlaybackInspection, hasIndependentInference } from "./playback-inspection.js";
import { rewardReferenceStore } from "./panels/reward-reference.js";
import { createChartHistory } from "./chart-history.js";
import { usesChartHistory } from "./panels/chart-status.js";
import { bindTimelineRange } from "./chart-range.js";
import {
  FRAME_GAME,
  PANEL_TYPES,
  panelDefinition,
  panelLabels,
  panelProcessing,
  panelSubscriptions,
} from "./panels/catalog.js";
import { episodeReport } from "./episode-report.js";
import { timelineEventMarkers } from "./episode-timeline.js";
import { eventColorFill, eventLabels } from "./event-colors.js";
import { mountPlaybackSettings } from "./playback-settings.js";
import { CheckpointSelection } from "./checkpoint-selection.js";
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
  socket: null,
  connected: false,
  clientId: null,
  attributionCommand: null,
  attributionPreference: { mode: "gradcam", interval: 1 },
  cnnCaptureCommand: null,
  hasControl: false,
  publicationAuthority: false,
  publicationCapability: null,
  controlEpoch: 0,
  publicationCurrent: null,
  publicationJob: null,
  publicationPoll: null,
  mode: null,
  lastStatus: null,
  actionNamesKey: "",
  workspaceId,
  windowId,
  layout: null,
  selectedPanel: null,
  activeWindows: new Map(),
  applicationSnapshot: null,
  workspaceReady: false,
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
    (name) => name !== "game" || inspection.view.rgbEnabled !== false,
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
    .map((definition) => inspection.view.rgbEnabled === false
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

const checkpointSelection = new CheckpointSelection(command);
checkpointSelection.subscribe(({ loading }) => {
  const mask = $("#checkpoint-loading-mask");
  if (loading && mask.hidden) document.activeElement?.blur?.();
  mask.hidden = !loading;
  document.body.classList.toggle("checkpoint-loading", loading);
  if (loading) document.body.setAttribute("aria-busy", "true");
  else document.body.removeAttribute("aria-busy");
});

function updateConnection(label, kind = "") {
  const badge = $("#connection-status");
  badge.textContent = label;
  badge.className = `sync-status ${kind}`.trim();
  badge.hidden = label === "Synced" && !kind;
}

function resetSession(epoch) {
  state.attributionCommand = null;
  state.cnnCaptureCommand = null;
  inspection.reset(epoch);
  chartHistory.updateContext({ epoch: inspection.view.sessionEpoch });
}

async function ensureSourceBrowser() {
  if (sourceBrowser) return sourceBrowser;
  if (!sourceBrowserPromise) {
    sourceBrowserPromise = import("./sources/browser.js").then(({ SourceBrowser }) => {
      sourceBrowser = new SourceBrowser($("#source-browser"), $("#source-breadcrumbs"), {
        token,
        command,
        getState: playerState,
        showToast,
        checkpointNavigationRoot: $("#checkpoint-navigation"),
        selection: checkpointSelection,
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
  const current = state.applicationSnapshot || inspection.view.liveSnapshot || {};
  const snapshot = {
    ...current,
    app: {
      ...(current.app || {}),
      phase: "selecting",
      message: "",
      error: "",
      route: { ...route },
      has_active_runner: Boolean(
        current?.app?.has_active_runner || checkpointSelection.view.backgroundSnapshot,
      ),
    },
  };
  state.applicationSnapshot = snapshot;
  renderSourceMode(snapshot);
}

function renderSourceMode(snapshot = null) {
  const { route, sourceMode } = checkpointSelection.view;
  const activeCheckpointRoute = (
    !sourceMode
    && route?.checkpoint_id
  );
  const activeRecordingRoute = (
    !sourceMode
    && snapshot?.mode === "trajectory"
    && route?.environment_id
  );
  document.body.classList.toggle("source-selection", sourceMode);
  $("#source-browser").hidden = !sourceMode;
  $("#checkpoint-navigation").hidden = Boolean(sourceMode || !activeCheckpointRoute);
  $("#page-title").hidden = Boolean(sourceMode || activeRecordingRoute);
  $("#source-back").hidden = Boolean(
    sourceMode
    || activeRecordingRoute
    || !(snapshot?.app?.has_active_runner || inspection.view.liveSnapshot?.app?.has_active_runner)
  );
  $("#more-toggle").hidden = sourceMode;
  $("#inspect-active").hidden = !(
    snapshot?.app?.has_active_runner || inspection.view.liveSnapshot?.app?.has_active_runner
  );
  if (!sourceMode) {
    const expected = snapshot;
    if (activeCheckpointRoute || activeRecordingRoute) {
      if (sourceBrowser) {
        sourceBrowser.renderActiveBreadcrumbs(expected);
        $("#source-breadcrumbs").hidden = Boolean(activeCheckpointRoute);
      } else {
        void ensureSourceBrowser().then((browser) => {
          if (!checkpointSelection.view.sourceMode && state.applicationSnapshot === expected) {
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
    if (checkpointSelection.view.sourceMode && state.applicationSnapshot === expected) browser.render(expected);
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
    inspection.updateConnection({ connected: false, hasControl: false });
    state.publicationAuthority = false;
    state.publicationCapability = null;
    checkpointSelection.terminate();
    updateConnection("Disconnected", "error");
    updateControlState();
  });
  socket.addEventListener("error", () => {
    checkpointSelection.terminate();
    updateConnection("Connection error", "error");
  });
}

function handleMessage(message) {
  if (message.type === "welcome") {
    state.connected = true;
    state.clientId = message.client_id;
    inspection.updateConnection({ connected: true, hasControl: state.hasControl, historyLimit: message.history_limit });
    updateConnection("Synced", "");
    return;
  }
  if (message.type === "history") {
    inspection.receiveHistory(message);
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
    checkpointSelection.receive(message);
    resetSession(message.session_epoch);
    return;
  }
  if (message.type === "snapshot") {
    const epoch = Number(message.session_epoch || 0);
    if (epoch !== inspection.view.sessionEpoch) resetSession(epoch);
    const selectionResult = checkpointSelection.receive(message);
    if (selectionResult.error) showToast(selectionResult.error, true);
    if (selectionResult.background) {
      state.hasControl = Boolean(message.control?.has_control);
      inspection.updateConnection({ hasControl: state.hasControl });
      state.controlEpoch = Number(message.control_epoch || 0);
      trajectoryControls.render();
      return;
    }
    state.applicationSnapshot = message;
    state.hasControl = Boolean(message.control?.has_control);
    inspection.updateConnection({ hasControl: state.hasControl });
    state.controlEpoch = Number(message.control_epoch || 0);
    updatePublicationButton();
    if (typeof message.session?.rgb_enabled === "boolean" && inspection.view.rgbEnabled !== message.session.rgb_enabled) {
      void inspection.setFrameDemand({ rgbEnabled: message.session.rgb_enabled });
      send({ type: "subscribe", subscriptions: subscriptions(), processing: processing() });
      refreshPanels();
    }
    if (message.app && message.app.phase !== "active") {
      renderSourceMode(message);
      updateControlState();
      return;
    }
    renderSourceMode(message);
    void inspection.admitSnapshot(message, checkpointSelection.presentationFor(message));
    return;
  }
  if (message.type === "command_result") {
    inspection.commandResult(message);
    if (message.id === state.attributionCommand?.id) state.attributionCommand = null;
    if (message.id === state.cnnCaptureCommand?.id) state.cnnCaptureCommand = null;
    checkpointSelection.receive(message);
    if (!message.ok) showToast(message.error || "Command failed", true);
    return;
  }
  if (message.type === "error") showToast(message.error || "Player error", true);
}

function updatePublicationButton() {
  const button = $("#publish-episode");
  if (!button) return;
  const snapshot = state.applicationSnapshot || inspection.view.liveSnapshot;
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

async function handleFrame(buffer) {
  if (buffer.byteLength <= FRAME_HEADER_BYTES) return;
  const view = new DataView(buffer);
  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 4));
  if (magic !== "RLP3") return;
  await inspection.receiveFrame({
    epoch: Number(view.getBigUint64(8)), sequence: Number(view.getBigUint64(16)),
    generation: Number(view.getBigUint64(24)), kind: view.getUint8(4),
    blob: new Blob([buffer.slice(FRAME_HEADER_BYTES)], { type: "image/png" }),
  });
}

function episodeForSnapshot(snapshot) {
  const episode = snapshot?.transition?.episode ?? snapshot?.session?.episode;
  return episode == null ? null : Number(episode);
}

function currentEpisodeHistory() { return inspection.view.currentHistory; }

function setRewardReference(step) {
  if (!Number.isInteger(step) || step !== inspection.view.snapshot?.transition?.step) return;
  rewardReferences.set(inspection.view.snapshot, inspection.view.sessionEpoch);
  panelRuntime?.renderHistory(currentEpisodeHistory(), inspection.view.snapshot, panelView());
}

function updateChartContext() {
  chartHistory.updateContext({
    epoch: inspection.view.sessionEpoch,
    episodeId: inspection.view.liveSnapshot?.trajectory?.episode_id,
    episode: episodeForSnapshot(inspection.view.liveSnapshot),
    lastStep: inspection.view.liveSnapshot?.trajectory?.last_step,
    liveHistory: inspection.view.history,
    throughStep: inspection.view.snapshot?.transition?.step ?? 0,
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
    rewardReference: rewardReferences.get(inspection.view.snapshot, inspection.view.sessionEpoch),
    chartRange: chart.range,
    inspection: inspection.view.inspectionSequence !== null,
    sessionEpoch: inspection.view.sessionEpoch,
    selectedSequence: inspection.view.inspectionSequence ?? inspection.view.snapshot?.sequence ?? null,
    liveSequence: inspection.view.liveSnapshot?.sequence ?? null,
  };
}

function hideGoExploreValuePanel(snapshot) {
  if (snapshot?.policy?.provenance?.search_algorithm_id !== "go-explore") return false;
  const panel = state.layout?.panels?.value;
  if (!panel?.builtin || !panel.placement.visible) return false;
  panel.placement.visible = false;
  persistLayout();
  return true;
}

const inspection = createPlaybackInspection({
  windowId, command, send,
  peer: message => workspaceChannel?.postMessage(message),
  async fetchStep({ epoch, episode_id, step, rgbEnabled }, { signal } = {}) {
    const query = new URLSearchParams({ epoch, episode_id, step });
    if (rgbEnabled === false) query.set("rgb", "off");
    const response = await fetch(`/api/playback/recorded-step?${query}`, {
      signal, headers: { Authorization: `Bearer ${token}` },
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Unable to load the recorded step");
    return payload;
  },
  prepareFrame: (...args) => panelRuntime.prepareFrame(...args),
  renderFrame: (...args) => panelRuntime.renderFrame(...args),
  resetFrames: () => panelRuntime?.resetFrames(),
  presented: (ticket, snapshot) => checkpointSelection.presented(ticket, snapshot),
  onError: error => showToast(error.message, true),
});
inspection.subscribe(view => {
  updateChartContext();
  // Source admission belongs to CheckpointSelection. Inspection can never
  // bring a background runner over a locally selected discovery route.
  if (!view.snapshot || !panelRuntime || checkpointSelection.view.sourceMode) return;
  if (hideGoExploreValuePanel(view.liveSnapshot)) void applyLayout();
  renderSnapshot();
  if (view.inspectionSequence !== null) panelRuntime.invoke("controls", "render", view.liveSnapshot);
  renderHistory();
  updateControlState();
  syncAttributionToPanel();
  syncCnnCaptureToPanel();
});

// Existing panel/settings services consume a read-only projection of inspection.
function playerState() { return { ...state, ...inspection.view }; }

function send(value) {
  if (state.socket?.readyState === WebSocket.OPEN) state.socket.send(JSON.stringify(value));
}

function command(name, payload = {}) {
  if (!state.hasControl) {
    showToast("This window is an observer. Choose Control here first.", true);
    return null;
  }
  if (name === "set_fps") payload = { rgb_enabled: inspection.view.rgbEnabled !== false, ...payload };
  const id = crypto.randomUUID();
  send({
    type: "command",
    id,
    name,
    payload,
    expected_revision: inspection.view.liveSnapshot?.revision ?? null,
  });
  return id;
}

function syncAttributionToPanel() {
  if (inspection.view.liveSnapshot?.mode === "trajectory") return;
  const panel = state.layout?.panels?.attribution;
  const attribution = inspection.view.liveSnapshot?.session?.attribution;
  if (
    !panel
    || panel.placement?.window !== state.windowId
    || !attribution
  ) return;
  const supported = inspection.view.liveSnapshot?.policy?.attribution?.supported_modes;
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
  if (inspection.view.liveSnapshot?.mode === "trajectory") return;
  const panel = state.layout?.panels?.cnn;
  const cnn = inspection.view.liveSnapshot?.session?.cnn;
  if (
    !panel
    || panel.placement?.window !== state.windowId
    || !cnn
  ) return;
  const desired = Boolean(panel.enabled && panel.placement.visible);
  const layers = inspection.view.liveSnapshot?.policy?.cnn?.layers;
  if (desired && (!Array.isArray(layers) || !layers.length)) return;
  if (Boolean(cnn.enabled) === desired) {
    if (state.cnnCaptureCommand?.desired === desired) state.cnnCaptureCommand = null;
    return;
  }
  if (state.cnnCaptureCommand?.desired === desired || !state.hasControl) return;
  const id = command("set_cnn_inspection", { enabled: desired });
  if (id) state.cnnCaptureCommand = { id, desired };
}

function updateTimelinePlaybackControl() {
  const playbackToggle = $("#timeline-playback-toggle");
  const playbackIcon = $("#timeline-playback-icon");
  if (!playbackToggle || !playbackIcon) return;
  const session = inspection.view.liveSnapshot?.session || inspection.view.snapshot?.session || {};
  const presentation = transportPresentation({
    running: inspection.view.running,
    replaying: inspection.view.replayingInspection,
    independentInference: hasIndependentInference(inspection.view.liveSnapshot),
    hasControl: state.hasControl,
    canReplay: inspection.view.canReplay || Boolean(inspection.view.liveSnapshot?.trajectory?.imported && session.awaiting_next_episode),
    session,
    recording: (inspection.view.liveSnapshot?.mode || inspection.view.snapshot?.mode) === "recording",
  });
  playbackToggle.dataset.action = presentation.action;
  playbackToggle.disabled = presentation.disabled;
  playbackToggle.title = presentation.reason;
  playbackToggle.classList.toggle("primary", presentation.action !== "pause");
  playbackToggle.setAttribute("aria-label", presentation.label);
  setSvgUseHref(playbackIcon, `/assets/tabler-icons.svg#ti-${presentation.icon}`);
  const reset = $("#timeline-reset");
  if (reset) {
    const mode = inspection.view.liveSnapshot?.mode || inspection.view.snapshot?.mode;
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
    inspection.view.snapshot || inspection.view.liveSnapshot,
  );
}

function renderPlaybackEvidenceStatus(snapshot) {
  const status = $("#playback-evidence-status");
  if (!status || !snapshot || checkpointSelection.view.sourceMode || state.windowId !== "main") {
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
  const snapshot = inspection.view.snapshot;
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
  if (inspection.view.inspectionSequence === null && snapshot.status_message && snapshot.status_message !== state.lastStatus) {
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
  if (broadcast) workspaceChannel?.postMessage({ type: "chart-range", source: state.windowId, epoch: inspection.view.sessionEpoch, episode: inspection.view.liveSnapshot?.trajectory?.episode_id, range });
  renderTimeline();
}

const chartHistory = createChartHistory({ token, onChange: () => scheduleHistoryRender() });

const scheduleHistoryRender = frameScheduler(() => {
  if (!inspection.view.snapshot || !panelRuntime) return;
  panelRuntime.renderHistory(currentEpisodeHistory(), inspection.view.snapshot, panelView());
  fitGridToViewport();
  renderTimeline();
});

function renderHistory() {
  updateChartContext();
  scheduleHistoryRender();
}

function renderTimeline() {
  const scrubber = $("#timeline-scrubber");
  if (!scrubber) return;
  const trajectory = inspection.view.liveSnapshot?.trajectory;
  const range = inspection.view.range;
  const selected = inspection.view.seekingStep ?? Number(trajectory?.imported ? trajectory.current_step
    : inspection.view.snapshot?.transition?.step ?? inspection.view.snapshot?.session?.step ?? range?.first ?? 0);
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
  $("#timeline").setAttribute("aria-busy", String(inspection.view.seekingStep !== null));
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
  const interesting = timelineEventMarkers(inspection.view.eventPoints, range, markerSlots);
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
  const route = (state.applicationSnapshot || inspection.view.liveSnapshot)?.app?.route || {};
  const environmentId = String(
    route.environment_id || inspection.view.liveSnapshot?.session?.env_id || "",
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
  void inspection.setFrameDemand({ kinds: enabledPanelDefinitions().flatMap(definition => definition.frameKinds) });
  if (inspection.view.snapshot) {
    panelRuntime.renderSnapshot(inspection.view.snapshot, panelView());
    panelRuntime.renderHistory(currentEpisodeHistory(), inspection.view.snapshot, panelView());
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
      panelRuntime?.renderHistory(currentEpisodeHistory(), inspection.view.snapshot, panelView());
    }
  });
  if (workspaceChannel) {
    workspaceChannel.addEventListener("message", (event) => {
      const message = event.data || {};
      if (message.type === "chart-range" && message.source !== state.windowId) {
        if (message.epoch === inspection.view.sessionEpoch && message.episode === inspection.view.liveSnapshot?.trajectory?.episode_id) setChartRange(message.range, { broadcast: false });
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
      } else if (["inspection-cursor", "inspection-frame-request", "inspection-frame"].includes(message.type)) {
        inspection.receivePeer(message);
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
    inspection.dispose();
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
    if (inspection.view.liveSnapshot) renderTimeline();
  });
  markerResizeObserver.observe(scrubber);
  $("#timeline-playback-toggle").addEventListener("click", (event) => {
    const action = event.currentTarget.dataset.action;
    if (action === "pause") {
      inspection.pause();
    } else if (action === "next_episode") {
      const options = playbackSettings?.episodeOptions() || {};
      command("next_episode", {
        sampling_mode: options.sampling_mode,
        driver: "policy",
        enabled_termination_conditions: options.enabled_termination_conditions,
      });
    } else {
      inspection.play();
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
  const selectStep = (step) => { void inspection.selectStep(step); };
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
      snapshot: inspection.view.snapshot,
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
      getState: playerState,
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
    getState: playerState,
    setRgbEnabled(enabled) {
      command("set_fps", {
        fps: Number(inspection.view.liveSnapshot?.session?.target_fps || 0),
        rgb_enabled: enabled,
      });
    },
    send,
    command,
    inspectSequence: inspection.selectSequence,
    async loadRewardHistory(episodeId, first, last) {
      const query = new URLSearchParams({ epoch: inspection.view.sessionEpoch, episode_id: episodeId });
      query.set("first", first);
      if (last !== null) query.set("last", last);
      const response = await fetch(`/api/playback/reward-history?${query}`, { headers: { Authorization: `Bearer ${token}` } });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "Unable to load rewards");
      return result;
    },
    async loadEvents(episodeId, last) {
      const query = new URLSearchParams({ epoch: inspection.view.sessionEpoch, episode_id: episodeId });
      if (last !== null) query.set("last", last);
      const response = await fetch(`/api/playback/event-history?${query}`, { headers: { Authorization: `Bearer ${token}` } });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "Unable to load events");
      return result;
    },
    inspectStep: inspection.selectStep,
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
  inspectStep: inspection.selectStep,
  getState: playerState,
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
