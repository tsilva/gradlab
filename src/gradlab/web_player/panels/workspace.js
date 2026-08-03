import {
  BUILTIN_PANEL_PRESETS,
  PANEL_TYPES,
  defaultPanelInstances,
} from "./catalog.js";

export const WORKSPACE_VERSION = 6;
export const CUSTOM_PANEL_ID = /^panel-[0-9a-f]{8}-[0-9a-f-]{27}$/;
const BLOCK_KINDS = new Set([
  "stats",
  "line",
  "histogram",
  "distribution",
  "namespace-explorer",
]);

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function clamp(value, minimum, maximum, fallback) {
  const numeric = Number(value);
  return Math.max(
    minimum,
    Math.min(maximum, Number.isFinite(numeric) ? numeric : fallback),
  );
}

function cleanTitle(value, fallback) {
  const title = String(value || "").replace(/[\u0000-\u001f\u007f]/g, " ").trim();
  return title ? title.slice(0, 80) : fallback;
}

function cleanMetric(value) {
  const metric = String(value || "").trim();
  return metric && metric.length <= 240 ? metric : null;
}

function normalizeBlock(value) {
  if (!value || typeof value !== "object" || !BLOCK_KINDS.has(value.kind)) return null;
  const block = { kind: value.kind };
  if (value.title) block.title = cleanTitle(value.title, "");
  if (value.foot) block.foot = String(value.foot).slice(0, 320);
  if (value.kind === "stats" || value.kind === "line") {
    const metrics = [...new Set(
      (Array.isArray(value.metrics) ? value.metrics : [])
        .map(cleanMetric)
        .filter(Boolean),
    )].slice(0, 12);
    if (!metrics.length) return null;
    block.metrics = metrics;
  } else if (value.kind === "histogram" || value.kind === "distribution") {
    const metric = cleanMetric(value.metric);
    if (!metric) return null;
    block.metric = metric;
  } else {
    const namespace = String(value.namespace || "").trim();
    if (!["signal", "reward-component"].includes(namespace)) return null;
    block.namespace = namespace;
    block.metric = cleanMetric(value.metric) || "";
  }
  return block;
}

export function normalizePanelConfig(value) {
  const blocks = (Array.isArray(value?.blocks) ? value.blocks : [])
    .map(normalizeBlock)
    .filter(Boolean)
    .slice(0, 12);
  return { blocks };
}

function normalizePlacement(value, minimum, fallback) {
  const w = clamp(value?.w, minimum.w, 12, fallback.w);
  const h = clamp(value?.h, minimum.h, 40, fallback.h);
  return {
    x: clamp(value?.x, 0, 12 - w, fallback.x),
    y: clamp(value?.y, 0, 199, fallback.y),
    w,
    h,
    visible: value?.visible === undefined
      ? fallback.visible
      : Boolean(value.visible),
    window: typeof value?.window === "string" && value.window.trim()
      ? value.window.trim().slice(0, 80)
      : fallback.window,
  };
}

function normalizePanel(id, value, fallback) {
  if (!value || typeof value !== "object") return clone(fallback);
  const requestedType = String(value.type || fallback?.type || "");
  const type = PANEL_TYPES[requestedType];
  if (!type) return fallback ? clone(fallback) : null;
  const builtin = Object.hasOwn(BUILTIN_PANEL_PRESETS, id);
  if (!builtin && (!CUSTOM_PANEL_ID.test(id) || requestedType !== "telemetry")) return null;
  if (builtin && requestedType !== BUILTIN_PANEL_PRESETS[id].type) return clone(fallback);
  const defaultPlacement = fallback?.placement || {
    x: 0,
    y: 0,
    w: Math.max(3, type.minimum.w),
    h: Math.max(6, type.minimum.h),
    visible: true,
    window: "main",
  };
  return {
    type: requestedType,
    title: cleanTitle(value.title, fallback?.title || "Telemetry"),
    config: requestedType === "telemetry"
      ? normalizePanelConfig(value.config)
      : {},
    builtin,
    placement: normalizePlacement(value.placement, type.minimum, defaultPlacement),
  };
}

export function createDefaultWorkspace({ paired = false, writer = "" } = {}) {
  return {
    version: WORKSPACE_VERSION,
    revision: { counter: 0, writer: String(writer) },
    name: "Default layout",
    panels: defaultPanelInstances({ paired }),
  };
}

export function normalizeWorkspace(value, { paired = false, writer = "" } = {}) {
  const fallback = createDefaultWorkspace({ paired, writer });
  if (!value || typeof value !== "object" || value.version !== WORKSPACE_VERSION) {
    return fallback;
  }
  const panels = {};
  Object.entries(fallback.panels).forEach(([id, panel]) => {
    panels[id] = normalizePanel(id, value.panels?.[id], panel);
  });
  Object.entries(value.panels || {}).forEach(([id, panel]) => {
    if (Object.hasOwn(panels, id)) return;
    const normalized = normalizePanel(id, panel, null);
    if (normalized) panels[id] = normalized;
  });
  return {
    version: WORKSPACE_VERSION,
    revision: {
      counter: Math.max(0, Number(value.revision?.counter) || 0),
      writer: String(value.revision?.writer || writer).slice(0, 80),
    },
    name: cleanTitle(value.name, fallback.name).slice(0, 48),
    panels,
  };
}

export function compareWorkspaceRevisions(left, right) {
  const counter = Number(left?.counter || 0) - Number(right?.counter || 0);
  if (counter) return counter;
  return String(left?.writer || "").localeCompare(String(right?.writer || ""));
}

export function bumpWorkspaceRevision(workspace, writer) {
  workspace.revision = {
    counter: Number(workspace.revision?.counter || 0) + 1,
    writer: String(writer),
  };
  return workspace.revision;
}

export function createTelemetryInstance({
  id,
  title = "Telemetry",
  config = {
    blocks: [{ kind: "line", metrics: ["reward/shaped"] }],
  },
  window = "main",
  y = 0,
} = {}) {
  return normalizePanel(id, {
    type: "telemetry",
    title,
    config,
    placement: { x: 0, y, w: 4, h: 8, visible: true, window },
  }, null);
}
