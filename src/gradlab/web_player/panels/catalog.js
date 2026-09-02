export const FRAME_GAME = 1;
export const FRAME_OBSERVATION = 2;
export const FRAME_ATTRIBUTION = 3;
export const FRAME_CNN_INSPECTION = 4;

export const PANEL_TYPES = Object.freeze({
  game: {
    module: "./game.js",
    minimum: { w: 4, h: 8 },
    subscriptions: ["game"],
    processing: ["game"],
    frameKinds: [FRAME_GAME],
    singleton: true,
    switchable: false,
  },
  controls: {
    module: "./controls.js",
    minimum: { w: 2, h: 4 },
    subscriptions: [],
    processing: [],
    frameKinds: [],
    singleton: true,
    switchable: false,
  },
  observation: {
    module: "./observation.js",
    minimum: { w: 2, h: 4 },
    subscriptions: ["observation", "attribution", "cnn-inspection"],
    processing: ["observation"],
    frameKinds: [FRAME_OBSERVATION, FRAME_ATTRIBUTION, FRAME_CNN_INSPECTION],
    singleton: true,
    switchable: true,
  },
  attribution: {
    module: "./attribution.js",
    minimum: { w: 2, h: 4 },
    subscriptions: [],
    processing: ["attribution"],
    frameKinds: [],
    singleton: true,
    switchable: true,
  },
  cnn: {
    module: "./cnn.js",
    minimum: { w: 4, h: 8 },
    subscriptions: ["cnn-inspection"],
    processing: ["cnn-inspection"],
    frameKinds: [FRAME_CNN_INSPECTION],
    singleton: true,
    switchable: true,
  },
  telemetry: {
    module: "./telemetry-panel.js",
    minimum: { w: 2, h: 4 },
    subscriptions: [],
    processing: [],
    frameKinds: [],
    singleton: false,
    switchable: true,
  },
  events: {
    module: "./events.js",
    minimum: { w: 2, h: 4 },
    subscriptions: [],
    processing: ["events", "history"],
    frameKinds: [],
    singleton: true,
    switchable: true,
  },
  raw: {
    module: "./raw.js",
    minimum: { w: 2, h: 4 },
    subscriptions: [],
    processing: ["raw"],
    frameKinds: [],
    singleton: true,
    switchable: true,
  },
});

const ALL_PANELS_LAYOUT = Object.freeze({
  game: { x: 0, y: 0, w: 8, h: 15, visible: true, window: "main" },
  controls: { x: 8, y: 0, w: 2, h: 15, visible: false, window: "main" },
  policy: { x: 8, y: 0, w: 4, h: 15, visible: true, window: "main" },
  observation: { x: 0, y: 15, w: 4, h: 8, visible: true, window: "main" },
  value: { x: 4, y: 15, w: 4, h: 8, visible: true, window: "main" },
  signals: { x: 8, y: 15, w: 4, h: 8, visible: true, window: "main" },
  "step-reward": { x: 0, y: 23, w: 4, h: 7, visible: true, window: "main" },
  "episode-return": { x: 4, y: 23, w: 4, h: 7, visible: true, window: "main" },
  events: { x: 8, y: 23, w: 4, h: 7, visible: true, window: "main" },
  raw: { x: 0, y: 30, w: 12, h: 7, visible: true, window: "main" },
  "reward-analysis": { x: 0, y: 37, w: 12, h: 15, visible: true, window: "main" },
  attribution: { x: 0, y: 52, w: 4, h: 15, visible: true, window: "main" },
  cnn: { x: 4, y: 52, w: 8, h: 15, visible: true, window: "main" },
});

const PAIRED_LAYOUT = Object.freeze({
  game: { x: 0, y: 0, w: 12, h: 15, visible: true, window: "main" },
  controls: { x: 9, y: 0, w: 3, h: 15, visible: false, window: "main" },
  policy: { x: 0, y: 0, w: 6, h: 8, visible: true, window: "stats" },
  value: { x: 6, y: 0, w: 6, h: 8, visible: true, window: "stats" },
  "step-reward": { x: 0, y: 8, w: 6, h: 7, visible: true, window: "stats" },
  "episode-return": { x: 6, y: 8, w: 6, h: 7, visible: true, window: "stats" },
  observation: { x: 0, y: 15, w: 6, h: 8, visible: true, window: "stats" },
  signals: { x: 6, y: 15, w: 3, h: 8, visible: true, window: "stats" },
  events: { x: 9, y: 15, w: 3, h: 8, visible: true, window: "stats" },
  raw: { x: 0, y: 23, w: 12, h: 7, visible: true, window: "stats" },
  "reward-analysis": { x: 0, y: 30, w: 12, h: 15, visible: true, window: "stats" },
  attribution: { x: 0, y: 45, w: 4, h: 15, visible: true, window: "stats" },
  cnn: { x: 4, y: 45, w: 8, h: 15, visible: true, window: "stats" },
});

export const BUILTIN_PANEL_PRESETS = Object.freeze({
  game: {
    type: "game",
    title: "Game",
    config: {},
  },
  controls: {
    type: "controls",
    title: "Controls",
    config: {},
  },
  policy: {
    type: "telemetry",
    title: "Policy decision",
    config: {
      blocks: [
        {
          kind: "stats",
          metrics: [
            "policy/mode",
            "action/policy",
            "policy/value",
            "policy/entropy",
            "policy/log-probability",
            "policy/program",
          ],
        },
        { kind: "distribution", metric: "policy/distribution" },
      ],
    },
  },
  value: {
    type: "telemetry",
    title: "Value estimate",
    config: {
      blocks: [
        {
          kind: "line",
          title: "Value estimate vs realized return-to-go",
          metrics: ["policy/value", "policy/realized-return"],
        },
      ],
    },
  },
  "step-reward": {
    type: "telemetry",
    title: "Step reward",
    config: {
      blocks: [
        {
          kind: "line",
          title: "After-action step reward",
          metrics: ["reward/provider", "reward/shaped"],
        },
      ],
    },
  },
  "episode-return": {
    type: "telemetry",
    title: "Episode return",
    config: {
      blocks: [
        {
          kind: "line",
          title: "Episode return",
          metrics: ["reward/return"],
        },
      ],
    },
  },
  "reward-analysis": {
    type: "telemetry",
    title: "Reward analysis",
    config: {
      blocks: [
        { kind: "reward-breakdown", scope: "step" },
      ],
    },
  },
  observation: {
    type: "observation",
    title: "Observation",
    config: {},
  },
  attribution: {
    type: "attribution",
    title: "Attribution",
    config: {},
  },
  cnn: {
    type: "cnn",
    title: "CNN feature explorer",
    config: {},
  },
  signals: {
    type: "telemetry",
    title: "Live signals",
    config: {
      blocks: [
        { kind: "namespace-explorer", namespace: "signal", metric: "" },
      ],
    },
  },
  events: {
    type: "events",
    title: "Events",
    config: {},
  },
  raw: {
    type: "raw",
    title: "Transition inspector",
    config: {},
  },
});

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function metricProcessing(metric) {
  const key = String(metric || "");
  const processing = new Set();
  if (key.startsWith("policy/")) processing.add("policy");
  if (key === "policy/realized-return" || key === "policy/value-error") {
    processing.add("critic-calibration");
    processing.add("rewards");
  }
  if (key === "action/policy") processing.add("actions");
  if (key.startsWith("reward/")) processing.add("rewards");
  if (key.startsWith("reward-component/")) processing.add("reward-accounting");
  if (key.startsWith("signal/")) processing.add("signals");
  if (key === "transition/outcome") processing.add("events");
  return processing;
}

export function telemetryPanelProcessing(config) {
  const processing = new Set();
  (Array.isArray(config?.blocks) ? config.blocks : []).forEach((block) => {
    if (["line", "histogram", "distribution", "namespace-explorer"].includes(block.kind)) {
      processing.add("history");
    }
    if (block.kind === "reward-breakdown") {
      processing.add("reward-accounting");
      if (block.scope === "episode") processing.add("history");
    }
    if (block.kind === "namespace-explorer") {
      processing.add(block.namespace === "reward-component" ? "reward-accounting" : "signals");
    }
    const metrics = block.kind === "stats" || block.kind === "line"
      ? block.metrics
      : [block.metric];
    (Array.isArray(metrics) ? metrics : []).forEach((metric) => {
      metricProcessing(metric).forEach((feature) => processing.add(feature));
    });
  });
  return [...processing];
}

export function defaultPanelInstances({ paired = false } = {}) {
  const layout = paired ? PAIRED_LAYOUT : ALL_PANELS_LAYOUT;
  return Object.fromEntries(
    Object.entries(BUILTIN_PANEL_PRESETS).map(([id, preset]) => [
      id,
      {
        ...clone(preset),
        enabled: !["attribution", "cnn"].includes(id),
        builtin: true,
        placement: clone(layout[id]),
      },
    ]),
  );
}

export function panelDefinition(workspace, id) {
  const instance = workspace?.panels?.[id];
  const type = instance ? PANEL_TYPES[instance.type] : null;
  if (!instance || !type) return null;
  return {
    ...type,
    id,
    label: instance.title,
    title: instance.title,
    type: instance.type,
    config: instance.config,
    enabled: instance.enabled !== false,
    builtin: Boolean(instance.builtin),
    switchable: Boolean(type.switchable),
    processing: instance.type === "telemetry"
      ? telemetryPanelProcessing(instance.config)
      : type.processing,
  };
}

export function panelLabels(workspace) {
  return Object.fromEntries(
    Object.entries(workspace?.panels || {}).map(([id, panel]) => [id, panel.title]),
  );
}

export function panelSubscriptions(workspace, names) {
  const values = new Set(["telemetry"]);
  names.forEach((id) => {
    const definition = panelDefinition(workspace, id);
    if (!definition?.enabled) return;
    definition.subscriptions.forEach((subscription) => {
      values.add(subscription);
    });
  });
  return [...values];
}

export function panelProcessing(workspace, names) {
  const values = new Set();
  names.forEach((id) => {
    const definition = panelDefinition(workspace, id);
    if (!definition?.enabled) return;
    definition.processing.forEach((feature) => values.add(feature));
  });
  return [...values];
}
