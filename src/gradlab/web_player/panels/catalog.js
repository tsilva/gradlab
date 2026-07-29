export const FRAME_GAME = 1;
export const FRAME_OBSERVATION = 2;

export const PANEL_TYPES = Object.freeze({
  game: {
    module: "./game.js",
    minimum: { w: 4, h: 8 },
    subscriptions: ["game"],
    frameKinds: [FRAME_GAME],
    singleton: true,
  },
  controls: {
    module: "./controls.js",
    minimum: { w: 2, h: 4 },
    subscriptions: [],
    frameKinds: [],
    singleton: true,
  },
  observation: {
    module: "./observation.js",
    minimum: { w: 2, h: 4 },
    subscriptions: ["observation"],
    frameKinds: [FRAME_OBSERVATION],
    singleton: true,
  },
  telemetry: {
    module: "./telemetry-panel.js",
    minimum: { w: 2, h: 4 },
    subscriptions: [],
    frameKinds: [],
    singleton: false,
  },
  events: {
    module: "./events.js",
    minimum: { w: 2, h: 4 },
    subscriptions: [],
    frameKinds: [],
    singleton: true,
  },
  raw: {
    module: "./raw.js",
    minimum: { w: 2, h: 4 },
    subscriptions: [],
    frameKinds: [],
    singleton: true,
  },
});

const SINGLE_LAYOUT = Object.freeze({
  game: { x: 0, y: 0, w: 7, h: 15, visible: true, window: "main" },
  controls: { x: 7, y: 0, w: 2, h: 15, visible: true, window: "main" },
  policy: { x: 9, y: 0, w: 3, h: 15, visible: true, window: "main" },
  value: { x: 0, y: 15, w: 4, h: 7, visible: true, window: "main" },
  "step-reward": { x: 4, y: 15, w: 4, h: 7, visible: true, window: "main" },
  "episode-return": { x: 8, y: 15, w: 4, h: 7, visible: true, window: "main" },
  actions: { x: 0, y: 22, w: 4, h: 8, visible: false, window: "main" },
  observation: { x: 4, y: 22, w: 5, h: 8, visible: false, window: "main" },
  signals: { x: 9, y: 22, w: 3, h: 8, visible: false, window: "main" },
  events: { x: 0, y: 30, w: 4, h: 7, visible: false, window: "main" },
  raw: { x: 4, y: 30, w: 8, h: 7, visible: false, window: "main" },
});

const PAIRED_LAYOUT = Object.freeze({
  game: { x: 0, y: 0, w: 9, h: 15, visible: true, window: "main" },
  controls: { x: 9, y: 0, w: 3, h: 15, visible: true, window: "main" },
  policy: { x: 0, y: 0, w: 6, h: 8, visible: true, window: "stats" },
  actions: { x: 6, y: 0, w: 6, h: 8, visible: true, window: "stats" },
  value: { x: 0, y: 8, w: 4, h: 7, visible: true, window: "stats" },
  "step-reward": { x: 4, y: 8, w: 4, h: 7, visible: true, window: "stats" },
  "episode-return": { x: 8, y: 8, w: 4, h: 7, visible: true, window: "stats" },
  observation: { x: 0, y: 15, w: 6, h: 8, visible: true, window: "stats" },
  signals: { x: 6, y: 15, w: 3, h: 8, visible: true, window: "stats" },
  events: { x: 9, y: 15, w: 3, h: 8, visible: true, window: "stats" },
  raw: { x: 0, y: 23, w: 12, h: 7, visible: true, window: "stats" },
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
            "policy/selected-q-value",
            "policy/entropy",
            "policy/log-probability",
            "policy/program",
          ],
        },
        { kind: "distribution", metric: "policy/distribution" },
        { kind: "distribution", metric: "policy/q-values" },
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
          foot: "V(s) is expected discounted future policy reward; G(s) is this trajectory’s realized discounted future reward—not its success flag or cumulative episode return.",
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
  actions: {
    type: "telemetry",
    title: "Action history",
    config: {
      blocks: [
        { kind: "histogram", metric: "action/executed" },
      ],
    },
  },
  observation: {
    type: "observation",
    title: "Observation and attribution",
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

export function defaultPanelInstances({ paired = false } = {}) {
  const layout = paired ? PAIRED_LAYOUT : SINGLE_LAYOUT;
  return Object.fromEntries(
    Object.entries(BUILTIN_PANEL_PRESETS).map(([id, preset]) => [
      id,
      {
        ...clone(preset),
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
    builtin: Boolean(instance.builtin),
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
    panelDefinition(workspace, id)?.subscriptions.forEach((subscription) => {
      values.add(subscription);
    });
  });
  return [...values];
}
