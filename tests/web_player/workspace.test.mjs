import assert from "node:assert/strict";
import test from "node:test";

import {
  WORKSPACE_VERSION,
  bumpWorkspaceRevision,
  compareWorkspaceRevisions,
  createDefaultWorkspace,
  createTelemetryInstance,
  normalizeWorkspace,
} from "../../src/gradlab/web_player/panels/workspace.js";
import {
  panelProcessing,
  telemetryPanelProcessing,
} from "../../src/gradlab/web_player/panels/catalog.js";

const CUSTOM_ID = "panel-00000000-0000-4000-8000-000000000000";

test("default workspace is a v8 editable all-panels view without redundant controls", () => {
  const workspace = createDefaultWorkspace({ writer: "main" });
  assert.equal(workspace.version, 8);
  assert.equal(workspace.version, WORKSPACE_VERSION);
  assert.equal(workspace.preset, "all");
  assert.equal(Object.hasOwn(workspace.panels, "reward"), false);
  assert.equal(Object.hasOwn(workspace.panels, "actions"), false);
  assert.deepEqual(
    ["value", "step-reward", "episode-return"].map(
      (id) => workspace.panels[id].config.blocks.length,
    ),
    [1, 1, 1],
  );
  assert.deepEqual(
    ["value", "step-reward", "episode-return"].map(
      (id) => workspace.panels[id].config.blocks[0].kind,
    ),
    ["line", "line", "line"],
  );
  assert.deepEqual(workspace.panels.value.config.blocks[0].metrics, [
    "policy/value",
    "policy/realized-return",
    "policy/value-error",
  ]);
  assert.deepEqual(workspace.panels.game.placement, {
    x: 0,
    y: 0,
    w: 8,
    h: 15,
    visible: true,
    window: "main",
  });
  assert.deepEqual(workspace.panels.policy.placement, {
    x: 8,
    y: 0,
    w: 4,
    h: 15,
    visible: true,
    window: "main",
  });
  assert.deepEqual(workspace.panels.observation.placement, {
    x: 0,
    y: 15,
    w: 4,
    h: 7,
    visible: true,
    window: "main",
  });
  assert.equal(workspace.panels.controls.placement.visible, false);
  assert.deepEqual(workspace.panels["step-reward"].placement, {
    x: 4,
    y: 15,
    w: 3,
    h: 7,
    visible: true,
    window: "main",
  });
  assert.deepEqual(
    ["observation", "step-reward", "episode-return", "events", "value", "signals"]
      .map((id) => {
        const { x, y, w, h } = workspace.panels[id].placement;
        return [id, x, y, w, h];
      }),
    [
      ["observation", 0, 15, 4, 7],
      ["step-reward", 4, 15, 3, 7],
      ["episode-return", 7, 15, 3, 7],
      ["events", 10, 15, 2, 7],
      ["value", 0, 22, 6, 8],
      ["signals", 6, 22, 6, 19],
    ],
  );
  assert.deepEqual(workspace.panels["reward-analysis"].config.blocks, [
    { kind: "reward-breakdown", scope: "step" },
  ]);
  assert.deepEqual(workspace.panels["reward-analysis"].placement, {
    x: 0,
    y: 30,
    w: 6,
    h: 15,
    visible: true,
    window: "main",
  });
  assert.equal(workspace.panels.cnn.type, "cnn");
  assert.equal(workspace.panels.cnn.placement.visible, false);
  assert.equal(workspace.panels.cnn.enabled, false);
  assert.equal(workspace.panels.attribution.type, "attribution");
  assert.equal(workspace.panels.attribution.placement.visible, false);
  assert.equal(workspace.panels.attribution.enabled, false);
  assert.equal(workspace.panels.raw.placement.visible, false);
  assert.ok(
    Object.entries(workspace.panels)
      .filter(([id]) => !["attribution", "cnn"].includes(id))
      .every(([, panel]) => panel.enabled === true),
  );
  assert.ok(
    Object.entries(workspace.panels)
      .filter(([id]) => !["attribution", "cnn", "controls", "raw"].includes(id))
      .every(([, panel]) => panel.placement.visible === true),
  );
});

test("workspace normalization migrates legacy built-in panel titles", () => {
  const workspace = createDefaultWorkspace();
  workspace.panels.observation.title = "Observation";
  workspace.panels.policy.title = "Policy decision";
  workspace.panels.value.title = "Critic history";

  const normalized = normalizeWorkspace(workspace);

  assert.equal(normalized.panels.observation.title, "Input");
  assert.equal(normalized.panels.policy.title, "Action decision");
  assert.equal(normalized.panels.value.title, "Critic history");
});

test("legacy fixed views migrate to all panels without deleting custom panels", () => {
  const workspace = createDefaultWorkspace();
  workspace.panels[CUSTOM_ID] = createTelemetryInstance({ id: CUSTOM_ID, title: "Mine" });
  workspace.preset = "debug";
  workspace.name = "Debug";
  workspace.panels.value.placement.visible = false;
  workspace.panels.attribution.placement.visible = false;
  workspace.panels[CUSTOM_ID].placement.visible = false;

  const normalized = normalizeWorkspace(workspace);

  assert.equal(normalized.preset, "all");
  assert.equal(normalized.name, "All panels");
  assert.equal(normalized.panels.value.placement.visible, true);
  assert.equal(normalized.panels.attribution.placement.visible, false);
  assert.equal(normalized.panels[CUSTOM_ID].placement.visible, false);
});

test("panel processing demand excludes disabled panels and follows telemetry metrics", () => {
  const workspace = createDefaultWorkspace();
  workspace.panels.policy.enabled = false;

  assert.deepEqual(
    new Set(panelProcessing(workspace, ["policy", "value", "step-reward"])),
    new Set(["policy", "critic-calibration", "rewards", "history"]),
  );
  workspace.panels.value.enabled = false;
  assert.deepEqual(
    new Set(panelProcessing(workspace, ["policy", "value", "step-reward"])),
    new Set(["rewards", "history"]),
  );
  assert.deepEqual(
    new Set(telemetryPanelProcessing({
      blocks: [
        { kind: "stats", metrics: ["signal/ammo", "action/policy"] },
        { kind: "reward-breakdown", scope: "episode" },
      ],
    })),
    new Set(["signals", "actions", "reward-accounting", "history"]),
  );
});

test("workspace normalization preserves explicit disabled processing state", () => {
  const workspace = createDefaultWorkspace();
  workspace.panels.value.enabled = false;

  const normalized = normalizeWorkspace(workspace);

  assert.equal(normalized.panels.value.enabled, false);
  delete workspace.panels.value.enabled;
  assert.equal(normalizeWorkspace(workspace).panels.value.enabled, true);
});

test("workspace normalization preserves a saved custom panel arrangement", () => {
  const workspace = createDefaultWorkspace();
  workspace.panels.raw.placement.visible = true;
  workspace.panels.policy.placement = {
    x: 0,
    y: 30,
    w: 6,
    h: 10,
    visible: true,
    window: "main",
  };
  workspace.panels.observation.placement = {
    x: 6,
    y: 30,
    w: 6,
    h: 10,
    visible: true,
    window: "main",
  };

  const normalized = normalizeWorkspace(workspace);

  assert.equal(normalized.panels.raw.placement.visible, true);
  assert.deepEqual(normalized.panels.policy.placement, workspace.panels.policy.placement);
  assert.deepEqual(
    normalized.panels.observation.placement,
    workspace.panels.observation.placement,
  );
});

test("existing v8 workspaces receive reward analysis hidden on the shelf", () => {
  const workspace = createDefaultWorkspace();
  delete workspace.panels["reward-analysis"];

  const normalized = normalizeWorkspace(workspace);

  assert.equal(normalized.panels["reward-analysis"].placement.visible, false);
  assert.equal(normalized.panels["reward-analysis"].builtin, true);
});

test("v7 workspaces add value error only to the old built-in value configuration", () => {
  const workspace = createDefaultWorkspace();
  workspace.version = 7;
  workspace.panels.value.config.blocks[0].metrics = [
    "policy/value",
    "policy/realized-return",
  ];

  const normalized = normalizeWorkspace(workspace);

  assert.equal(normalized.version, 8);
  assert.deepEqual(normalized.panels.value.config.blocks[0].metrics, [
    "policy/value",
    "policy/realized-return",
    "policy/value-error",
  ]);

  workspace.panels.value.config.blocks[0].metrics = ["policy/value"];
  assert.deepEqual(
    normalizeWorkspace(workspace).panels.value.config.blocks[0].metrics,
    ["policy/value"],
  );
});

test("new and upgraded paired workspaces keep the CNN explorer hidden", () => {
  const workspace = createDefaultWorkspace({ paired: true });
  assert.equal(workspace.panels.cnn.placement.visible, false);
  assert.equal(workspace.panels.cnn.enabled, false);
  delete workspace.panels.cnn;

  const normalized = normalizeWorkspace(workspace, { paired: true });

  assert.equal(normalized.panels.cnn.placement.visible, false);
  assert.equal(normalized.panels.cnn.builtin, true);
});

test("new and upgraded paired workspaces keep attribution hidden", () => {
  const workspace = createDefaultWorkspace({ paired: true });
  assert.equal(workspace.panels.attribution.placement.visible, false);
  assert.equal(workspace.panels.attribution.enabled, false);
  delete workspace.panels.attribution;

  const normalized = normalizeWorkspace(workspace, { paired: true });

  assert.equal(normalized.panels.attribution.placement.visible, false);
  assert.equal(normalized.panels.attribution.builtin, true);
});

test("reward breakdown scope persists through workspace normalization", () => {
  const workspace = createDefaultWorkspace();
  workspace.panels["reward-analysis"].config.blocks[0].scope = "episode";

  const normalized = normalizeWorkspace(workspace);

  assert.equal(
    normalized.panels["reward-analysis"].config.blocks[0].scope,
    "episode",
  );
});

test("paired workspace gives every panel in a logical row the same height", () => {
  const workspace = createDefaultWorkspace({ paired: true });
  const rows = [
    ["policy", "value"],
    ["step-reward", "episode-return"],
    ["observation", "signals", "events"],
  ];
  rows.forEach((ids) => {
    const placements = ids.map((id) => workspace.panels[id].placement);
    assert.equal(new Set(placements.map(({ y }) => y)).size, 1, `${ids} y`);
    assert.equal(new Set(placements.map(({ h }) => h)).size, 1, `${ids} h`);
    assert.equal(new Set(placements.map(({ window }) => window)).size, 1, `${ids} window`);
  });
  assert.deepEqual(
    ["policy", "value"].map((id) => workspace.panels[id].placement.w),
    [6, 6],
  );
  assert.deepEqual(
    ["step-reward", "episode-return"].map((id) => workspace.panels[id].placement.w),
    [6, 6],
  );
  assert.deepEqual(workspace.panels.game.placement, {
    x: 0,
    y: 0,
    w: 12,
    h: 15,
    visible: true,
    window: "main",
  });
  assert.equal(workspace.panels.controls.placement.visible, false);
  assert.equal(workspace.panels.raw.placement.visible, false);
  assert.equal(workspace.panels["reward-analysis"].placement.y, 23);
  assert.equal(workspace.panels.attribution.placement.y, 38);
  assert.equal(workspace.panels.cnn.placement.y, 38);
});

test("non-current workspace data is replaced instead of interpreted", () => {
  const workspace = normalizeWorkspace({
    version: 5,
    panels: {
      game: { col: 9, row: 99, w: 1, h: 1 },
    },
  }, { writer: "test" });
  assert.equal(workspace.version, WORKSPACE_VERSION);
  assert.equal(workspace.panels.game.placement.x, 0);
  assert.equal(workspace.revision.writer, "test");
});

test("valid custom telemetry instances survive normalization", () => {
  const workspace = createDefaultWorkspace();
  workspace.panels[CUSTOM_ID] = createTelemetryInstance({
    id: CUSTOM_ID,
    title: "My telemetry",
    config: {
      blocks: [
        {
          kind: "line",
          metrics: ["reward/provider", "reward/shaped", "reward/provider"],
        },
      ],
    },
    y: 44,
  });
  const normalized = normalizeWorkspace(workspace);
  assert.equal(normalized.panels[CUSTOM_ID].title, "My telemetry");
  assert.deepEqual(
    normalized.panels[CUSTOM_ID].config.blocks[0].metrics,
    ["reward/provider", "reward/shaped"],
  );
  assert.equal(normalized.panels[CUSTOM_ID].placement.y, 44);
});

test("unknown, malformed, and non-telemetry custom panels are discarded", () => {
  const workspace = createDefaultWorkspace();
  workspace.panels["not-a-panel-id"] = {
    type: "telemetry",
    title: "Invalid",
    config: { blocks: [] },
    placement: {},
  };
  workspace.panels[CUSTOM_ID] = {
    type: "game",
    title: "Second game",
    config: {},
    placement: {},
  };
  const normalized = normalizeWorkspace(workspace);
  assert.equal(normalized.panels["not-a-panel-id"], undefined);
  assert.equal(normalized.panels[CUSTOM_ID], undefined);
});

test("workspace revisions have a deterministic writer tie-break", () => {
  assert.ok(
    compareWorkspaceRevisions(
      { counter: 3, writer: "b" },
      { counter: 3, writer: "a" },
    ) > 0,
  );
  const workspace = createDefaultWorkspace({ writer: "a" });
  assert.deepEqual(
    bumpWorkspaceRevision(workspace, "window-b"),
    { counter: 1, writer: "window-b" },
  );
});
