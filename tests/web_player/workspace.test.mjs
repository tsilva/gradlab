import assert from "node:assert/strict";
import test from "node:test";

import {
  WORKSPACE_VERSION,
  applyWorkspacePreset,
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

test("default workspace is a v7 Watch view without a standalone action panel", () => {
  const workspace = createDefaultWorkspace({ writer: "main" });
  assert.equal(workspace.version, 7);
  assert.equal(workspace.version, WORKSPACE_VERSION);
  assert.equal(workspace.preset, "watch");
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
  assert.deepEqual(workspace.panels.game.placement, {
    x: 0,
    y: 0,
    w: 12,
    h: 15,
    visible: true,
    window: "main",
  });
  assert.equal(workspace.panels.controls.placement.visible, false);
  assert.deepEqual(workspace.panels["step-reward"].placement, {
    x: 4,
    y: 15,
    w: 4,
    h: 7,
    visible: false,
    window: "main",
  });
  assert.deepEqual(workspace.panels["reward-analysis"].config.blocks, [
    { kind: "reward-breakdown", scope: "step" },
  ]);
  assert.deepEqual(workspace.panels["reward-analysis"].placement, {
    x: 0,
    y: 37,
    w: 12,
    h: 15,
    visible: false,
    window: "main",
  });
  assert.equal(workspace.panels.cnn.type, "cnn");
  assert.equal(workspace.panels.cnn.placement.visible, false);
  assert.equal(workspace.panels.cnn.enabled, false);
  assert.equal(workspace.panels.attribution.type, "attribution");
  assert.equal(workspace.panels.attribution.placement.visible, false);
  assert.equal(workspace.panels.attribution.enabled, false);
  assert.ok(
    Object.entries(workspace.panels)
      .filter(([id]) => !["attribution", "cnn"].includes(id))
      .every(([, panel]) => panel.enabled === true),
  );
});

test("task presets switch panel emphasis without deleting custom panels", () => {
  const workspace = createDefaultWorkspace();
  workspace.panels[CUSTOM_ID] = createTelemetryInstance({ id: CUSTOM_ID, title: "Mine" });

  applyWorkspacePreset(workspace, "explain");
  assert.equal(workspace.preset, "explain");
  assert.equal(workspace.panels.policy.placement.visible, true);
  assert.equal(workspace.panels.observation.placement.visible, false);
  assert.equal(workspace.panels[CUSTOM_ID].placement.visible, false);

  applyWorkspacePreset(workspace, "debug");
  assert.equal(workspace.preset, "debug");
  assert.equal(workspace.panels.observation.placement.visible, true);
  assert.equal(workspace.panels.raw.placement.visible, true);
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

test("existing v7 workspaces receive reward analysis hidden on the shelf", () => {
  const workspace = createDefaultWorkspace();
  delete workspace.panels["reward-analysis"];

  const normalized = normalizeWorkspace(workspace);

  assert.equal(normalized.panels["reward-analysis"].placement.visible, false);
  assert.equal(normalized.panels["reward-analysis"].builtin, true);
});

test("existing workspaces receive the CNN explorer hidden while new paired workspaces show it", () => {
  const workspace = createDefaultWorkspace({ paired: true });
  assert.equal(workspace.panels.cnn.placement.visible, true);
  assert.equal(workspace.panels.cnn.enabled, false);
  delete workspace.panels.cnn;

  const normalized = normalizeWorkspace(workspace, { paired: true });

  assert.equal(normalized.panels.cnn.placement.visible, false);
  assert.equal(normalized.panels.cnn.builtin, true);
});

test("existing workspaces receive attribution hidden while new paired workspaces show it", () => {
  const workspace = createDefaultWorkspace({ paired: true });
  assert.equal(workspace.panels.attribution.placement.visible, true);
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
