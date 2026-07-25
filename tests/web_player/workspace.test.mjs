import assert from "node:assert/strict";
import test from "node:test";

import {
  WORKSPACE_VERSION,
  bumpWorkspaceRevision,
  compareWorkspaceRevisions,
  createDefaultWorkspace,
  createTelemetryInstance,
  normalizeWorkspace,
} from "../../src/rlab/web_player/panels/workspace.js";

const CUSTOM_ID = "panel-00000000-0000-4000-8000-000000000000";

test("default workspace is a v4 collection of typed panel instances", () => {
  const workspace = createDefaultWorkspace({ writer: "main" });
  assert.equal(workspace.version, WORKSPACE_VERSION);
  assert.equal(workspace.panels.reward.type, "telemetry");
  assert.equal(workspace.panels.reward.builtin, true);
  assert.deepEqual(
    workspace.panels.reward.config.blocks.map((block) => block.kind),
    ["stats"],
  );
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
    w: 7,
    h: 15,
    visible: true,
    window: "main",
  });
  assert.deepEqual(workspace.panels["step-reward"].placement, {
    x: 4,
    y: 15,
    w: 4,
    h: 9,
    visible: true,
    window: "main",
  });
});

test("paired workspace keeps independent graph panels in one stats row", () => {
  const workspace = createDefaultWorkspace({ paired: true });
  assert.deepEqual(
    ["value", "step-reward", "episode-return"].map((id) => (
      workspace.panels[id].placement
    )),
    [
      { x: 0, y: 8, w: 4, h: 9, visible: true, window: "stats" },
      { x: 4, y: 8, w: 4, h: 9, visible: true, window: "stats" },
      { x: 8, y: 8, w: 4, h: 9, visible: true, window: "stats" },
    ],
  );
});

test("legacy workspace data is deliberately replaced instead of migrated", () => {
  const workspace = normalizeWorkspace({
    version: 3,
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
