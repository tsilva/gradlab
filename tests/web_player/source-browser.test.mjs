import assert from "node:assert/strict";
import test from "node:test";

import {
  activeRunMetricColumns,
  bestRunEfficiency,
  checkpointPlaybackSeed,
  formatMetricValue,
  metricLabel,
  rankRunItems,
  SourceBrowser,
  sortRunItems,
  sourceRouteFromPath,
  sourceRoutePath,
} from "../../src/rlab/web_player/sources/browser.js";

const METRIC = "eval/full/episode/return/mean";

test("playback home is the root project route", () => {
  assert.deepEqual(sourceRouteFromPath("/"), {
    level: "projects",
    entity: "",
    project: "",
    goal_id: "",
    run_id: "",
    checkpoint_id: "",
  });
  assert.equal(sourceRoutePath({
    level: "projects",
    entity: "research",
    project: "",
    goal_id: "",
    run_id: "",
    checkpoint_id: "",
  }), "/");
});

test("run metrics use compact labels and values", () => {
  assert.equal(metricLabel("leader/checkpoint/step"), "Checkpoint step");
  assert.equal(metricLabel(METRIC), "Mean return");
  assert.equal(
    formatMetricValue("eval/full/outcome/success/rate/min", 0.875),
    "87.5%",
  );
  assert.equal(formatMetricValue(METRIC, null), "—");
});

test("checkpoint playback seed accepts catalog provenance and rejects invalid values", () => {
  assert.equal(checkpointPlaybackSeed({ playback_seed: 42_000 }), 42_000);
  assert.equal(checkpointPlaybackSeed({ playback_seed: 0 }), 0);
  assert.equal(checkpointPlaybackSeed({ playback_seed: null }), null);
  assert.equal(checkpointPlaybackSeed({ playback_seed: -1 }), null);
});

test("direct project routes inherit the server catalog entity", async (context) => {
  const originalLocation = globalThis.location;
  const originalWindow = globalThis.window;
  const originalFetch = globalThis.fetch;
  const requests = [];
  globalThis.location = {
    pathname: "/projects/Mario",
    search: "",
    hash: "",
  };
  globalThis.window = { addEventListener() {} };
  globalThis.fetch = async (url) => {
    requests.push(url);
    return {
      ok: true,
      json: async () => ({ items: [], next_cursor: null }),
    };
  };
  context.after(() => {
    if (originalLocation === undefined) delete globalThis.location;
    else globalThis.location = originalLocation;
    if (originalWindow === undefined) delete globalThis.window;
    else globalThis.window = originalWindow;
    if (originalFetch === undefined) delete globalThis.fetch;
    else globalThis.fetch = originalFetch;
  });

  const commands = [];
  const sourceBrowser = new SourceBrowser(
    {},
    { replaceChildren() {}, hidden: false },
    {
      token: "token",
      command: (name, payload) => commands.push({ name, payload }),
      getState: () => ({ hasControl: true }),
      showToast() {},
    },
  );
  sourceBrowser.renderView = () => {};
  sourceBrowser.updatePolling = () => {};

  sourceBrowser.render({
    app: {
      phase: "selecting",
      route: { level: "projects" },
      catalog: { entity: "research", items: [] },
    },
  });
  await new Promise((resolve) => setImmediate(resolve));

  assert.equal(
    requests[0],
    "/api/catalog/projects/research/Mario/goals?",
  );
  assert.equal(commands[0].name, "browse_sources");
  assert.equal(commands[0].payload.route.entity, "research");
});

test("late catalog responses cannot populate a newer route", async (context) => {
  const originalLocation = globalThis.location;
  const originalFetch = globalThis.fetch;
  const requests = [];
  globalThis.location = { pathname: "/embedded-player", search: "", hash: "" };
  globalThis.fetch = (url, options) => new Promise((resolve, reject) => {
    const request = { url, options, resolve, reject };
    requests.push(request);
    options.signal.addEventListener("abort", () => reject(new Error("aborted")));
  });
  context.after(() => {
    if (originalLocation === undefined) delete globalThis.location;
    else globalThis.location = originalLocation;
    if (originalFetch === undefined) delete globalThis.fetch;
    else globalThis.fetch = originalFetch;
  });

  const sourceBrowser = new SourceBrowser(
    {},
    { replaceChildren() {}, hidden: false },
    {
      token: "token",
      command() {},
      getState: () => ({ hasControl: true }),
      showToast() {},
    },
  );
  sourceBrowser.renderView = () => {};
  sourceBrowser.updatePolling = () => {};
  sourceBrowser.route.entity = "research";

  const projectsRequest = sourceBrowser.load();
  assert.equal(requests.length, 1);
  assert.match(requests[0].url, /^\/api\/catalog\/projects/);

  sourceBrowser.applyRoute({
    level: "goals",
    project: "Mario",
    goal_id: "",
    run_id: "",
    checkpoint_id: "",
  });
  assert.equal(requests.length, 2);
  assert.equal(requests[0].options.signal.aborted, true);
  assert.match(requests[1].url, /\/projects\/research\/Mario\/goals/);
  requests[0].resolve({
    ok: true,
    json: async () => ({
      items: [{ entity: "research", name: "Mario", goal_count: 10 }],
      next_cursor: null,
    }),
  });
  await projectsRequest;

  assert.equal(requests.length, 2);
  assert.deepEqual(sourceBrowser.items, []);

  requests[1].resolve({
    ok: true,
    json: async () => ({
      items: [{
        entity: "research",
        project: "Mario",
        goal_id: "Level1-1",
        goal_slug: "Mario/Level1-1",
        title: "Mario Level 1-1 completion",
        recipe_count: 1,
      }],
      next_cursor: null,
    }),
  });
  await new Promise((resolve) => setImmediate(resolve));

  assert.equal(sourceBrowser.items[0].goal_id, "Level1-1");
  assert.equal(sourceBrowser.items[0].recipe_count, 1);
});

test("stalled catalog requests time out with a recoverable error", async (context) => {
  const originalLocation = globalThis.location;
  const originalFetch = globalThis.fetch;
  globalThis.location = { pathname: "/embedded-player", search: "", hash: "" };
  globalThis.fetch = (_url, options) => new Promise((_resolve, reject) => {
    options.signal.addEventListener("abort", () => reject(new Error("aborted")));
  });
  context.after(() => {
    if (originalLocation === undefined) delete globalThis.location;
    else globalThis.location = originalLocation;
    if (originalFetch === undefined) delete globalThis.fetch;
    else globalThis.fetch = originalFetch;
  });

  const sourceBrowser = new SourceBrowser(
    {},
    { replaceChildren() {}, hidden: false },
    {
      token: "token",
      command() {},
      getState: () => ({ hasControl: true }),
      showToast() {},
      catalogRequestTimeoutMs: 1,
    },
  );
  sourceBrowser.renderView = () => {};

  await sourceBrowser.load();

  assert.equal(sourceBrowser.loading, false);
  assert.equal(sourceBrowser.error, "Catalog request timed out. Try Refresh.");
});

test("run metric sorting respects direction and keeps missing values last", () => {
  const items = [
    { run_id: "missing", metrics: { [METRIC]: null } },
    { run_id: "low", metrics: { [METRIC]: 10 } },
    { run_id: "high", metrics: { [METRIC]: 20 } },
  ];

  assert.deepEqual(
    sortRunItems(items, { metric: METRIC, direction: "descending" })
      .map((item) => item.run_id),
    ["high", "low", "missing"],
  );
  assert.deepEqual(
    sortRunItems(items, { metric: METRIC, direction: "ascending" })
      .map((item) => item.run_id),
    ["low", "high", "missing"],
  );
});

test("run efficiency prefers complete goal evaluation and follows its rank order", () => {
  const primary = [
    { metric: "leader/checkpoint/step", direction: "min" },
    { metric: METRIC, direction: "max" },
  ];
  const fallback = [
    {
      metric: "train/outcome/success/window_100/rate/min",
      direction: "max",
    },
    { metric: "train/global_step", direction: "min" },
  ];
  const items = [
    {
      run_id: "training-only",
      recipe: "fast-training",
      metrics: {
        "train/outcome/success/window_100/rate/min": 1,
        "train/global_step": 100,
      },
    },
    {
      run_id: "later-checkpoint",
      recipe: "high-return",
      metrics: {
        "leader/checkpoint/step": 2_000,
        [METRIC]: 500,
      },
    },
    {
      run_id: "earlier-checkpoint",
      recipe: "sample-efficient",
      metrics: {
        "leader/checkpoint/step": 1_000,
        [METRIC]: 100,
      },
    },
  ];

  assert.equal(activeRunMetricColumns(items, primary, fallback), primary);
  assert.deepEqual(
    rankRunItems(items, primary).map((item) => item.run_id),
    ["earlier-checkpoint", "later-checkpoint", "training-only"],
  );
  const leader = bestRunEfficiency(items, primary, fallback);
  assert.equal(leader.evidence, "evaluation");
  assert.equal(leader.item.recipe, "sample-efficient");
});

test("run efficiency labels training fallback without evaluation evidence", () => {
  const primary = [
    { metric: "leader/checkpoint/step", direction: "min" },
    { metric: METRIC, direction: "max" },
  ];
  const fallback = [
    {
      metric: "train/outcome/success/window_100/rate/min",
      direction: "max",
    },
    { metric: "train/global_step", direction: "min" },
  ];
  const items = [
    {
      run_id: "slower",
      metrics: {
        "train/outcome/success/window_100/rate/min": 0.9,
        "train/global_step": 2_000,
      },
    },
    {
      run_id: "faster",
      metrics: {
        "train/outcome/success/window_100/rate/min": 0.9,
        "train/global_step": 1_000,
      },
    },
  ];

  assert.equal(activeRunMetricColumns(items, primary, fallback), fallback);
  const leader = bestRunEfficiency(items, primary, fallback);
  assert.equal(leader.evidence, "training");
  assert.equal(leader.item.run_id, "faster");
});
