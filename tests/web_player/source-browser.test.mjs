import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  activeRunMetricColumns,
  bestRunEfficiency,
  checkpointCanEvaluate,
  checkpointEvaluationCell,
  checkpointPlaybackSeed,
  formatMetricValue,
  metricLabel,
  rankRunItems,
  runFinishPresentation,
  SourceBrowser,
  sortRunItems,
  sourceBreadcrumbItems,
  sourceRouteFromPath,
  sourceRoutePath,
} from "../../src/gradlab/web_player/sources/browser.js";

const METRIC = "eval/full/episode/return/mean";

test("active-run status icon is available in the shared icon sprite", async () => {
  const sprite = await readFile(
    new URL("../../src/gradlab/web_player/tabler-icons.svg", import.meta.url),
    "utf8",
  );

  assert.match(sprite, /id="ti-activity-heartbeat"/);
});

test("playback home is the root environment route", () => {
  assert.deepEqual(sourceRouteFromPath("/"), {
    level: "environments",
    entity: "",
    project: "",
    goal_id: "",
    goal_variant_id: "",
    run_id: "",
    checkpoint_id: "",
  });
  assert.equal(sourceRoutePath({
    level: "environments",
    entity: "research",
    project: "",
    goal_id: "",
    goal_variant_id: "",
    run_id: "",
    checkpoint_id: "",
  }), "/");
});

test("active checkpoint breadcrumbs retain the full source hierarchy", () => {
  const items = sourceBreadcrumbItems({
    level: "runs",
    project: "ViZDoom",
    goal_id: "DefendTheLine-v1",
    goal_variant_id: "goal-variant-a27a8239",
    run_id: "gradlab-c22f7c7a",
    checkpoint_id: "checkpoint-10002432-b285ff3b",
  });

  assert.deepEqual(
    items.map(({ label, current }) => ({ label, current })),
    [
      { label: "Environments", current: false },
      { label: "ViZDoom", current: false },
      { label: "DefendTheLine-v1", current: false },
      { label: "goal-variant-a27a8239", current: false },
      { label: "gradlab-c22f7c7a", current: false },
      { label: "checkpoint-10002432-b285ff3b", current: true },
    ],
  );
  assert.deepEqual(items.at(-2).route, {
    level: "runs",
    checkpoint_id: "",
  });
  assert.equal(items.at(-1).route, null);
});

test("environment breadcrumb remains clickable for a partial active checkpoint route", () => {
  const items = sourceBreadcrumbItems({
    level: "environments",
    checkpoint_id: "checkpoint-10002432-b285ff3b",
  });

  assert.deepEqual(items[0], {
    label: "Environments",
    current: false,
    route: {
      level: "environments",
      project: "",
      goal_id: "",
      goal_variant_id: "",
      run_id: "",
      checkpoint_id: "",
    },
  });
  assert.deepEqual(items[1], {
    label: "checkpoint-10002432-b285ff3b",
    current: true,
    route: null,
  });
});

test("all active checkpoint ancestors remain clickable with stale route levels", () => {
  const routes = [
    {
      level: "goals",
      project: "ViZDoom",
      checkpoint_id: "checkpoint-a",
    },
    {
      level: "goal_variants",
      project: "ViZDoom",
      goal_id: "DefendTheLine-v1",
      checkpoint_id: "checkpoint-b",
    },
    {
      level: "runs",
      project: "ViZDoom",
      goal_id: "DefendTheLine-v1",
      goal_variant_id: "goal-variant-a27a8239",
      checkpoint_id: "checkpoint-c",
    },
  ];

  for (const route of routes) {
    const items = sourceBreadcrumbItems(route);
    assert.deepEqual(
      items.map(({ label, current }) => ({ label, current })),
      [
        ...items.slice(0, -1).map(({ label }) => ({ label, current: false })),
        { label: route.checkpoint_id, current: true },
      ],
    );
  }
});

test("active breadcrumb rendering hides routes without a selected checkpoint", () => {
  const browser = Object.create(SourceBrowser.prototype);
  browser.breadcrumbsRoot = {
    hidden: false,
    replaceChildrenCalled: 0,
    replaceChildren() {
      this.replaceChildrenCalled += 1;
    },
  };
  browser.stop = () => {};
  browser.renderBreadcrumbs = () => {
    throw new Error("breadcrumbs should not render without a checkpoint");
  };

  browser.renderActiveBreadcrumbs({
    app: {
      phase: "active",
      route: { level: "runs", run_id: "gradlab-c22f7c7a", checkpoint_id: "" },
    },
  });

  assert.equal(browser.breadcrumbsRoot.hidden, true);
  assert.equal(browser.breadcrumbsRoot.replaceChildrenCalled, 1);
});

test("unchanged active checkpoint routes do not rebuild breadcrumbs", () => {
  const browser = Object.create(SourceBrowser.prototype);
  browser.activeBreadcrumbRoute = "";
  browser.breadcrumbsRoot = {
    hidden: true,
    replaceChildren() {},
  };
  browser.stop = () => {};
  let renders = 0;
  browser.renderBreadcrumbs = () => {
    renders += 1;
  };
  const snapshot = {
    app: {
      phase: "active",
      route: {
        level: "runs",
        project: "ViZDoom",
        goal_id: "DefendTheLine-v1",
        goal_variant_id: "goal-variant-a27a8239",
        run_id: "gradlab-c22f7c7a",
        checkpoint_id: "checkpoint-10002432-b285ff3b",
      },
    },
  };

  browser.renderActiveBreadcrumbs(snapshot);
  browser.renderActiveBreadcrumbs(snapshot);

  assert.equal(renders, 1);
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

test("run finish reasons distinguish resource, training, and evaluation outcomes", () => {
  assert.deepEqual(
    runFinishPresentation({
      state: "succeeded",
      stop_reason: "training_cap_complete",
      final_step: 2_000_000,
    }),
    {
      label: "Maximum timesteps reached",
      detail: "Stopped at 2,000,000 steps",
      tone: "neutral",
    },
  );
  assert.deepEqual(
    runFinishPresentation({
      state: "failed",
      stop_reason: "early_stop_failure:return_plateau",
      final_step: 500_000,
      early_stop: { trigger: "no_improvement" },
    }),
    {
      label: "Training stalled",
      detail: "Return Plateau · Stopped at 500,000 steps",
      tone: "failure",
    },
  );
  assert.equal(
    runFinishPresentation({
      state: "succeeded",
      stop_reason: "early_stop_success:training_target",
    }).label,
    "Training success criterion met",
  );
  assert.equal(
    runFinishPresentation({
      state: "succeeded",
      stop_reason: "eval_acceptance",
    }).label,
    "Evaluation criteria met",
  );
});

test("finished runs without terminal evidence do not get a guessed reason", () => {
  assert.deepEqual(
    runFinishPresentation({ state: "finished" }),
    {
      label: "Reason unavailable",
      detail: "This run has no projected terminal receipt.",
      tone: "unknown",
    },
  );
  assert.equal(runFinishPresentation({ state: "running" }).label, "—");
});

test("checkpoint playback seed accepts catalog provenance and rejects invalid values", () => {
  assert.equal(checkpointPlaybackSeed({ playback_seed: 42_000 }), 42_000);
  assert.equal(checkpointPlaybackSeed({ playback_seed: 0 }), 0);
  assert.equal(checkpointPlaybackSeed({ playback_seed: null }), null);
  assert.equal(checkpointPlaybackSeed({ playback_seed: -1 }), null);
});

test("only checkpoints without terminal or queued evaluation state are selectable", () => {
  assert.equal(checkpointCanEvaluate({ evaluation: null }), true);
  assert.equal(
    checkpointCanEvaluate({ evaluation: { status: "rejected", pass: false } }),
    false,
  );
  assert.equal(
    checkpointCanEvaluate({ evaluation: null, evaluation_queue: { state: "submitted" } }),
    false,
  );
  assert.equal(
    checkpointCanEvaluate({ evaluation: null, evaluation_queue: { state: "expired" } }),
    false,
  );
  assert.equal(
    checkpointCanEvaluate({
      evaluation: null,
      evaluation_queue: { state: "waiting_for_training_terminal" },
    }),
    false,
  );
});

test("checkpoint evaluation cells distinguish queued work from terminal evidence", () => {
  assert.deepEqual(
    checkpointEvaluationCell({
      evaluation: null,
      evaluation_queue: { state: "submitted" },
    }),
    [
      "Running",
      "Submitted to the evaluation worker",
      "evaluation-cell submitted",
    ],
  );
  assert.deepEqual(
    checkpointEvaluationCell({
      evaluation: null,
      evaluation_queue: {
        state: "flusher_unavailable",
        message: "worker startup timed out",
      },
    }),
    [
      "Flusher unavailable",
      "worker startup timed out",
      "evaluation-cell flusher_unavailable",
    ],
  );
  assert.deepEqual(
    checkpointEvaluationCell({
      evaluation: null,
      evaluation_queue: {
        state: "waiting_for_run_lease",
        message: "writer is still draining",
      },
    }),
    [
      "Waiting for writer",
      "writer is still draining",
      "evaluation-cell waiting_for_run_lease",
    ],
  );
  assert.equal(
    checkpointEvaluationCell({
      evaluation: {
        status: "rejected",
        pass: false,
        episodes_completed: 1,
        episodes_planned: 100,
        failure_count: 1,
        criteria: [],
      },
    })[0],
    "Failed",
  );
});

test("selected checkpoints are admitted together through the evaluation API", async (context) => {
  const originalLocation = globalThis.location;
  const originalFetch = globalThis.fetch;
  const requests = [];
  const toasts = [];
  globalThis.location = { pathname: "/embedded-player", search: "", hash: "" };
  globalThis.fetch = async (url, options) => {
    requests.push({ url, options });
    return {
      ok: true,
      status: 202,
      json: async () => ({
        worker: { state: "started", pid: 123, message: null },
        items: [
          {
            checkpoint_id: "checkpoint-a",
            state: "submitted",
            evaluation: null,
            message: null,
          },
          {
            checkpoint_id: "checkpoint-b",
            state: "submitted",
            evaluation: null,
            message: null,
          },
        ],
      }),
    };
  };
  context.after(() => {
    if (originalLocation === undefined) delete globalThis.location;
    else globalThis.location = originalLocation;
    if (originalFetch === undefined) delete globalThis.fetch;
    else globalThis.fetch = originalFetch;
  });
  const browser = new SourceBrowser(
    {},
    { replaceChildren() {}, hidden: false },
    {
      token: "token",
      command() {},
      getState: () => ({ hasControl: true }),
      showToast: (...args) => toasts.push(args),
    },
  );
  browser.renderView = () => {};
  browser.route = { level: "runs", run_id: "gradlab-run" };
  browser.items = [
    { checkpoint_id: "checkpoint-a", evaluation: null },
    { checkpoint_id: "checkpoint-b", evaluation: null },
  ];
  browser.selectedCheckpoints = new Set(["checkpoint-a", "checkpoint-b"]);

  await browser.evaluateSelected();

  assert.equal(requests[0].url, "/api/catalog/runs/gradlab-run/evaluations");
  assert.equal(requests[0].options.method, "POST");
  assert.deepEqual(JSON.parse(requests[0].options.body), {
    checkpoint_ids: ["checkpoint-a", "checkpoint-b"],
  });
  assert.equal(browser.items[0].evaluation_queue.state, "submitted");
  assert.equal(browser.selectedCheckpoints.size, 0);
  assert.match(toasts[0][0], /2 checkpoints queued/);
});

test("legacy project routes inherit the entity and use canonical environment APIs", async (context) => {
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
      route: { level: "environments" },
      catalog: { entity: "research", items: [] },
    },
  });
  await new Promise((resolve) => setImmediate(resolve));

  assert.equal(
    requests[0],
    "/api/catalog/environments/research/Mario/goals?",
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

  const environmentsRequest = sourceBrowser.load();
  assert.equal(requests.length, 1);
  assert.match(requests[0].url, /^\/api\/catalog\/environments/);

  sourceBrowser.applyRoute({
    level: "goals",
    project: "Mario",
    goal_id: "",
    goal_variant_id: "",
    run_id: "",
    checkpoint_id: "",
  });
  assert.equal(requests.length, 2);
  assert.equal(requests[0].options.signal.aborted, true);
  assert.match(requests[1].url, /\/environments\/research\/Mario\/goals/);
  requests[0].resolve({
    ok: true,
    json: async () => ({
      items: [{ entity: "research", name: "Mario", goal_count: 10 }],
      next_cursor: null,
    }),
  });
  await environmentsRequest;

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

test("goal variants are a first-class canonical route between goals and runs", () => {
  const variant = "goal-variant-0123456789abcdef01234567";
  const run = `gradlab-${"a".repeat(32)}`;
  const checkpoint = `checkpoint-12-${"b".repeat(16)}`;
  const path = (
    `/environments/Mario/goals/Level1-1/variants/${variant}`
    + `/runs/${run}/checkpoints/${checkpoint}`
  );
  assert.deepEqual(sourceRouteFromPath(path), {
    level: "runs",
    entity: "",
    project: "Mario",
    goal_id: "Level1-1",
    goal_variant_id: variant,
    run_id: run,
    checkpoint_id: checkpoint,
  });
  assert.equal(sourceRoutePath(sourceRouteFromPath(path)), path);
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
