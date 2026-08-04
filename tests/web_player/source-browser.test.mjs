import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  activeRunMetricColumns,
  availableRunMetricColumns,
  bestRunEfficiency,
  checkpointCanEvaluate,
  checkpointMetricIsBest,
  checkpointPlaybackSeed,
  formatGoalDiffValue,
  formatMetricValue,
  goalConfigurationPresentation,
  metricLabel,
  rankRunItems,
  runFinishPresentation,
  runStatePresentation,
  SourceBrowser,
  successBadgeLabels,
  sortRunItems,
  sourceBreadcrumbItems,
  sourceRouteFromPath,
  sourceRoutePath,
} from "../../src/gradlab/web_player/sources/browser.js";

const METRIC = "eval/full/episode/return/shaped/mean";

test("checkpoint selection boxes are centered within their rows", async () => {
  const styles = await readFile(
    new URL("../../src/gradlab/web_player/styles.css", import.meta.url),
    "utf8",
  );

  assert.match(
    styles,
    /\.source-table \.source-selection-cell \{[^}]*vertical-align: middle;/,
  );
  assert.match(
    styles,
    /\.source-selection-cell input \{[^}]*display: block;[^}]*margin: 0 auto;/,
  );
});

test("checkpoint table uses compact API metric labels without a verdict column", async () => {
  const source = await readFile(
    new URL("../../src/gradlab/web_player/sources/browser.js", import.meta.url),
    "utf8",
  );

  assert.match(source, /label: column\.label \|\| metricLabel\(column\.metric\)/);
  assert.doesNotMatch(source, /\{ label: "Evaluation" \}/);
});

test("catalog list hover highlights the complete row", async () => {
  const styles = await readFile(
    new URL("../../src/gradlab/web_player/styles.css", import.meta.url),
    "utf8",
  );

  assert.match(
    styles,
    /\.environment-row:hover:not\(:disabled\),\s*\.goal-row:hover,\s*\.goal-row:focus-within\s*\{[^}]*background: #1b3b48;/,
  );
  assert.match(
    styles,
    /\.goal-row-navigation:hover:not\(:disabled\)\s*\{[^}]*background: transparent;/,
  );
});

test("scientific success badges are ordered, independent, and evidence-labelled", async () => {
  assert.deepEqual(successBadgeLabels({
    success_badges: ["eval/success", "train/success", "unknown"],
  }), ["train/success", "eval/success"]);
  assert.deepEqual(successBadgeLabels({ success_badges: ["train/success"] }), [
    "train/success",
  ]);
  assert.deepEqual(successBadgeLabels({}), []);

  const source = await readFile(
    new URL("../../src/gradlab/web_player/sources/browser.js", import.meta.url),
    "utf8",
  );
  const styles = await readFile(
    new URL("../../src/gradlab/web_player/styles.css", import.meta.url),
    "utf8",
  );
  assert.match(source, /renderSuccessBadges\(environment\)/);
  assert.match(source, /renderSuccessBadges\(goal\)/);
  assert.match(source, /renderSuccessBadges\(variant\)/);
  assert.match(source, /renderSuccessBadges\(run\)/);
  assert.match(source, /renderSuccessBadges\(item\)/);
  assert.match(styles, /\.success-badge\.evaluation/);
});

test("goal configurations expose exact diff counts and date columns", () => {
  const now = Date.parse("2026-08-01T12:00:00Z");
  assert.deepEqual(goalConfigurationPresentation({
    configuration_kind: "current_default",
    comparison_available: true,
    current_diff_count: 0,
    current_diff_count_exact: true,
    run_count: 0,
  }, now), {
    kind: "current_default",
    kindLabel: "Current default",
    sourceLabel: "Current goal",
    behaviorLabel: "Default",
    differenceCount: 0,
    differenceCountExact: true,
    differenceLabel: "0 changes",
    comparisonAvailable: true,
    runCount: 0,
    runLabel: "0 runs",
    firstUsedDate: "—",
    lastActivityDate: "—",
  });
  assert.deepEqual(goalConfigurationPresentation({
    configuration_kind: "previous_default",
    comparison_available: true,
    current_diff_count: 1,
    current_diff_count_exact: true,
    run_count: 2,
    first_used_at: "2026-08-01T10:00:00Z",
    last_activity_at: "2026-08-01T11:00:00Z",
  }, now), {
    kind: "previous_default",
    kindLabel: "Previous default",
    sourceLabel: "Older goal",
    behaviorLabel: "Default",
    differenceCount: 1,
    differenceCountExact: true,
    differenceLabel: "1 change",
    comparisonAvailable: true,
    runCount: 2,
    runLabel: "2 runs",
    firstUsedDate: "2 hours ago",
    lastActivityDate: "1 hour ago",
  });
  const older = goalConfigurationPresentation({
    run_count: 1,
    first_used_at: "2026-07-29T10:00:00Z",
    last_activity_at: "2026-07-30T11:00:00Z",
  }, now);
  assert.equal(older.firstUsedDate, "29 Jul 2026");
  assert.equal(older.lastActivityDate, "30 Jul 2026");
  assert.equal(older.differenceLabel, "Exact diff unavailable");
});

test("goal diff values preserve JSON types", () => {
  assert.equal(formatGoalDiffValue(false), "false");
  assert.equal(formatGoalDiffValue("discrete"), '"discrete"');
  assert.equal(formatGoalDiffValue({ threshold: 10 }), '{"threshold":10}');
  assert.equal(formatGoalDiffValue(null), "null");
  assert.equal(formatGoalDiffValue(null, { unavailable: true }), "—");
});

test("goal configuration metadata and exact changes render as tables", async () => {
  const source = await readFile(
    new URL("../../src/gradlab/web_player/sources/browser.js", import.meta.url),
    "utf8",
  );
  const styles = await readFile(
    new URL("../../src/gradlab/web_player/styles.css", import.meta.url),
    "utf8",
  );

  assert.match(
    source,
    /\["Configuration", "Differences", "Runs", "First used", "Last activity"\]/,
  );
  assert.match(
    styles,
    /\.goal-configuration-table \{ min-width: 64rem; table-layout: fixed; \}/,
  );
  assert.match(
    source,
    /\["configuration", "differences", "runs", "first-used", "last-activity"\]/,
  );
  assert.match(styles, /\.goal-configuration-column\.differences \{ width: 18rem; \}/);
  assert.match(styles, /\.goal-configuration-column\.runs \{ width: 6rem; \}/);
  assert.match(
    styles,
    /\.goal-configuration-column\.first-used,\s*\.goal-configuration-column\.last-activity \{ width: 11rem; \}/,
  );
  assert.match(
    styles,
    /\.goal-configuration-table-scroll,\s*\.goal-configuration-diff-scroll \{ min-width: 0; overflow: auto; \}/,
  );
  assert.match(source, /\["Operation", "Exact contract path", "Before", "After"\]/);
  assert.match(
    source,
    /goal-configuration-value goal-configuration-after \$\{kind\}/,
  );
  assert.match(
    styles,
    /\.goal-configuration-operation\.added,\s*\.goal-configuration-after\.added \{ color: #78d89f; \}/,
  );
  assert.match(
    styles,
    /\.goal-configuration-operation\.removed,\s*\.goal-configuration-after\.removed \{ color: #ff8990; \}/,
  );
  assert.match(
    styles,
    /\.goal-configuration-operation\.changed,\s*\.goal-configuration-after\.changed \{ color: #69d9ea; \}/,
  );
});

test("goal activity unifies variants with recent and best runs", async () => {
  const source = await readFile(
    new URL("../../src/gradlab/web_player/sources/browser.js", import.meta.url),
    "utf8",
  );

  assert.match(source, /\/goals\/\$\{encodeURIComponent\(this\.route\.goal_id\)\}\/activity/);
  assert.match(source, /headers\["If-None-Match"\] = `"\$\{this\.activityRevision\}"`/);
  assert.match(source, /this\.activityHasActiveRuns = Boolean\(payload\.has_active_runs\)/);
  assert.match(
    source,
    /this\.route\.level === "goal_variants"\s*&& this\.activityHasActiveRuns/,
  );
  assert.match(source, /heading\.textContent = "Runs using this configuration"/);
  assert.match(source, /\["recent", "Recent"\]/);
  assert.match(source, /\["best", "Best"\]/);
  assert.match(source, /page\?\.nextCursor \? "Load more" : "Load older runs"/);
});

test("run checkpoint pages refresh only on explicit request", (context) => {
  const originalLocation = globalThis.location;
  const originalWindow = globalThis.window;
  let intervalStarts = 0;
  globalThis.location = { pathname: "/embedded-player", search: "", hash: "" };
  globalThis.window = {
    setInterval() {
      intervalStarts += 1;
      return 73;
    },
  };
  context.after(() => {
    if (originalLocation === undefined) delete globalThis.location;
    else globalThis.location = originalLocation;
    if (originalWindow === undefined) delete globalThis.window;
    else globalThis.window = originalWindow;
  });

  const browser = new SourceBrowser(
    {},
    { replaceChildren() {}, hidden: false },
    {
      token: "token",
      command() {},
      getState: () => ({ hasControl: true }),
      showToast() {},
    },
  );
  browser.app = { phase: "selecting" };
  browser.route = { level: "runs", run_id: "gradlab-run" };
  browser.activityHasActiveRuns = true;

  browser.updatePolling();

  assert.equal(intervalStarts, 0);
  assert.equal(browser.pollTimer, null);

  browser.route = { level: "goal_variants" };
  browser.updatePolling();

  assert.equal(intervalStarts, 1);
  assert.equal(browser.pollTimer, 73);
});

test("run table hover highlights only the complete row", async () => {
  const styles = await readFile(
    new URL("../../src/gradlab/web_player/styles.css", import.meta.url),
    "utf8",
  );

  assert.match(
    styles,
    /\.source-table \.run-identity:hover:not\(:disabled\)\s*\{[^}]*background: transparent;[^}]*color: inherit;/,
  );
  assert.match(
    styles,
    /\.source-table tbody tr:hover,\s*\.source-table tbody tr:focus-visible\s*\{[^}]*background: rgba\(83, 212, 232, \.07\);/,
  );
});

test("run ranking badges render in the run column", async () => {
  const source = await readFile(
    new URL("../../src/gradlab/web_player/sources/browser.js", import.meta.url),
    "utf8",
  );

  assert.match(
    source,
    /className\.includes\("run-cell"\) && isEfficiencyLeader/,
  );
  assert.doesNotMatch(
    source,
    /className\.includes\("recipe-cell"\) && isEfficiencyLeader/,
  );
});

test("run result evidence is visually promoted above supporting metadata", async () => {
  const styles = await readFile(
    new URL("../../src/gradlab/web_player/styles.css", import.meta.url),
    "utf8",
  );

  assert.match(
    styles,
    /\.source-table \.finish-evidence \{[^}]*display: grid;/,
  );
  assert.match(
    styles,
    /\.source-table \.finish-evidence-value \{[^}]*font-size: \.96rem;/,
  );
});

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
    environment_id: "",
    goal_id: "",
    goal_variant_id: "",
    run_id: "",
    checkpoint_id: "",
  });
  assert.equal(sourceRoutePath({
    level: "environments",
    environment_id: "",
    goal_id: "",
    goal_variant_id: "",
    run_id: "",
    checkpoint_id: "",
  }), "/");
  assert.equal(sourceRouteFromPath("/projects/Mario"), null);
});

test("active checkpoint breadcrumbs retain the full source hierarchy", () => {
  const items = sourceBreadcrumbItems({
    level: "runs",
    environment_id: "ViZDoom",
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
      environment_id: "",
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
      environment_id: "ViZDoom",
      checkpoint_id: "checkpoint-a",
    },
    {
      level: "goal_variants",
      environment_id: "ViZDoom",
      goal_id: "DefendTheLine-v1",
      checkpoint_id: "checkpoint-b",
    },
    {
      level: "runs",
      environment_id: "ViZDoom",
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
  const historyModes = [];
  browser.syncUrl = (mode) => historyModes.push(mode);
  let renders = 0;
  browser.renderBreadcrumbs = () => {
    renders += 1;
  };
  const snapshot = {
    app: {
      phase: "active",
      route: {
        level: "runs",
        environment_id: "ViZDoom",
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
  assert.deepEqual(historyModes, ["replace"]);
});

test("active checkpoint breadcrumb navigation exits playback at the selected ancestor", () => {
  const browser = Object.create(SourceBrowser.prototype);
  browser.route = {
    level: "runs",
    environment_id: "ViZDoom",
    goal_id: "DefendTheLine-v1",
    goal_variant_id: "goal-variant-a27a8239",
    run_id: "gradlab-c22f7c7a",
    checkpoint_id: "checkpoint-10002432-b285ff3b",
  };
  const commands = [];
  const historyModes = [];
  const openedRoutes = [];
  browser.command = (name, payload) => {
    commands.push({ name, payload });
    return "command-id";
  };
  browser.applyRoute = (route) => {
    browser.route = { ...browser.route, ...route };
  };
  browser.syncUrl = (mode) => historyModes.push(mode);
  browser.openSourceRoute = (route) => openedRoutes.push(route);
  const target = sourceBreadcrumbItems(browser.route)[2].route;

  assert.equal(browser.navigate(target), true);

  const expected = {
    level: "goal_variants",
    environment_id: "ViZDoom",
    goal_id: "DefendTheLine-v1",
    goal_variant_id: "",
    run_id: "",
    checkpoint_id: "",
  };
  assert.deepEqual(commands, [{
    name: "browse_sources",
    payload: { route: expected },
  }]);
  assert.deepEqual(browser.route, expected);
  assert.deepEqual(historyModes, ["push"]);
  assert.deepEqual(openedRoutes, [expected]);
});

test("rejected active breadcrumb navigation leaves playback and history unchanged", () => {
  const browser = Object.create(SourceBrowser.prototype);
  browser.route = {
    level: "runs",
    environment_id: "ViZDoom",
    goal_id: "DefendTheLine-v1",
    goal_variant_id: "goal-variant-a27a8239",
    run_id: "gradlab-c22f7c7a",
    checkpoint_id: "checkpoint-10002432-b285ff3b",
  };
  const initialRoute = { ...browser.route };
  browser.command = () => null;
  browser.applyRoute = () => {
    throw new Error("a rejected navigation must not mutate the active route");
  };
  browser.syncUrl = () => {
    throw new Error("a rejected navigation must not mutate browser history");
  };
  browser.openSourceRoute = () => {
    throw new Error("a rejected navigation must not exit playback");
  };

  assert.equal(
    browser.navigate(sourceBreadcrumbItems(browser.route)[0].route),
    false,
  );
  assert.deepEqual(browser.route, initialRoute);
});

test("run metrics use compact labels and values", () => {
  assert.equal(metricLabel("leader/checkpoint/step"), "Checkpoint step");
  assert.equal(metricLabel(METRIC), "Mean return");
  assert.equal(
    metricLabel("train/outcome/success/across_starts/window_100/rate/min"),
    "Min success (last 100)",
  );
  assert.equal(
    metricLabel("train/outcome/success/across_starts/window_100/rate/mean"),
    "Mean success (last 100)",
  );
  assert.equal(
    metricLabel("train/episode/return/shaped/from/target/rolling_up_to_100/mean"),
    "Mean target-start return (up to 100)",
  );
  assert.equal(
    formatMetricValue("eval/full/outcome/success/across_starts/rate/min", 0.875),
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
      stop_reason: "early_stop_failure:loss_limit",
      final_step: 250_000,
      early_stop: { trigger: "threshold" },
    }),
    {
      label: "Training stop criterion met",
      detail: "Loss Limit · Stopped at 250,000 steps",
      tone: "failure",
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
      tone: "neutral",
    },
  );
  assert.deepEqual(
    runFinishPresentation({
      state: "stopped",
      stop_reason: "early_stop_neutral:return_plateau",
      final_step: 500_000,
      early_stop: { trigger: "no_improvement" },
    }),
    {
      label: "Training stalled",
      detail: "Return Plateau · Stopped at 500,000 steps",
      tone: "neutral",
    },
  );
  assert.deepEqual(
    runFinishPresentation({
      state: "succeeded",
      stop_reason: "early_stop_success:training_target",
      final_step: 16_384,
      early_stop: {
        condition_id: "training_target",
        trigger: "threshold",
        metric: "train/episode/return/shaped/from/target/window_100/mean",
        value: 5.25,
        condition: {
          metric: "train/episode/return/shaped/from/target/window_100/mean",
          trigger: "threshold",
          operator: ">=",
          threshold: 5,
        },
      },
    }),
    {
      label: "Training target met",
      detail: (
        "Mean target-start return (last 100) ≥ 5"
        + " · observed 5.25 · Stopped at 16,384 steps"
      ),
      evidence: {
        metric: "Mean target-start return (last 100)",
        observed: "5.25",
        required: "≥ 5",
        step: "Stopped at 16,384 steps",
      },
      tone: "success",
    },
  );
  assert.equal(
    runFinishPresentation({
      state: "succeeded",
      stop_reason: "eval_acceptance",
    }).label,
    "Evaluation criteria met",
  );
});

test("stalled training uses a neutral stop icon instead of a failure cross", () => {
  assert.deepEqual(
    runStatePresentation({
      state: "stopped",
      stop_reason: "early_stop_neutral:return_plateau",
      early_stop: { trigger: "no_improvement" },
    }),
    {
      iconName: "player-pause",
      tone: "stopped",
      label: "Training stalled",
    },
  );
  assert.deepEqual(
    runStatePresentation({
      state: "failed",
      stop_reason: "early_stop_failure:return_plateau",
      early_stop: { trigger: "no_improvement" },
    }),
    {
      iconName: "player-pause",
      tone: "stopped",
      label: "Training stalled",
    },
  );
  assert.deepEqual(
    runStatePresentation({
      state: "failed",
      stop_reason: "learner_failure",
    }),
    {
      iconName: "x",
      tone: "failed",
      label: "Failed",
    },
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

test("unevaluated and unsuccessfully evaluated checkpoints are selectable", () => {
  assert.equal(checkpointCanEvaluate({ evaluation: null }), true);
  assert.equal(
    checkpointCanEvaluate({ evaluation: { status: "rejected", pass: false } }),
    true,
  );
  assert.equal(
    checkpointCanEvaluate({ evaluation: { status: "failed", pass: false } }),
    true,
  );
  assert.equal(
    checkpointCanEvaluate({ evaluation: { status: "accepted", pass: true } }),
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

test("checkpoint metric cells identify their own leaders", async () => {
  const trainSuccess = "train/outcome/success/across_starts/window_100/rate/mean";
  const evalReturn = "eval/full/episode/return/shaped/mean";
  const checkpoint = { best_metrics: [trainSuccess, evalReturn] };

  assert.equal(checkpointMetricIsBest(checkpoint, trainSuccess), true);
  assert.equal(checkpointMetricIsBest(checkpoint, evalReturn), true);
  assert.equal(
    checkpointMetricIsBest(checkpoint, "eval/full/outcome/success/across_starts/rate/mean"),
    false,
  );
  assert.equal(checkpointMetricIsBest({}, trainSuccess), false);

  const source = await readFile(
    new URL("../../src/gradlab/web_player/sources/browser.js", import.meta.url),
    "utf8",
  );
  assert.match(source, /"checkpoint-metric-cell"/);
  assert.match(source, /badge\.className = "checkpoint-best-badge"/);
  assert.match(source, /badge\.textContent = "Best"/);
  assert.doesNotMatch(source, /Best training|Best eval/);
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
  browser.selectionFence = "f".repeat(64);
  browser.selectedCheckpoints = new Set(["checkpoint-a", "checkpoint-b"]);

  await browser.evaluateSelected();

  assert.equal(requests[0].url, "/api/catalog/runs/gradlab-run/evaluations");
  assert.equal(requests[0].options.method, "POST");
  assert.deepEqual(JSON.parse(requests[0].options.body), {
    checkpoint_ids: ["checkpoint-a", "checkpoint-b"],
    selection_fence: "f".repeat(64),
  });
  assert.equal(browser.items[0].evaluation_queue.state, "submitted");
  assert.equal(browser.selectedCheckpoints.size, 0);
  assert.match(toasts[0][0], /2 checkpoints queued/);
});

test("browser back through checkpoint routes preserves canonical environment identity", (context) => {
  const originalLocation = globalThis.location;
  const originalWindow = globalThis.window;
  let popstate;
  globalThis.location = {
    pathname: (
      "/environments/ViZDoom/goals/DefendTheLine-v1"
      + "/variants/goal-variant-a27a8239/runs/gradlab-c22f7c7a"
      + "/checkpoints/checkpoint-10002432-b285ff3b"
    ),
    search: "",
    hash: "",
  };
  globalThis.window = {
    addEventListener(name, listener) {
      if (name === "popstate") popstate = listener;
    },
  };
  context.after(() => {
    if (originalLocation === undefined) delete globalThis.location;
    else globalThis.location = originalLocation;
    if (originalWindow === undefined) delete globalThis.window;
    else globalThis.window = originalWindow;
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
  sourceBrowser.route = {
    level: "runs",
    environment_id: "ViZDoom",
    goal_id: "DefendTheLine-v1",
    goal_variant_id: "goal-variant-a27a8239",
    run_id: "gradlab-c22f7c7a",
    checkpoint_id: "checkpoint-10002432-b285ff3b",
  };
  sourceBrowser.app = {
    phase: "selecting",
    route: { ...sourceBrowser.route },
  };
  sourceBrowser.applyRoute = (route) => {
    sourceBrowser.route = route;
  };

  globalThis.location.pathname = (
    "/environments/ViZDoom/goals/DefendTheLine-v1"
    + "/variants/goal-variant-a27a8239/runs/gradlab-c22f7c7a"
  );
  popstate();

  assert.equal(sourceBrowser.route.environment_id, "ViZDoom");
  assert.equal(sourceBrowser.endpoint(), (
    "/api/catalog/runs/gradlab-c22f7c7a/checkpoints?"
    + "goal_variant_id=goal-variant-a27a8239"
  ));
  assert.equal(commands[0].name, "browse_sources");
  assert.equal(commands[0].payload.route.environment_id, "ViZDoom");

  globalThis.location.pathname = (
    "/environments/ViZDoom/goals/DefendTheLine-v1"
    + "/variants/goal-variant-a27a8239"
  );
  popstate();

  assert.equal(sourceBrowser.route.environment_id, "ViZDoom");
  assert.equal(sourceBrowser.endpoint(), (
    "/api/catalog/environments/ViZDoom/goals/DefendTheLine-v1"
    + "/variants/goal-variant-a27a8239/runs?"
  ));
  assert.equal(commands[1].name, "browse_sources");
  assert.equal(commands[1].payload.route.environment_id, "ViZDoom");
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
  const environmentsRequest = sourceBrowser.load();
  assert.equal(requests.length, 1);
  assert.match(requests[0].url, /^\/api\/catalog\/environments/);

  sourceBrowser.applyRoute({
    level: "goals",
    environment_id: "Mario",
    goal_id: "",
    goal_variant_id: "",
    run_id: "",
    checkpoint_id: "",
  });
  assert.equal(requests.length, 2);
  assert.equal(requests[0].options.signal.aborted, true);
  assert.match(requests[1].url, /\/environments\/Mario\/goals/);
  requests[0].resolve({
    ok: true,
    json: async () => ({
      items: [{ name: "Mario", goal_count: 10 }],
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
        environment_id: "Mario",
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
    environment_id: "Mario",
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

test("run panels omit metric columns with no visible evidence", () => {
  const columns = [
    { metric: "missing", direction: "max" },
    { metric: "available", direction: "max" },
  ];
  const items = [
    { metrics: { missing: null, available: 7.63 } },
    { metrics: { missing: undefined, available: null } },
  ];

  assert.deepEqual(availableRunMetricColumns(items, columns), [columns[1]]);
});

test("run efficiency prefers complete goal evaluation and follows its rank order", () => {
  const primary = [
    { metric: "leader/checkpoint/step", direction: "min" },
    { metric: METRIC, direction: "max" },
  ];
  const fallback = [
    {
      metric: "train/outcome/success/across_starts/window_100/rate/min",
      direction: "max",
    },
    { metric: "train/global_step", direction: "min" },
  ];
  const items = [
    {
      run_id: "training-only",
      recipe: "fast-training",
      metrics: {
        "train/outcome/success/across_starts/window_100/rate/min": 1,
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
      metric: "train/outcome/success/across_starts/window_100/rate/min",
      direction: "max",
    },
    { metric: "train/global_step", direction: "min" },
  ];
  const items = [
    {
      run_id: "slower",
      metrics: {
        "train/outcome/success/across_starts/window_100/rate/min": 0.9,
        "train/global_step": 2_000,
      },
    },
    {
      run_id: "faster",
      metrics: {
        "train/outcome/success/across_starts/window_100/rate/min": 0.9,
        "train/global_step": 1_000,
      },
    },
  ];

  assert.equal(activeRunMetricColumns(items, primary, fallback), fallback);
  const leader = bestRunEfficiency(items, primary, fallback);
  assert.equal(leader.evidence, "training");
  assert.equal(leader.item.run_id, "faster");
});
