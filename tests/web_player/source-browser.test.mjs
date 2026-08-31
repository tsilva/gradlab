import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  activeRunMetricColumns,
  availableRunMetricColumns,
  bestRunEfficiency,
  catalogItemMatchesSearch,
  checkpointCanEvaluate,
  checkpointEvidencePresentation,
  checkpointMetricBestBadge,
  checkpointMetricDescription,
  checkpointMetricHeaderLabel,
  checkpointMetricIsBest,
  checkpointMetricRoleLabel,
  checkpointNavigationPresentation,
  checkpointPrefetchSources,
  checkpointPlaybackSeed,
  checkpointRecommendation,
  checkpointRecommendationMetrics,
  environmentEvidenceRank,
  environmentSuccessStatus,
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

test("checkpoint navigation reports chronological position and neighbors", () => {
  const checkpoints = [
    { checkpoint_id: "checkpoint-300-c", step: 300, sha256: "c" },
    { checkpoint_id: "checkpoint-100-a", step: 100, sha256: "a" },
    { checkpoint_id: "checkpoint-200-b", step: 200, sha256: "b" },
  ];

  assert.deepEqual(
    checkpointNavigationPresentation(checkpoints, "checkpoint-200-b"),
    {
      count: 3,
      position: 2,
      current: checkpoints[2],
      previous: checkpoints[1],
      next: checkpoints[0],
    },
  );
  assert.deepEqual(
    checkpointNavigationPresentation(checkpoints, "checkpoint-100-a"),
    {
      count: 3,
      position: 1,
      current: checkpoints[1],
      previous: null,
      next: checkpoints[2],
    },
  );
});

test("checkpoint navigation renders only the compact position ratio", async () => {
  const source = await readFile(
    new URL("../../src/gradlab/web_player/sources/browser.js", import.meta.url),
    "utf8",
  );

  assert.match(source, /\? "… \/ …"/);
  assert.match(
    source,
    /`\$\{presentation\.position\.toLocaleString\(\)\} \/ \$\{presentation\.count\.toLocaleString\(\)\}`/,
  );
  assert.doesNotMatch(source, /`Checkpoint \$\{presentation\.position/);
});

test("checkpoint navigation fails closed when the active checkpoint is absent", () => {
  assert.deepEqual(
    checkpointNavigationPresentation([
      { checkpoint_id: "checkpoint-100-a", step: 100, sha256: "a" },
    ], "checkpoint-missing"),
    {
      count: 1,
      position: null,
      current: null,
      previous: null,
      next: null,
    },
  );
});

test("checkpoint prefetch selects only the immediate disk-cache neighbors", () => {
  const checkpoints = [
    {
      checkpoint_id: "checkpoint-300-c",
      run_id: "gradlab-run",
      manifest_url: "https://models.example/checkpoint-300/manifest.json",
      step: 300,
    },
    {
      checkpoint_id: "checkpoint-100-a",
      run_id: "gradlab-run",
      manifest_url: "https://models.example/checkpoint-100/manifest.json",
      step: 100,
    },
    {
      checkpoint_id: "checkpoint-200-b",
      run_id: "gradlab-run",
      manifest_url: "https://models.example/checkpoint-200/manifest.json",
      step: 200,
    },
    {
      checkpoint_id: "checkpoint-400-d",
      run_id: "gradlab-run",
      manifest_url: "https://models.example/checkpoint-400/manifest.json",
      step: 400,
    },
  ];

  assert.deepEqual(
    checkpointPrefetchSources(checkpoints, "checkpoint-200-b"),
    [checkpoints[1], checkpoints[0]].map((item) => ({
      kind: "public_run",
      value: item.manifest_url,
      run_id: item.run_id,
      checkpoint_id: item.checkpoint_id,
    })),
  );
});

test("top checkpoint navigation starts one blocking load state", () => {
  const browser = Object.create(SourceBrowser.prototype);
  browser.route = {
    run_id: "gradlab-run",
    checkpoint_id: "checkpoint-200-b",
  };
  browser.activeCheckpointPendingId = "";
  browser.activeCheckpointItems = () => [
    { checkpoint_id: "checkpoint-100-a", step: 100 },
    { checkpoint_id: "checkpoint-200-b", step: 200 },
  ];
  browser.selectCheckpoint = () => "load-command";
  browser.renderActiveCheckpointNavigation = () => {};
  const loads = [];
  browser.beginCheckpointLoad = (load) => loads.push(load);

  assert.equal(browser.selectAdjacentCheckpoint("previous"), true);
  assert.deepEqual(loads, [{
    commandId: "load-command",
    checkpointId: "checkpoint-100-a",
  }]);
  assert.equal(browser.activeCheckpointPendingId, "checkpoint-100-a");
  assert.equal(browser.selectAdjacentCheckpoint("previous"), false);
  assert.equal(loads.length, 1);
});

test("adjacent checkpoint prefetch is deduplicated before reaching the worker", () => {
  const browser = Object.create(SourceBrowser.prototype);
  browser.route = {
    run_id: "gradlab-run",
    checkpoint_id: "checkpoint-200-b",
  };
  browser.adjacentPrefetchKey = "";
  browser.hasControl = () => true;
  browser.activeCheckpointItems = () => [
    {
      checkpoint_id: "checkpoint-100-a",
      run_id: "gradlab-run",
      manifest_url: "https://models.example/checkpoint-100/manifest.json",
      step: 100,
    },
    {
      checkpoint_id: "checkpoint-200-b",
      run_id: "gradlab-run",
      manifest_url: "https://models.example/checkpoint-200/manifest.json",
      step: 200,
    },
    {
      checkpoint_id: "checkpoint-300-c",
      run_id: "gradlab-run",
      manifest_url: "https://models.example/checkpoint-300/manifest.json",
      step: 300,
    },
  ];
  const commands = [];
  browser.command = (name, payload) => {
    commands.push({ name, payload });
    return "prefetch-command";
  };

  assert.equal(browser.prefetchAdjacentCheckpoints(), true);
  assert.equal(browser.prefetchAdjacentCheckpoints(), false);
  assert.equal(commands.length, 1);
  assert.equal(commands[0].name, "prefetch_sources");
  assert.deepEqual(
    commands[0].payload.sources.map((source) => source.checkpoint_id),
    ["checkpoint-100-a", "checkpoint-300-c"],
  );
});

test("catalog search filters the displayed authoritative page synchronously", () => {
  const mario = { run_id: "gradlab-mario", description: "Level 1-1" };
  const doom = { run_id: "gradlab-doom", description: "Deathmatch" };

  assert.equal(catalogItemMatchesSearch(mario, "level 1"), true);
  assert.equal(catalogItemMatchesSearch(doom, "level 1"), false);
  assert.equal(catalogItemMatchesSearch(doom, ""), true);
});

test("environment discovery prioritizes accepted and successful evidence", () => {
  assert.equal(environmentEvidenceRank({ success_badges: ["eval/success"] }), 0);
  assert.equal(environmentEvidenceRank({ success_badges: ["train/success"] }), 1);
  assert.equal(environmentEvidenceRank({}), 2);
});

test("goal variant selection uses the exact live activity diff", () => {
  const browser = Object.create(SourceBrowser.prototype);
  const state = browser.goalVariantDiffFromActivity({
    variant_id: "goal-variant-live",
    comparison_available: true,
    current_diff_count: 1,
    current_diff_count_exact: true,
    current_diff: [{ kind: "changed", path: "/train/checkpoint_freq", before: 1, after: 2 }],
    current_diff_truncated: false,
  });

  assert.equal(state.availability, "exact");
  assert.equal(state.changeCount, 1);
  assert.equal(state.entries[0].path, "/train/checkpoint_freq");
});

test("selecting a goal configuration collapses the expanded list", () => {
  const browser = Object.create(SourceBrowser.prototype);
  browser.selectedGoalVariantId = "goal-variant-current";
  browser.goalConfigurationsExpanded = true;
  browser.goalVariantDiffController = null;
  browser.goalVariantDiffSerial = 0;
  browser.goalVariantDiff = null;
  let renders = 0;
  browser.renderView = () => { renders += 1; };

  browser.selectGoalVariant({
    variant_id: "goal-variant-previous",
    comparison_available: false,
  });

  assert.equal(browser.selectedGoalVariantId, "goal-variant-previous");
  assert.equal(browser.goalConfigurationsExpanded, false);
  assert.equal(renders, 1);
});

const METRIC = "eval/full/episode/return/shaped/mean";

test("checkpoint selection boxes are centered and distinguish enabled from disabled", async () => {
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
    /\.source-selection-cell input \{[^}]*appearance: none;[^}]*display: grid;[^}]*margin: 0 auto;[^}]*border: 1px solid var\(--color-interaction\);/,
  );
  assert.match(styles, /\.source-selection-cell input:checked \{[^}]*background: var\(--color-interaction\);/);
  assert.match(styles, /\.source-selection-cell input:disabled \{[^}]*border-color: var\(--color-border\);[^}]*opacity: \.42;/);
});

test("checkpoint table uses compact metric labels with full accessible descriptions", async () => {
  const source = await readFile(
    new URL("../../src/gradlab/web_player/sources/browser.js", import.meta.url),
    "utf8",
  );

  assert.match(source, /fullLabel: column\.label \|\| metricLabel\(column\.metric\)/);
  assert.match(source, /label: checkpointMetricHeaderLabel\(column\)/);
  assert.match(source, /sortButton\.title = `\$\{fullLabel\}/);
  assert.match(source, /`Sort by \$\{fullLabel\}, \$\{nextDirection\}`/);
  assert.match(source, /table\.classList\.add\("checkpoint-table"\)/);
  assert.match(source, /indicator\.hidden = showingCheckpoints && !active/);
  assert.doesNotMatch(source, /roleLabel\.className = `checkpoint-metric-role/);
  assert.doesNotMatch(source, /\{ label: "Evaluation" \}/);
  assert.doesNotMatch(source, /\{ label: "Evidence" \}/);
  assert.doesNotMatch(source, /checkpoint-evidence/);
});

test("catalog list hover highlights the complete row", async () => {
  const styles = await readFile(
    new URL("../../src/gradlab/web_player/styles.css", import.meta.url),
    "utf8",
  );

  assert.match(
    styles,
    /\.environment-row:hover,\s*\.environment-row:focus-within\s*\{[^}]*background: var\(--color-surface-tertiary\);/,
  );
  assert.match(
    styles,
    /\.goal-row:hover,\s*\.goal-row:focus-within\s*\{[^}]*background: var\(--color-surface-tertiary\);/,
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
  assert.doesNotMatch(source, /renderSuccessBadges\(environment\)/);
  assert.match(source, /renderSuccessBadges\(goal\)/);
  assert.match(source, /renderSuccessBadges\(variant\)/);
  assert.match(source, /renderSuccessBadges\(run\)/);
  assert.match(source, /renderSuccessBadges\(item\)/);
  assert.match(source, /heading\.className = "goal-row-heading"/);
  assert.match(
    styles,
    /\.goal-row-heading\s*\{[^}]*display: flex;[^}]*align-items: center;[^}]*gap: \.45rem;/,
  );
  assert.match(
    styles,
    /\.success-badge\s*\{[^}]*padding: \.14rem \.32rem;[^}]*border: 1px solid color-mix\(in srgb, var\(--color-training-success-text\) 36%, transparent\);[^}]*background: color-mix\(in srgb, var\(--color-training-success-surface\) 48%, transparent\);[^}]*font-weight: var\(--font-weight-semibold\);[^}]*letter-spacing: 0;/,
  );
  assert.match(styles, /\.success-badge\.evaluation/);
  assert.match(
    styles,
    /\.success-badge\.training\s*\{[^}]*font-size: \.6875rem;[^}]*font-weight: var\(--font-weight-regular\);/,
  );
});

test("environment success is rendered as table status columns", async () => {
  assert.deepEqual(environmentSuccessStatus({
    run_count: 2,
    success_badges: ["train/success"],
  }, "train/success"), {
    label: "✅",
    className: "satisfied",
    description: "Training success satisfied",
  });
  assert.deepEqual(environmentSuccessStatus({
    run_count: 2,
    success_badges: ["train/success"],
  }, "eval/success"), {
    label: "❌",
    className: "unsatisfied",
    description: "Training runs exist, but evaluation success is not satisfied",
  });
  assert.deepEqual(environmentSuccessStatus({
    run_count: 0,
    success_badges: [],
  }, "train/success"), {
    label: "N/A",
    className: "not-applicable",
    description: "No runs yet",
  });

  const source = await readFile(
    new URL("../../src/gradlab/web_player/sources/browser.js", import.meta.url),
    "utf8",
  );
  assert.match(source, /document\.createElement\("table"\)/);
  assert.match(source, /\["Environment", "train\/success", "eval\/success", "Goals"\]/);
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
    sourceLabel: "Current",
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
    sourceLabel: "",
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
  assert.match(source, /this\.goalConfigurationsExpanded \? variants : \[selected\]/);
  assert.match(source, /this\.goalConfigurationsExpanded = false;\s*this\.renderView\(\);/);
  assert.doesNotMatch(source, /Older goal/);
  assert.match(source, /sourceLabel: "Current"/);
  assert.match(styles, /\.goal-configuration-toggle \{[^}]*text-transform: none;/);
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
    /\.goal-configuration-operation\.added,\s*\.goal-configuration-after\.added \{ color: var\(--color-evaluation-text\); \}/,
  );
  assert.match(
    styles,
    /\.goal-configuration-operation\.removed,\s*\.goal-configuration-after\.removed \{ color: var\(--color-error-text\); \}/,
  );
  assert.match(
    styles,
    /\.goal-configuration-operation\.changed,\s*\.goal-configuration-after\.changed \{ color: var\(--color-interaction-active\); \}/,
  );
});

test("goal activity unifies variants with recent and best runs", async () => {
  const source = await readFile(
    new URL("../../src/gradlab/web_player/sources/browser.js", import.meta.url),
    "utf8",
  );

  assert.match(source, /\/goals\/\$\{encodeURIComponent\(this\.route\.goal_id\)\}\/activity/);
  assert.doesNotMatch(source, /If-None-Match/);
  assert.match(source, /Array\.isArray\(variant\.current_diff\)/);
  assert.doesNotMatch(source, /Goal diff request failed/);
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
    /\.source-table tbody tr:hover,\s*\.source-table tbody tr:focus-visible\s*\{[^}]*background: var\(--color-interaction-tint\);/,
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
    /\.source-table \.finish-evidence-value \{[^}]*font-size: var\(--font-size-lg\);/,
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
      { label: "Defend The Line", current: false },
      { label: "Goal configuration", current: false },
      { label: "Run", current: false },
      { label: "Checkpoint · 10,002,432 steps", current: true },
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
    label: "Checkpoint · 10,002,432 steps",
    title: "checkpoint-10002432-b285ff3b",
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
        { label: "Checkpoint", current: true },
      ],
    );
  }
});

test("checkpoint recommendations prefer authoritative evidence over later diagnostic checkpoints", () => {
  const items = [
    {
      checkpoint_id: "checkpoint-300-training",
      step: 300,
      best_metrics: ["train/outcome/success/starts/all/rolling/rate/mean"],
    },
    {
      checkpoint_id: "checkpoint-200-accepted",
      step: 200,
      evaluation: { status: "accepted", pass: true },
    },
    { checkpoint_id: "checkpoint-400-final", step: 400, purpose: "final" },
  ];

  const result = checkpointRecommendation(items);
  assert.equal(result.item.checkpoint_id, "checkpoint-200-accepted");
  assert.equal(result.evidence.label, "Accepted");
  assert.match(checkpointEvidencePresentation(items[0]).detail, /evaluation remains authoritative/);
});

test("checkpoint recommendations surface exact evidence metrics and roles", () => {
  const metrics = checkpointRecommendationMetrics(
    {
      metrics: {
        "eval/full/outcome/success/starts/rate/min": 1,
        "train/episode/return/shaped/origin/target/rolling/mean": 3119.135,
      },
    },
    [
      {
        metric: "eval/full/outcome/success/starts/rate/min",
        evidence: "evaluation",
        roles: ["acceptance"],
      },
      {
        metric: "train/episode/return/shaped/origin/target/rolling/mean",
        evidence: "training",
        roles: ["training_proxy"],
      },
    ],
  );

  assert.deepEqual(metrics.map(({ role, tone, value }) => ({ role, tone, value })), [
    { role: "Acceptance", tone: "evaluation", value: "100%" },
    { role: "Training proxy", tone: "training", value: "3,119.135" },
  ]);
});

test("source discovery progressively discloses secondary controls", async () => {
  const source = await readFile(
    new URL("../../src/gradlab/web_player/sources/browser.js", import.meta.url),
    "utf8",
  );
  const html = await readFile(
    new URL("../../src/gradlab/web_player/index.html", import.meta.url),
    "utf8",
  );

  assert.match(source, /return "Choose a goal version"/);
  assert.match(source, /source-search-disclosure/);
  assert.match(source, /disclosure\.open = this\.searchOpen/);
  assert.doesNotMatch(source, /resultCount > 8/);
  assert.match(html, /id="contract-search-disclosure" class="contract-search-disclosure"/);
  assert.match(source, /Contract differences · \$\{presentation\.differenceLabel\}/);
  assert.doesNotMatch(source, /Evaluation & technical details/);
  assert.doesNotMatch(source, /Compare all checkpoints/);
  assert.match(source, /body\.append\(this\.renderEvaluationActions\(\), results\)/);
});

test("catalog refresh animates and disables only the header refresh control", async () => {
  const source = await readFile(
    new URL("../../src/gradlab/web_player/sources/browser.js", import.meta.url),
    "utf8",
  );
  const styles = await readFile(
    new URL("../../src/gradlab/web_player/styles.css", import.meta.url),
    "utf8",
  );

  assert.doesNotMatch(source, /source-list-loading-indicator/);
  assert.match(source, /const refresh = button\("", \{ iconName: "refresh", quiet: true \}\)/);
  assert.match(source, /refresh\.classList\.add\("icon-only"\)/);
  assert.match(source, /if \(this\.loading\) refresh\.classList\.add\("refreshing"\)/);
  assert.match(source, /refresh\.setAttribute\("aria-label", this\.loading \? "Refreshing" : "Refresh"\)/);
  assert.match(source, /refresh\.disabled = this\.loading/);
  assert.doesNotMatch(source, /button\("Refresh", \{ iconName: "refresh", quiet: true \}\)/);
  assert.doesNotMatch(source, /loadingState\("Loading catalog…"\)/);
  assert.doesNotMatch(source, /Some catalog evidence is unavailable/);
  assert.doesNotMatch(styles, /source-list-loading-indicator/);
  assert.match(
    styles,
    /\.source-head \.icon-only\.refreshing \.icon \{ animation: source-spin \.8s linear infinite; \}/,
  );
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

test("hidden breadcrumbs do not restart active checkpoint loading", () => {
  const browser = Object.create(SourceBrowser.prototype);
  browser.activeBreadcrumbRoute = "";
  browser.breadcrumbsRoot = {
    hidden: true,
    replaceChildren() {},
  };
  browser.stop = () => {};
  const historyModes = [];
  browser.syncUrl = (mode) => historyModes.push(mode);
  let loads = 0;
  browser.loadActiveCheckpointNavigation = () => {
    loads += 1;
  };
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
  browser.breadcrumbsRoot.hidden = true;
  browser.renderActiveBreadcrumbs(snapshot);

  assert.equal(renders, 1);
  assert.equal(loads, 1);
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

test("browsing from an active player keeps the current runner available", () => {
  const browser = Object.create(SourceBrowser.prototype);
  browser.app = { has_active_runner: true };
  browser.route = {
    level: "runs",
    environment_id: "ViZDoom",
    goal_id: "DefendTheLine-v1",
    goal_variant_id: "goal-variant-a27a8239",
    run_id: "gradlab-c22f7c7a",
    checkpoint_id: "checkpoint-a",
  };
  const commands = [];
  browser.command = (...args) => commands.push(args);
  browser.applyRoute = (route) => {
    browser.route = { ...browser.route, ...route };
  };
  browser.syncUrl = () => {};
  browser.openSourceRoute = () => {};

  assert.equal(browser.navigate({ level: "runs", checkpoint_id: "" }), true);
  assert.deepEqual(commands, []);
  assert.equal(browser.route.checkpoint_id, "");
});

test("continue playback context retains the active checkpoint while browsing", () => {
  const activeRoute = {
    level: "runs",
    environment_id: "ViZDoom",
    goal_id: "DefendTheLine-v1",
    goal_variant_id: "goal-variant-a27a8239",
    run_id: "gradlab-c22f7c7a",
    checkpoint_id: "checkpoint-10002432-b285ff3b",
  };
  const browser = Object.create(SourceBrowser.prototype);
  browser.getState = () => ({
    backgroundPlaybackSnapshot: { app: { route: activeRoute } },
  });
  browser.playbackRoute = null;
  browser.app = {
    route: {
      level: "runs",
      environment_id: "ViZDoom",
      goal_id: "DefendTheLine-v1",
      goal_variant_id: "goal-variant-a27a8239",
      run_id: "gradlab-c22f7c7a",
      checkpoint_id: "",
    },
  };

  assert.deepEqual(browser.currentPlaybackRoute(), activeRoute);
});

test("continue playback banner identifies the checkpoint and uses the eye icon", async () => {
  const source = await readFile(
    new URL("../../src/gradlab/web_player/sources/browser.js", import.meta.url),
    "utf8",
  );
  const styles = await readFile(
    new URL("../../src/gradlab/web_player/styles.css", import.meta.url),
    "utf8",
  );
  const sprite = await readFile(
    new URL("../../src/gradlab/web_player/tabler-icons.svg", import.meta.url),
    "utf8",
  );

  assert.match(source, /sourceBreadcrumbItems\(playbackRoute\)\.slice\(1\)/);
  assert.match(source, /className = "continue-playback-breadcrumb"/);
  assert.match(source, /button\("Continue watching", \{ iconName: "eye", primary: true \}\)/);
  assert.match(styles, /\.continue-playback-breadcrumb \{/);
  assert.match(sprite, /id="ti-eye"/);
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
    metricLabel("train/outcome/success/starts/all/rolling/rate/min"),
    "Recent all-start success rate min",
  );
  assert.equal(
    metricLabel("train/outcome/success/starts/all/rolling/rate/mean"),
    "Recent all-start success rate mean",
  );
  assert.equal(
    metricLabel("train/episode/return/shaped/origin/target/rolling/mean"),
    "Recent target return mean",
  );
  assert.equal(
    metricLabel("train/progress/kills/origin/target/rolling/mean"),
    "Recent target kills mean",
  );
  assert.equal(
    formatMetricValue("eval/full/outcome/success/starts/rate/min", 0.875),
    "87.5%",
  );
  assert.equal(formatMetricValue(METRIC, null), "—");
});

test("checkpoint metric headers preserve semantics in one short line", () => {
  assert.equal(checkpointMetricHeaderLabel({
    metric: "eval/full/outcome/success/starts/rate/min",
    evidence: "evaluation",
  }), "Eval success");
  assert.equal(checkpointMetricHeaderLabel({
    metric: "train/outcome/success/starts/all/rolling/rate/min",
    evidence: "training",
  }), "Train success");
  assert.equal(checkpointMetricHeaderLabel({
    metric: "eval/full/episode/return/shaped/mean",
    evidence: "evaluation",
  }), "Eval return");
  assert.equal(checkpointMetricHeaderLabel({
    metric: "train/episode/return/shaped/origin/target/rolling/mean",
    evidence: "training",
  }), "Train return");
  assert.equal(checkpointMetricHeaderLabel({
    metric: "eval/full/progress/kills/mean",
    evidence: "evaluation",
  }), "Eval kills");
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
        metric: "train/episode/return/shaped/origin/target/rolling/mean",
        value: 5.25,
        condition: {
          metric: "train/episode/return/shaped/origin/target/rolling/mean",
          trigger: "threshold",
          operator: ">=",
          threshold: 5,
        },
      },
    }),
    {
      label: "Training target met",
      detail: (
        "Recent target return mean ≥ 5"
        + " · observed 5.25 · Stopped at 16,384 steps"
      ),
      evidence: {
        metric: "Recent target return mean",
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
  const trainSuccess = "train/outcome/success/starts/all/rolling/rate/mean";
  const evalReturn = "eval/full/episode/return/shaped/mean";
  const checkpoint = { best_metrics: [trainSuccess, evalReturn] };

  assert.equal(checkpointMetricIsBest(checkpoint, trainSuccess), true);
  assert.equal(checkpointMetricIsBest(checkpoint, evalReturn), true);
  assert.equal(
    checkpointMetricIsBest(checkpoint, "eval/full/outcome/success/starts/rate/mean"),
    false,
  );
  assert.equal(checkpointMetricIsBest({}, trainSuccess), false);

  const objective = {
    evidence: "evaluation",
    direction: "max",
    roles: ["objective", "acceptance"],
  };
  const proxy = {
    evidence: "training",
    direction: "max",
    roles: ["training_proxy"],
  };
  assert.equal(checkpointMetricRoleLabel(objective), "Objective · gate");
  assert.equal(checkpointMetricRoleLabel(proxy), "Training proxy");
  assert.equal(checkpointMetricBestBadge(objective), "Best");
  assert.equal(checkpointMetricBestBadge(proxy), "Best");
  assert.match(checkpointMetricDescription(objective), /Frozen checkpoint-evaluation evidence/);
  assert.match(checkpointMetricDescription(proxy), /Diagnostic online training proxy/);

  const source = await readFile(
    new URL("../../src/gradlab/web_player/sources/browser.js", import.meta.url),
    "utf8",
  );
  assert.match(source, /"checkpoint-metric-cell"/);
  assert.match(source, /badge\.className = "checkpoint-best-badge"/);
  assert.match(source, /badge\.textContent = badgeLabel/);
  assert.doesNotMatch(source, /Best objective/);
  assert.doesNotMatch(source, /Best observed/);

  const styles = await readFile(
    new URL("../../src/gradlab/web_player/styles.css", import.meta.url),
    "utf8",
  );
  assert.match(
    styles,
    /\.source-table \.checkpoint-best-badge \{[^}]*border: 1px solid var\(--color-training-success-text\);[^}]*background: var\(--color-training-success-surface\);[^}]*color: var\(--color-training-success-text\);/,
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

test("catalog checkpoint playback preserves public-run publication identity", async () => {
  const source = await readFile(
    new URL("../../src/gradlab/web_player/sources/browser.js", import.meta.url),
    "utf8",
  );
  assert.match(
    source,
    /selectCheckpoint[\s\S]*?kind: "public_run",[\s\S]*?value: item\.manifest_url/,
  );
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
      metric: "train/outcome/success/starts/all/rolling/rate/min",
      direction: "max",
    },
    { metric: "train/global_step", direction: "min" },
  ];
  const items = [
    {
      run_id: "training-only",
      recipe: "fast-training",
      metrics: {
        "train/outcome/success/starts/all/rolling/rate/min": 1,
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
      metric: "train/outcome/success/starts/all/rolling/rate/min",
      direction: "max",
    },
    { metric: "train/global_step", direction: "min" },
  ];
  const items = [
    {
      run_id: "slower",
      metrics: {
        "train/outcome/success/starts/all/rolling/rate/min": 0.9,
        "train/global_step": 2_000,
      },
    },
    {
      run_id: "faster",
      metrics: {
        "train/outcome/success/starts/all/rolling/rate/min": 0.9,
        "train/global_step": 1_000,
      },
    },
  ];

  assert.equal(activeRunMetricColumns(items, primary, fallback), fallback);
  const leader = bestRunEfficiency(items, primary, fallback);
  assert.equal(leader.evidence, "training");
  assert.equal(leader.item.run_id, "faster");
});
