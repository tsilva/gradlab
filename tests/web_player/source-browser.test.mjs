import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { renderComponent } from "./helpers/svelte-render.mjs";

async function sourceHtml(overrides = {}) {
  const presentation = {
    route: { level: "runs", run_id: "run-a" }, items: [], sourceItems: [],
    loading: false, loadedKey: "loaded", error: "", query: "", searchOpen: false,
    app: { phase: "selecting" }, hasControl: true, heading: "Choose a run",
    selectedCheckpoints: new Set(), goalVariantRunPages: new Map(),
    favoriteEnvironments: new Set(), selectedGoalVariantId: "current",
    goalVariantDiff: null, metricColumns: [], fallbackMetricColumns: [],
    sort: { metric: "", direction: "" }, refreshing: false,
    searchPlaceholder: "Search checkpoints", ...overrides,
  };
  presentation.table = sourceTablePresentation(presentation);
  return renderComponent(new URL("../../frontend/components/SourceBrowser.svelte", import.meta.url), {
    presentation, controller: { embeddedGoalRunsHaveMore: () => false },
  });
}
const checkpoint = {checkpoint_id: "checkpoint-a", step: 100, metrics: {}};
const configuration = {
  variant_id: "current", configuration_kind: "current_default", goal_contract_sha256: "abcdef123456",
  run_count: 1, recent_runs: [],
};


import {
  sourceTablePresentation,
  activeRunMetricColumns,
  availableRunMetricColumns,
  bestRunEfficiency,
  catalogItemMatchesSearch,
  checkpointCanEvaluate,
  checkpointEvaluationPresentation,
  checkpointMetricDescription,
  checkpointMetricHeaderLabel,
  checkpointMetricIsLoading,
  checkpointMetricRoleLabel,
  checkpointNavigationPresentation,
  checkpointPrefetchSources,
  checkpointPlaybackSeed,
  environmentEvidenceRank,
  readEnvironmentFavorites,
  environmentSuccessStatus,
  formatGoalDiffValue,
  formatGoalConfigurationDate,
  formatMetricValue,
  groupGoalConfigurations,
  goalConfigurationPresentation,
  goalConfigurationSummary,
  metricLabel,
  rankRunItems,
  runEvaluationEvidenceStatus,
  runFinishPresentation,
  runStatePresentation,
  runTrainingEvidenceStatus,
  SourceBrowser,
  successBadgeLabels,
  sortRunItems,
  sortEnvironmentItems,
  sourceBreadcrumbItems,
  sourceRouteFromPath,
  sourceRoutePath,
  toggleEnvironmentFavorite,
  writeEnvironmentFavorites,
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
  const html = await renderComponent(new URL("../../frontend/components/SourceNavigation.svelte", import.meta.url), {controller: {}, kind: "checkpoints", presentation: {position: "2 / 3", previousDisabled: false, nextDisabled: false}});
  assert.match(html, /2 \/ 3/);
  assert.doesNotMatch(html, /Checkpoint 2/);
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

test("checkpoint navigation uses the session's fetched table", async () => {
  const browser = Object.create(SourceBrowser.prototype);
  const route = { run_id: "run-a", goal_variant_id: "variant-a", checkpoint_id: "checkpoint-1" };
  browser.activeCheckpointCache = new Map([[JSON.stringify({
    run_id: "run-a", goal_variant_id: "variant-a",
  }), [{ checkpoint_id: "checkpoint-1" }]]]);
  browser.activeCheckpointRequestSerial = 0;
  browser.activeCheckpointError = "old error";
  const controller = new AbortController();
  browser.activeCheckpointController = controller;
  const prefetched = [];
  browser.prefetchAdjacentCheckpoints = (value) => prefetched.push(value);

  await browser.loadActiveCheckpointNavigation(route, "route");

  assert.equal(controller.signal.aborted, true);
  assert.equal(browser.activeCheckpointRequestSerial, 1);
  assert.equal(browser.activeCheckpointError, "");
  assert.deepEqual(prefetched, [route]);
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

test("environment favorites persist defensively in browser storage", () => {
  const values = new Map();
  const storage = {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
  };
  const cookieTarget = { cookie: "" };

  assert.equal(
    writeEnvironmentFavorites(new Set(["VizDoom", "Acrobot"]), storage, cookieTarget),
    true,
  );
  assert.deepEqual([...readEnvironmentFavorites(storage)], ["Acrobot", "VizDoom"]);
  assert.match(
    cookieTarget.cookie,
    /^gradlab_playback_favorite_environments_v1=%5B%22Acrobot%22%2C%22VizDoom%22%5D;/,
  );

  storage.setItem("gradlab.playback.favorite-environments.v1", "not-json");
  assert.deepEqual([...readEnvironmentFavorites(storage)], []);
  assert.equal(writeEnvironmentFavorites(new Set(["Mario"]), null), false);
});

test("environment favorites use a host cookie across ephemeral player ports", () => {
  const storage = {
    getItem: () => JSON.stringify(["Stale local favorite"]),
    setItem: () => {},
  };
  const cookie = [
    "unrelated=value",
    "gradlab_playback_favorite_environments_v1=%5B%22Mario%22%2C%22VizDoom%22%5D",
  ].join("; ");

  assert.deepEqual([...readEnvironmentFavorites(storage, cookie)], ["Mario", "VizDoom"]);
  assert.deepEqual([
    ...readEnvironmentFavorites(storage, "gradlab_playback_favorite_environments_v1=%5B%5D"),
  ], []);
  assert.deepEqual(
    [...readEnvironmentFavorites(storage, "gradlab_playback_favorite_environments_v1=not-json")],
    ["Stale local favorite"],
  );
});

test("favorite environments sort first and alphabetically", () => {
  const environments = [
    { name: "Zulu", success_badges: ["eval/success"] },
    { name: "Mario" },
    { name: "Acrobot" },
    { name: "CartPole", success_badges: ["train/success"] },
  ];

  assert.deepEqual(
    sortEnvironmentItems(environments, new Set(["Mario", "Acrobot"]))
      .map((environment) => environment.name),
    ["Acrobot", "Mario", "Zulu", "CartPole"],
  );
});

test("environment favorites toggle on and back off", () => {
  const selected = toggleEnvironmentFavorite(new Set(["Acrobot"]), "Mario");
  assert.deepEqual([...selected], ["Acrobot", "Mario"]);
  assert.deepEqual([...toggleEnvironmentFavorite(selected, "Mario")], ["Acrobot"]);
});

test("environment favorite controls use a thin outline and filled selected star", async () => {
  const html = await sourceHtml({route: {level: "environments"}, items: [{name: "Mario", goal_count: 1}, {name: "Doom", goal_count: 2}], favoriteEnvironments: new Set(["Mario"])});
  assert.match(html, /ti-star-filled/);
  assert.match(html, /ti-star["<]/);
  assert.match(html, /aria-pressed="true"/);
  assert.match(html, /Remove Mario from favorites/);
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

test("goal configurations group launch overrides under their exact revisions", () => {
  const currentDefault = {
    variant_id: "current-default",
    goal_contract_sha256: "revision-current",
    configuration_kind: "current_default",
    source_relation: "canonical",
  };
  const historicalDefault = {
    variant_id: "historical-default",
    goal_contract_sha256: "revision-historical",
    configuration_kind: "previous_default",
    source_relation: "canonical",
  };
  const currentOverride = {
    variant_id: "current-override",
    goal_contract_sha256: "revision-current",
    configuration_kind: "current_modified",
    source_relation: "launch_override",
  };
  const historicalOverride = {
    variant_id: "historical-override",
    goal_contract_sha256: "revision-historical",
    configuration_kind: "previous_modified",
    source_relation: "launch_override",
  };

  const groups = groupGoalConfigurations([
    historicalDefault,
    currentOverride,
    currentDefault,
    historicalOverride,
  ]);

  assert.equal(groups.length, 2);
  assert.equal(groups[0].revisionId, "revision-current");
  assert.equal(groups[0].defaultVariant, currentDefault);
  assert.deepEqual(groups[0].overrides, [currentOverride]);
  assert.equal(groups[1].revisionId, "revision-historical");
  assert.equal(groups[1].defaultVariant, historicalDefault);
  assert.deepEqual(groups[1].overrides, [historicalOverride]);
});

test("selecting a goal configuration updates the master-detail selection", () => {
  const browser = Object.create(SourceBrowser.prototype);
  browser.selectedGoalVariantId = "goal-variant-current";
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
  assert.equal(renders, 1);
});

test("run evidence status distinguishes missing evaluation from failed evaluation", () => {
  assert.deepEqual(runTrainingEvidenceStatus({ success_badges: ["train/success"] }), {
    label: "✅",
    className: "met",
    description: "The Run met its declared Training Success proxy",
  });
  assert.equal(runTrainingEvidenceStatus({ state: "running" }).label, "⏳");
  assert.equal(runTrainingEvidenceStatus({ state: "finished" }).label, "❌");
  const historical = runTrainingEvidenceStatus({
    state: "finished",
    training_success: {
      criterion: {
        metric: "train/progress/bricks_destroyed_normalized/mean",
        operator: ">=",
        threshold: 0.5,
      },
      status: "not_met",
      best: { step: 189161472, value: 0.4992129623889923 },
    },
  });
  assert.equal(historical.label, "❌");
  assert.match(historical.description, /0\.4992129623889923 at step 189161472/);

  assert.deepEqual(runEvaluationEvidenceStatus({ evaluation_status: "not_evaluated" }), {
    label: "∅",
    className: "not-evaluated",
    description: "No evaluation evidence is available for this Run",
  });
  assert.equal(runEvaluationEvidenceStatus({ evaluation_status: "in_progress" }).label, "⏳");
  assert.equal(runEvaluationEvidenceStatus({ evaluation_status: "not_accepted" }).label, "❌");
  assert.equal(runEvaluationEvidenceStatus({ success_badges: ["eval/success"] }).label, "✅");
});

const METRIC = "eval/return/mean";

test("checkpoint selection boxes are centered and distinguish enabled from disabled", async () => {
  const styles = await readFile(
    new URL("../../src/gradlab/web_player/styles.css", import.meta.url),
    "utf8",
  );

  assert.match(
    styles,
    /\.source-table\.checkpoint-table \.source-selection-cell \{[^}]*vertical-align: middle;/,
  );
  assert.match(
    styles,
    /\.source-selection-cell input \{[^}]*appearance: none;[^}]*display: grid;[^}]*margin: 0 auto;[^}]*border: 1px solid var\(--color-interaction\);/,
  );
  assert.match(styles, /\.source-selection-cell input:checked \{[^}]*background: var\(--color-interaction\);/);
  assert.match(styles, /\.source-selection-cell input:disabled \{[^}]*border-color: var\(--color-border\);[^}]*opacity: \.42;/);
});

test("checkpoint table keeps full metric descriptions for accessible sorting", async () => {
  const html = await sourceHtml({items: [checkpoint], metricColumns: [{metric: "eval/return/mean", label: "eval/return/mean", evidence: "evaluation", direction: "max"}]});
  assert.match(html, /Sort by eval\/return\/mean, descending/);
  assert.match(html, /Frozen checkpoint-evaluation evidence/);
  assert.match(html, /<wbr/);
  assert.match(html, /scope="colgroup"[^>]*>Eval/);
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

  const html = await sourceHtml({route: {level: "runs", run_id: ""}, items: [{run_id: "run-a", metrics: {}, success_badges: ["eval/success", "train/success"]}]});
  assert.match(html, /success-badge training/);
  assert.match(html, /success-badge evaluation/);
  assert.ok(html.indexOf(">train/success<") < html.indexOf(">eval/success<"));
  assert.match(html, /verified evaluation/);
});

test("goals render as table status columns", async () => {
  const html = await sourceHtml({route: {level: "goals"}, items: [{goal_id: "Level1-1", goal_slug: "level", title: "Finish level", recipe_count: 2, success_badges: ["train/success"]}]});
  assert.match(html, /class="goal-table"/);
  assert.match(html, /train\/success/);
  assert.match(html, /eval\/success/);
  assert.match(html, /Inspect Level1-1 YAML/);
});

test("environment success is rendered as table status columns", async () => {
  const html = await sourceHtml({route: {level: "environments"}, items: [{name: "Mario", goal_count: 2, success_badges: ["train/success"]}]});
  assert.match(html, /class="environment-table"/);
  assert.match(html, /train\/success/);
  assert.match(html, /eval\/success/);
  assert.match(html, /class="environment-status/);
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
  assert.equal(older.firstUsedDate, "3 days ago");
  assert.equal(older.lastActivityDate, "2 days ago");
  assert.equal(older.differenceLabel, "Exact diff unavailable");
});

test("goal configuration ages cross midnight and fall back to dates at 30 days", () => {
  const now = Date.parse("2026-09-16T00:05:00Z");
  const ago = (milliseconds) => new Date(now - milliseconds).toISOString();
  assert.equal(formatGoalConfigurationDate(ago(0), now), "just now");
  assert.equal(formatGoalConfigurationDate(ago(1_000), now), "1 second ago");
  assert.equal(formatGoalConfigurationDate(ago(59_000), now), "59 seconds ago");
  assert.equal(formatGoalConfigurationDate(ago(60_000), now), "1 minute ago");
  assert.equal(formatGoalConfigurationDate(ago(600_000), now), "10 minutes ago");
  assert.equal(formatGoalConfigurationDate(ago(3_600_000), now), "1 hour ago");
  assert.equal(formatGoalConfigurationDate(ago(86_400_000), now), "1 day ago");
  assert.equal(formatGoalConfigurationDate(ago(29 * 86_400_000), now), "29 days ago");
  assert.equal(formatGoalConfigurationDate(ago(30 * 86_400_000), now), "17 Aug 2026");
  assert.equal(formatGoalConfigurationDate(ago(-3_600_000), now), "in 1 hour");
  assert.equal(formatGoalConfigurationDate("invalid", now), "—");
  assert.equal(formatGoalConfigurationDate(null, now), "—");
  assert.equal(formatGoalConfigurationDate("2025-01-02T23:00:00Z", now), "2 Jan 2025");
});

test("goal diff values preserve JSON types", () => {
  assert.equal(formatGoalDiffValue(false), "false");
  assert.equal(formatGoalDiffValue("discrete"), '"discrete"');
  assert.equal(formatGoalDiffValue({ threshold: 10 }), '{"threshold":10}');
  assert.equal(formatGoalDiffValue(null), "null");
  assert.equal(formatGoalDiffValue(null, { unavailable: true }), "—");
});

test("goal configuration summaries preserve useful science without leaking raw contracts", () => {
  const presentation = {
    kind: "previous_default",
    differenceCount: 5,
    differenceLabel: "5 changes",
  };
  assert.equal(goalConfigurationSummary({
    display_label: "Eval → {\"acceptance\":[{\"metric\":\"eval/return/mean\"}]} · Evaluation mode training_only → evaluated · +3 more",
  }, presentation), "Evaluation mode training_only → evaluated · 5 changes total");
  assert.equal(goalConfigurationSummary({ display_label: "" }, presentation), "5 changes from current goal");
});

test("goal configurations render as a master-detail browser with exact changes", async () => {
  const variant = {...configuration, configuration_kind: "previous_modified", comparison_available: true, current_diff_count: 1, current_diff_count_exact: true};
  const html = await sourceHtml({route: {level: "goal_variants"}, items: [variant], goalVariantDiff: {state: "ready", availability: "exact", entries: [{path: "eval.metric", kind: "changed", before: "old", after: "new"}]}});
  for (const value of ["goal-configuration-layout", "Goal configurations", "aria-pressed=\"true\"", "History", "First used", "Last activity", "View goal YAML", "Exact contract path", "Before", "After", "eval.metric", "goal-configuration-after changed"]) assert.ok(html.includes(value), value);
  assert.doesNotMatch(html, /treegrid/);
});

test("goal activity renders recent runs as a compact success table", async () => {
  const html = await sourceHtml({route: {level: "goal_variants"}, items: [{...configuration, recent_runs: [{run_id: "run-a", recipe: "ppo", seed: 7, state: "running", success_badges: ["train/success"]}]}]});
  for (const value of ["goal-configuration-run-table", "goal-configuration-run-identity", "train/success", "eval/success", "Copy ID", "View checkpoints for ppo"]) assert.ok(html.includes(value), value);
  assert.doesNotMatch(html, /goal-configuration-run-sort/);
});

test("catalog pages refresh only on explicit request", async () => {
  const source = await readFile(new URL("../../src/gradlab/web_player/sources/browser.js", import.meta.url), "utf8");
  assert.doesNotMatch(source, /setInterval/);
  const html = await sourceHtml({items: [checkpoint]});
  assert.match(html, /aria-label="Refresh"/);
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
  const html = await sourceHtml({route: {level: "runs", run_id: ""}, items: [{run_id: "run-a", metrics: {"train/success/mean": 1}}], fallbackMetricColumns: [{metric: "train/success/mean", direction: "max", evidence: "training"}]});
  assert.match(html, /run-cell[\s\S]*Training lead[\s\S]*recipe-cell/);
  assert.doesNotMatch(html, /recipe-cell[^<]*Training lead/);
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

test("checkpoint selection shows the authoritative training run state", async () => {
  const html = await sourceHtml({items: [checkpoint], runStatus: {state: "running", updated_at: "2026-01-01"}});
  assert.match(html, /Training run state: Running/);
  assert.match(html, /ti-activity-heartbeat/);
  assert.match(html, /Training run/);
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

test("run checkpoint discovery omits the redundant recommendation banner", async () => {
  const source = await readFile(
    new URL("../../src/gradlab/web_player/sources/browser.js", import.meta.url),
    "utf8",
  );
  const styles = await readFile(
    new URL("../../src/gradlab/web_player/styles.css", import.meta.url),
    "utf8",
  );

  assert.doesNotMatch(source, /BEST AVAILABLE TO WATCH/);
  assert.doesNotMatch(source, /renderCheckpointRecommendation/);
  assert.doesNotMatch(styles, /checkpoint-recommendation/);
});

test("source discovery progressively discloses secondary controls", async () => {
  const html = await sourceHtml({items: [checkpoint], searchOpen: false});
  assert.match(html, /<details class="source-search-disclosure">/);
  assert.match(html, /Close search/);
  assert.match(html, /Evaluate selected/);
  assert.match(html, /Inspect run YAML/);
  const open = await sourceHtml({items: [checkpoint], searchOpen: true});
  assert.match(open, /<details class="source-search-disclosure" open(?:="")?>/);
});

test("catalog refresh animates and disables only the header refresh control", async () => {
  const html = await sourceHtml({route: {level: "goals"}, loading: true, refreshing: true, loadedKey: ""});
  assert.match(html, /class="quiet icon-only refreshing"/);
  assert.match(html, /aria-label="Refreshing"/);
  assert.match(html, /aria-busy="true"/);
  assert.match(html, /disabled/);
  assert.match(html, /Loading table value/);
  assert.doesNotMatch(html, /Loading catalog/);
});

test("returning to environments restores the completed table without refreshing", async (context) => {
  const originalLocation = globalThis.location;
  const originalFetch = globalThis.fetch;
  const requests = [];
  globalThis.location = { pathname: "/embedded-player", search: "", hash: "" };
  globalThis.fetch = (...args) => {
    requests.push(args);
    throw new Error("cached environments should not request the catalog");
  };
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
  sourceBrowser.sourceItems = [{ name: "Mario", goal_count: 18 }];
  sourceBrowser.items = [...sourceBrowser.sourceItems];
  sourceBrowser.loadedKey = sourceBrowser.routeKey();
  sourceBrowser.rememberEnvironmentCatalog();

  sourceBrowser.route = {
    ...sourceBrowser.route,
    level: "goals",
    environment_id: "Mario",
  };
  sourceBrowser.sourceItems = [{ goal_id: "Level1-1" }];
  sourceBrowser.items = [...sourceBrowser.sourceItems];
  sourceBrowser.renderView = () => {};

  sourceBrowser.applyRoute({
    level: "environments",
    environment_id: "",
    goal_id: "",
  });
  await new Promise(setImmediate);

  assert.deepEqual(sourceBrowser.items, [{ name: "Mario", goal_count: 18 }]);
  assert.equal(sourceBrowser.loadedKey, sourceBrowser.routeKey());
  assert.equal(requests.length, 0);
});

test("returning to goals restores the last table without refreshing", async (context) => {
  const originalLocation = globalThis.location;
  const originalFetch = globalThis.fetch;
  const requests = [];
  globalThis.location = { pathname: "/embedded-player", search: "", hash: "" };
  globalThis.fetch = (...args) => {
    requests.push(args);
    throw new Error("cached goals should not request the catalog");
  };
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
  sourceBrowser.route = {
    ...sourceBrowser.route,
    level: "goals",
    environment_id: "Mario",
  };
  sourceBrowser.sourceItems = [{ goal_id: "Level1-1", recipe_count: 1 }];
  sourceBrowser.items = [...sourceBrowser.sourceItems];
  sourceBrowser.loadedKey = sourceBrowser.routeKey();
  sourceBrowser.rememberGoalCatalog();
  sourceBrowser.route = {
    ...sourceBrowser.route,
    level: "goal_variants",
    goal_id: "Level1-1",
  };
  sourceBrowser.renderView = () => {};

  sourceBrowser.applyRoute({
    level: "goals",
    goal_id: "",
    goal_variant_id: "",
  });
  await new Promise((resolve) => setImmediate(resolve));

  assert.deepEqual(sourceBrowser.items, [{ goal_id: "Level1-1", recipe_count: 1 }]);
  assert.equal(sourceBrowser.loadedKey, sourceBrowser.routeKey());
  assert.equal(requests.length, 0);
});

test("repeating a search and clearing it restore session results without requests", (context) => {
  const originalLocation = globalThis.location;
  globalThis.location = { pathname: "/embedded-player", search: "", hash: "" };
  context.after(() => {
    if (originalLocation === undefined) delete globalThis.location;
    else globalThis.location = originalLocation;
  });
  const browser = new SourceBrowser({}, { replaceChildren() {}, hidden: false }, {
    token: "token", command() {}, getState: () => ({ hasControl: true }), showToast() {},
  });
  browser.renderView = () => {};
  browser.sourceItems = [{ name: "Mario" }, { name: "Doom" }];
  browser.loadedKey = browser.routeKey();
  browser.rememberEnvironmentCatalog();
  browser.query = "mario";
  browser.sourceItems = [{ name: "Mario" }];
  browser.loadedKey = browser.routeKey();
  browser.rememberSearchCatalog();

  browser.query = "other";
  browser.sourceItems = [];
  browser.setSearch("mario");
  assert.deepEqual(browser.items, [{ name: "Mario" }]);
  assert.equal(browser.loadedKey, browser.routeKey());
  browser.setSearch("");
  assert.deepEqual(browser.items, [{ name: "Mario" }, { name: "Doom" }]);
  assert.equal(browser.loadedKey, browser.routeKey());
});

test("revisiting Goal Variant activity restores the page without a request", async (context) => {
  const originalLocation = globalThis.location;
  const originalFetch = globalThis.fetch;
  const requests = [];
  globalThis.location = { pathname: "/embedded-player", search: "", hash: "" };
  globalThis.fetch = (...args) => {
    requests.push(args);
    throw new Error("cached activity should not request the catalog");
  };
  context.after(() => {
    if (originalLocation === undefined) delete globalThis.location;
    else globalThis.location = originalLocation;
    if (originalFetch === undefined) delete globalThis.fetch;
    else globalThis.fetch = originalFetch;
  });
  const browser = new SourceBrowser({}, { replaceChildren() {}, hidden: false }, {
    token: "token", command() {}, getState: () => ({ hasControl: true }), showToast() {},
  });
  browser.renderView = () => {};
  browser.route = { ...browser.route, level: "goal_variants", environment_id: "Mario", goal_id: "Level1-1" };
  browser.sourceItems = [{ variant_id: "current", run_count: 2 }];
  browser.goalVariantRunPages.set("current", { loaded: true, items: [{ run_id: "run-a" }], nextCursor: null });
  browser.loadedKey = browser.routeKey();
  browser.rememberGoalActivity();
  browser.route = { ...browser.route, level: "goals", goal_id: "" };

  browser.applyRoute({ level: "goal_variants", goal_id: "Level1-1" });
  await new Promise(setImmediate);

  assert.deepEqual(browser.items, [{ variant_id: "current", run_count: 2 }]);
  assert.deepEqual(browser.goalVariantRunPages.get("current").items, [{ run_id: "run-a" }]);
  assert.equal(browser.loadedKey, browser.routeKey());
  assert.equal(requests.length, 0);
});

test("checkpoint table returns from memory only within the current player session", async (context) => {
  const originalLocation = globalThis.location;
  const originalFetch = globalThis.fetch;
  const requests = [];
  globalThis.location = { pathname: "/embedded-player", search: "", hash: "" };
  globalThis.fetch = (...args) => {
    requests.push(args);
    throw new Error("cached checkpoints should not request the catalog");
  };
  context.after(() => {
    if (originalLocation === undefined) delete globalThis.location;
    else globalThis.location = originalLocation;
    if (originalFetch === undefined) delete globalThis.fetch;
    else globalThis.fetch = originalFetch;
  });

  const createBrowser = () => {
    const browser = new SourceBrowser({}, { replaceChildren() {}, hidden: false }, {
      token: "token", command() {}, getState: () => ({ hasControl: true }), showToast() {},
    });
    browser.renderView = () => {};
    return browser;
  };
  const browser = createBrowser();
  browser.route = { ...browser.route, level: "runs", run_id: "run-a", goal_variant_id: "variant-a" };
  browser.sourceItems = [{ checkpoint_id: "checkpoint-1", step: 100, metrics: { "train/return": 3 } }];
  browser.items = [...browser.sourceItems];
  browser.metricColumns = [{ metric: "train/return", evidence: "training" }];
  browser.runStatus = { state: "finished" };
  browser.selectionFence = "fence-a";
  browser.nextCursor = "page-two";
  assert.equal(browser.rememberCheckpointCatalog(), true);

  browser.route = { ...browser.route, checkpoint_id: "checkpoint-1" };
  browser.applyRoute({ checkpoint_id: "" });
  await new Promise(setImmediate);
  assert.deepEqual(browser.items, [{ checkpoint_id: "checkpoint-1", step: 100, metrics: { "train/return": 3 } }]);
  assert.deepEqual(browser.metricColumns, [{ metric: "train/return", evidence: "training" }]);
  assert.deepEqual(browser.runStatus, { state: "finished" });
  assert.equal(browser.selectionFence, "fence-a");
  assert.equal(browser.nextCursor, "page-two");
  assert.equal(browser.loadedKey, browser.routeKey());
  assert.equal(requests.length, 0);

  const newSession = createBrowser();
  newSession.route = { ...browser.route };
  assert.equal(newSession.restoreCheckpointCatalog(), false);
  browser.route = { ...browser.route, run_id: "run-b" };
  assert.equal(browser.restoreCheckpointCatalog(), false);
  browser.route = { ...browser.route, run_id: "run-a", goal_variant_id: "variant-b" };
  assert.equal(browser.restoreCheckpointCatalog(), false);
});

test("returning to a checkpoint table resumes pending evidence without reloading its rows", async (context) => {
  const originalLocation = globalThis.location;
  globalThis.location = { pathname: "/embedded-player", search: "", hash: "" };
  context.after(() => {
    if (originalLocation === undefined) delete globalThis.location;
    else globalThis.location = originalLocation;
  });
  const browser = new SourceBrowser({}, { replaceChildren() {}, hidden: false }, {
    token: "token", command() {}, getState: () => ({ hasControl: true }), showToast() {},
  });
  browser.route = { ...browser.route, level: "runs", run_id: "run-a" };
  browser.sourceItems = [{ checkpoint_id: "checkpoint-1", step: 100 }];
  browser.checkpointTrainingPending = true;
  browser.rememberCheckpointCatalog();
  browser.route = { ...browser.route, checkpoint_id: "checkpoint-1" };
  browser.renderView = () => {};
  const resumed = [];
  browser.loadCheckpointTraining = (key) => {
    browser.checkpointTrainingController = {};
    resumed.push(key);
  };

  browser.applyRoute({ checkpoint_id: "" });
  browser.restoreCheckpointCatalog(); // The app can render the same route after navigation.
  await new Promise(setImmediate);

  assert.deepEqual(browser.items, [{ checkpoint_id: "checkpoint-1", step: 100 }]);
  assert.deepEqual(resumed, [browser.routeKey()]);
  assert.equal(browser.loadedKey, browser.routeKey());
});

test("refreshing cached goals keeps old rows until the new catalog arrives", async (context) => {
  const originalLocation = globalThis.location;
  const originalFetch = globalThis.fetch;
  const requests = [];
  globalThis.location = { pathname: "/embedded-player", search: "", hash: "" };
  globalThis.fetch = (url, options) => new Promise((resolve) => {
    requests.push({ url, options, resolve });
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
  sourceBrowser.route = {
    ...sourceBrowser.route,
    level: "goals",
    environment_id: "Mario",
  };
  sourceBrowser.sourceItems = [{ goal_id: "Old", recipe_count: 1 }];
  sourceBrowser.items = [...sourceBrowser.sourceItems];
  sourceBrowser.renderView = () => {};

  const refresh = sourceBrowser.load({ force: true, quiet: true });
  assert.deepEqual(sourceBrowser.items, [{ goal_id: "Old", recipe_count: 1 }]);
  assert.equal(requests.length, 1);
  assert.match(requests[0].url, /\/environments\/Mario\/goals\?refresh=1/);

  requests[0].resolve({
    ok: true,
    json: async () => ({
      items: [{ goal_id: "New", recipe_count: 2 }],
      next_cursor: null,
    }),
  });
  await refresh;

  assert.deepEqual(sourceBrowser.items, [{ goal_id: "New", recipe_count: 2 }]);
  assert.deepEqual(
    sourceBrowser.goalCatalogCache.get("Mario").sourceItems,
    [{ goal_id: "New", recipe_count: 2 }],
  );
});

test("source discovery omits the redundant continue-watching banner", async () => {
  const source = await readFile(
    new URL("../../src/gradlab/web_player/sources/browser.js", import.meta.url),
    "utf8",
  );
  const styles = await readFile(
    new URL("../../src/gradlab/web_player/styles.css", import.meta.url),
    "utf8",
  );
  assert.doesNotMatch(source, /Continue current playback/);
  assert.doesNotMatch(source, /Continue watching/);
  assert.doesNotMatch(source, /renderContinuePlayback/);
  assert.doesNotMatch(styles, /continue-playback/);
});

test("run metrics use compact labels and values", () => {
  assert.equal(metricLabel("leader/step"), "Checkpoint step");
  assert.equal(metricLabel(METRIC), "Mean return");
  assert.equal(
    metricLabel("train/success/min"),
    "Recent all-start success rate min",
  );
  assert.equal(
    metricLabel("train/success/mean"),
    "Recent all-start success rate mean",
  );
  assert.equal(
    metricLabel("train/return/mean"),
    "Recent target return mean",
  );
  assert.equal(
    metricLabel("train/progress/kills/mean"),
    "Recent target kills mean",
  );
  assert.equal(
    metricLabel("train/progress/bricks_destroyed/max"),
    "Recent target bricks destroyed max",
  );
  assert.equal(
    formatMetricValue("eval/success/min", 0.875),
    "87.5%",
  );
  assert.equal(formatMetricValue(METRIC, null), "—");
});

test("checkpoint metric headers omit the prefix shown by their group", () => {
  assert.equal(checkpointMetricHeaderLabel({
    metric: "eval/success/min",
    evidence: "evaluation",
  }), "success/min");
  assert.equal(checkpointMetricHeaderLabel({
    metric: "train/success/min",
    evidence: "training",
  }), "success/min");
  assert.equal(checkpointMetricHeaderLabel({
    metric: "eval/return/mean",
    evidence: "evaluation",
  }), "return/mean");
  assert.equal(checkpointMetricHeaderLabel({
    metric: "train/progress/bricks_destroyed/max",
    evidence: "training",
  }), "progress/bricks_destroyed/max");
  assert.equal(checkpointMetricHeaderLabel({
    metric: "custom/score",
    evidence: "training",
  }), "custom/score");
});

test("checkpoint evaluation status distinguishes missing, running, and verified evidence", () => {
  assert.deepEqual(checkpointEvaluationPresentation({}), {
    label: "Not evaluated", tone: "absent",
  });
  assert.deepEqual(checkpointEvaluationPresentation({ evaluation_queue: { state: "running", evaluation: {
    episodes_completed: 43, episodes_planned: 100,
  } } }), { label: "Running · 43/100", tone: "running" });
  assert.deepEqual(checkpointEvaluationPresentation({ evaluation: {
    source: "monitoring", status: "queued", episodes_planned: 100,
  } }), { label: "Queued", tone: "pending", title: "Checkpoint Monitoring · observational" });
  assert.deepEqual(checkpointEvaluationPresentation({ evaluation: {
    source: "monitoring", status: "running", episodes_planned: 100,
  } }), { label: "Running", tone: "running", title: "Checkpoint Monitoring · observational" });
  assert.deepEqual(checkpointEvaluationPresentation({ evaluation: {
    source: "monitoring", status: "finalizing", episodes_completed: 100, episodes_planned: 100,
  } }), { label: "Finalizing · 100/100", tone: "pending", title: "Checkpoint Monitoring · observational" });
  assert.deepEqual(checkpointEvaluationPresentation({ evaluation: {
    source: "monitoring", status: "verified", episodes_completed: 100, episodes_planned: 100,
  } }), {
    label: "FINISHED", progress: "100/100", tone: "verified", title: "Checkpoint Monitoring · observational",
  });
  assert.deepEqual(checkpointEvaluationPresentation({ evaluation: {
    status: "verified", episodes_completed: 100, episodes_planned: 100,
  } }), { label: "FINISHED", progress: "100/100", tone: "verified" });
  for (const status of ["accepted", "rejected"]) {
    assert.deepEqual(checkpointEvaluationPresentation({ evaluation: {
      status, episodes_completed: status === "accepted" ? 100 : 1, episodes_planned: 100,
    } }), {
      label: "FINISHED", progress: status === "accepted" ? "100/100" : "1/100",
      tone: "verified", title: "Acceptance evaluation",
    });
  }
  assert.deepEqual(checkpointEvaluationPresentation({
    evaluation_queue: { state: "rejected" },
  }), {
    label: "FINISHED", progress: "", tone: "verified", title: "Acceptance evaluation",
  });
  for (const state of ["failed", "blocked", "expired", "canceled"]) {
    assert.deepEqual(checkpointEvaluationPresentation({ evaluation_queue: { state } }), {
      label: state[0].toUpperCase() + state.slice(1), tone: "failed",
    });
  }
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
        metric: "train/return/mean",
        value: 5.25,
        condition: {
          metric: "train/return/mean",
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

test("checkpoint metric cells show loading until their values resolve", async () => {
  const trainSuccess = "train/success/mean";
  const evalReturn = "eval/return/mean";
  const trainColumn = { metric: trainSuccess, evidence: "training" };
  const evalColumn = { metric: evalReturn, evidence: "evaluation" };
  const loading = { training_pending: true, training_loaded_metrics: [], metrics: {} };
  assert.equal(checkpointMetricIsLoading(loading, trainColumn), true);
  assert.equal(checkpointMetricIsLoading(loading, evalColumn), true);
  assert.equal(checkpointMetricIsLoading({ ...loading, metrics: { [evalReturn]: 1 } }, evalColumn), false);
  assert.equal(checkpointMetricIsLoading({ ...loading, training_loaded_metrics: [trainSuccess] }, trainColumn), false);
  assert.equal(checkpointMetricIsLoading({ ...loading, training_pending: false }, evalColumn), false);

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
  assert.match(checkpointMetricDescription(objective), /Frozen checkpoint-evaluation evidence/);
  assert.match(checkpointMetricDescription(proxy), /Diagnostic online training proxy/);

  const html = await sourceHtml({items: [{...checkpoint, training_pending: true, training_loaded_metrics: []}], metricColumns: [{metric: trainSuccess, label: trainSuccess, evidence: "training"}, {metric: evalReturn, label: evalReturn, evidence: "evaluation"}]});
  assert.match(html, /Loading train\/success\/mean/);
  assert.match(html, /Loading eval\/return\/mean/);
  assert.match(html, /checkpoint-eval-metric-cell/);
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

test("run and checkpoint routes preserve goal variant identity", () => {
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
  assert.equal(
    sourceRouteFromPath(`/environments/Mario/goals/Level1-1/variants/${variant}`),
    null,
  );
  assert.equal(sourceRoutePath({
    level: "goal_variants",
    environment_id: "Mario",
    goal_id: "Level1-1",
    goal_variant_id: variant,
    run_id: "",
    checkpoint_id: "",
  }), "/environments/Mario/goals/Level1-1");
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
    { metric: "leader/step", direction: "min" },
    { metric: METRIC, direction: "max" },
  ];
  const fallback = [
    {
      metric: "train/success/min",
      direction: "max",
    },
    { metric: "train/step", direction: "min" },
  ];
  const items = [
    {
      run_id: "training-only",
      recipe: "fast-training",
      metrics: {
        "train/success/min": 1,
        "train/step": 100,
      },
    },
    {
      run_id: "later-checkpoint",
      recipe: "high-return",
      metrics: {
        "leader/step": 2_000,
        [METRIC]: 500,
      },
    },
    {
      run_id: "earlier-checkpoint",
      recipe: "sample-efficient",
      metrics: {
        "leader/step": 1_000,
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
    { metric: "leader/step", direction: "min" },
    { metric: METRIC, direction: "max" },
  ];
  const fallback = [
    {
      metric: "train/success/min",
      direction: "max",
    },
    { metric: "train/step", direction: "min" },
  ];
  const items = [
    {
      run_id: "slower",
      metrics: {
        "train/success/min": 0.9,
        "train/step": 2_000,
      },
    },
    {
      run_id: "faster",
      metrics: {
        "train/success/min": 0.9,
        "train/step": 1_000,
      },
    },
  ];

  assert.equal(activeRunMetricColumns(items, primary, fallback), fallback);
  const leader = bestRunEfficiency(items, primary, fallback);
  assert.equal(leader.evidence, "training");
  assert.equal(leader.item.run_id, "faster");
});

test("goal evidence fills pending rows and ignores obsolete responses", async (context) => {
  const originalFetch = globalThis.fetch;
  context.after(() => { globalThis.fetch = originalFetch; });
  let finish;
  globalThis.fetch = () => new Promise((resolve) => { finish = resolve; });
  const browser = Object.create(SourceBrowser.prototype);
  Object.assign(browser, {
    token: "test", catalogRequestTimeoutMs: 1000, goalEvidenceEpoch: 1,
    sourceItems: [{ goal_id: "goal", recipe_count: 8, evidence_status: "pending" }],
    routeKey: () => "goals", endpoint: () => "/goals?evidence=0",
    rememberGoalCatalog() {}, renderView() {},
  });
  const loading = browser.loadGoalEvidence("goals", 1, null);
  assert.equal(environmentSuccessStatus(browser.sourceItems[0], "train/success").label, "Loading…");
  finish({ ok: true, json: async () => ({ items: [{ goal_id: "goal", run_count: 1, success_badges: ["train/success"] }] }) });
  await loading;
  assert.equal(browser.items[0].recipe_count, 8);
  assert.equal(environmentSuccessStatus(browser.items[0], "train/success").label, "✅");

  const obsolete = browser.loadGoalEvidence("goals", 1, null);
  browser.goalEvidenceEpoch = 2;
  finish({ ok: true, json: async () => ({ items: [{ goal_id: "goal", run_count: 0, success_badges: [] }] }) });
  await obsolete;
  assert.equal(browser.items[0].run_count, 1);

  browser.sourceItems = [{ goal_id: "goal", evidence_status: "pending" }];
  globalThis.fetch = async () => { throw new Error("offline"); };
  await browser.loadGoalEvidence("goals", 2, null);
  assert.equal(environmentSuccessStatus(browser.items[0], "eval/success").label, "Unavailable");
});


test("embedded runs use the fetched cursor after exhausting older runs", () => {
  const browser = Object.create(SourceBrowser.prototype);
  browser.goalVariantRunPages = new Map();
  const variant = { variant_id: "variant", has_more_runs: true };
  assert.equal(browser.embeddedGoalRunsHaveMore(variant), true);
  browser.goalVariantRunPages.set("variant", { loading: true });
  assert.equal(browser.embeddedGoalRunsHaveMore(variant), true);
  browser.goalVariantRunPages.set("variant", { loaded: true, nextCursor: "next" });
  assert.equal(browser.embeddedGoalRunsHaveMore(variant), true);
  browser.goalVariantRunPages.set("variant", { loaded: true, nextCursor: null });
  assert.equal(browser.embeddedGoalRunsHaveMore(variant), false);
  browser.goalVariantRunPages.clear();
  assert.equal(browser.embeddedGoalRunsHaveMore({ ...variant, has_more_runs: false }), false);
});


test("checkpoint evidence renders partial cells before completion and clears loading", async (t) => {
  let stream;
  t.mock.method(globalThis, "fetch", async () => new Response(new ReadableStream({
    start(controller) { stream = controller; },
  })));
  const browser = Object.create(SourceBrowser.prototype);
  Object.assign(browser, {
    route: { run_id: "gradlab-test" }, routeKey: () => "run", query: "", token: "test",
    checkpointTrainingSerial: 0, catalogRequestTimeoutMs: 1000,
    sourceItems: [{ checkpoint_id: "top", metrics: {} }, { checkpoint_id: "bottom", metrics: {} }],
    catalogWarnings: [], renderView() {},
  });
  const loading = browser.loadCheckpointTraining("run");
  assert.ok(browser.checkpointTrainingController);
  assert.ok(browser.sourceItems.every((item) => item.training_pending));
  const encode = (event) => new TextEncoder().encode(JSON.stringify(event) + "\n");
  const update = encode({ type: "metrics", items: [{ checkpoint_id: "top", metrics: { bricks: 42 } }] });
  // The network may split a JSON record across arbitrary chunks.
  stream.enqueue(update.slice(0, 12));
  stream.enqueue(update.slice(12));
  await new Promise(setImmediate);
  assert.equal(browser.items[0].metrics.bricks, 42);
  assert.deepEqual(browser.items[0].training_loaded_metrics, ["bricks"]);
  assert.deepEqual(browser.items[1].training_loaded_metrics, []);
  assert.ok(browser.checkpointTrainingController);
  stream.enqueue(encode({ type: "complete", items: browser.items, warnings: [] }));
  stream.close();
  await loading;
  assert.equal(browser.checkpointTrainingController, null);
  assert.ok(browser.items.every((item) => !item.training_pending));
});

test("interrupted checkpoint evidence keeps received values and releases refresh", async (t) => {
  const events = [{ type: "metrics", items: [{ checkpoint_id: "top", metrics: { bricks: 7 } }] }];
  t.mock.method(globalThis, "fetch", async () => new Response(events.map(JSON.stringify).join("\n") + "\n"));
  const browser = Object.create(SourceBrowser.prototype);
  Object.assign(browser, {
    route: { run_id: "gradlab-test" }, routeKey: () => "run", query: "", token: "test",
    checkpointTrainingSerial: 0, catalogRequestTimeoutMs: 1000,
    sourceItems: [{ checkpoint_id: "top", metrics: {} }], catalogWarnings: [], renderView() {},
  });
  await browser.loadCheckpointTraining("run");
  assert.equal(browser.items[0].metrics.bricks, 7);
  assert.equal(browser.items[0].training_pending, false);
  assert.equal(browser.checkpointTrainingController, null);
  assert.equal(browser.freshness, "partial");
  assert.match(browser.catalogWarnings[0].message, /ended before completion/);
});

test("goal activity restores its complete table and expanded run pages", () => {
  const view = Object.create(SourceBrowser.prototype);
  Object.assign(view, {
    route: { level: "goal_variants", environment_id: "env", goal_id: "goal" },
    query: "", goalActivityCache: new Map(), loadedKey: "",
    sourceItems: [{ variant_id: "current", recent_runs: [] }], activityRevision: "revision",
    metricColumns: [], fallbackMetricColumns: [], nextCursor: "next", freshness: "fresh",
    catalogWarnings: [], catalogSource: null, generatedAt: null, selectionFence: "fence",
    goalVariantRunPages: new Map([["current", { loaded: true, items: [{ run_id: "run-a" }], nextCursor: null }]]),
  });
  view.rememberGoalActivity();
  view.sourceItems = [];
  view.items = [];
  view.goalVariantRunPages.clear();
  assert.equal(view.restoreGoalActivity(), true);
  assert.equal(view.items[0].variant_id, "current");
  assert.equal(view.loadedKey, view.routeKey());
  assert.equal(view.freshness, "fresh");
  assert.equal(view.nextCursor, "next");
  assert.deepEqual(view.goalVariantRunPages.get("current").items, [{ run_id: "run-a" }]);
  view.route.goal_id = "different";
  assert.equal(view.restoreGoalActivity(), false);
});

test("goal configuration loading renders both columns before data arrives", async () => {
  const html = await sourceHtml({route: {level: "goal_variants"}, loading: true, loadedKey: ""});
  assert.match(html, /aria-busy="true"/);
  assert.match(html, /goal-configuration-list/);
  assert.match(html, /goal-configuration-panel/);
  assert.equal((html.match(/Loading goal configuration/g) || []).length, 16);
  assert.equal((html.match(/Loading configuration details/g) || []).length, 3);
});
