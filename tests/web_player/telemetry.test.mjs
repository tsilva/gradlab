import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  compatibleMetricKeys,
  decodeDynamicSegment,
  descriptorCatalog,
  descriptorAvailability,
  descriptorFor,
  descriptorValue,
  dynamicDescriptorKey,
  encodeDynamicSegment,
  metricOptions,
  seriesForMetric,
} from "../../src/gradlab/web_player/panels/telemetry.js";
import {
  actionComparisonPresentation,
  distributionBlockTitle,
  cursorIndex,
  distributionBlockVisible,
  histogramSelectedLabel,
  lineBlockAvailability,
  lineLegendPrefix,
  lineLegendPresentation,
  lineLegendPresentationAtIndex,
  lineCursorSequence,
  lineBlockFootPresentation,
  ordinal,
  policyDecisionLayoutEnabled,
  policyDecisionPresentation,
  policyDecisionRank,
  rewardSummaryCards,
  selectedPoint,
  statsBlockFoot,
} from "../../src/gradlab/web_player/panels/telemetry-panel.js";
import {
  displayedEpisode,
  displayedStep,
  lineCursorIndex,
  lineCursorX,
  timelineLabel,
} from "../../src/gradlab/web_player/panels/shared.js";
import {
  discreteActionLabels,
  formatActionValue,
  scalarActionIndex,
} from "../../src/gradlab/web_player/panels/action-contract.js";

const styles = readFileSync(
  new URL("../../src/gradlab/web_player/styles.css", import.meta.url),
  "utf8",
);
const telemetryPanelSource = readFileSync(
  new URL("../../src/gradlab/web_player/panels/telemetry-panel.js", import.meta.url),
  "utf8",
);

test("signal selectors leave focus-safe space before the chart", () => {
  assert.match(
    styles,
    /\.signal-toolbar \{[^}]*margin: \.3rem \.3rem \.5rem;/,
  );
});

test("distribution contents use the panel as their only scroll container", () => {
  const rule = styles.match(/\.action-probabilities \{([^}]*)\}/)?.[1] || "";
  assert.doesNotMatch(rule, /(?:max-height|overflow)\s*:/);
  const comparisonRule = styles.match(/\.action-comparison \{([^}]*)\}/)?.[1] || "";
  assert.doesNotMatch(comparisonRule, /(?:max-height|overflow)\s*:/);
});

test("policy distribution legend labels align with their data columns", () => {
  const layoutRule = styles.match(/\.action-comparison-layout \{([^}]*)\}/)?.[1] || "";
  const legendRule = styles.match(/\.action-comparison-legend \{([^}]*)\}/)?.[1] || "";
  const legendSeriesRule = styles.match(
    /\.action-comparison-legend-series \{([^}]*)\}/,
  )?.[1] || "";
  const comparisonRule = styles.match(/\.action-comparison \{([^}]*)\}/)?.[1] || "";
  const rowRule = styles.match(/\.action-comparison-row \{([^}]*)\}/)?.[1] || "";
  const barsRule = styles.match(/\.action-comparison-bars \{([^}]*)\}/)?.[1] || "";

  assert.match(layoutRule, /grid-template-columns: max-content minmax\(0, 1fr\);/);
  assert.match(legendRule, /grid-template-columns: subgrid;/);
  assert.match(comparisonRule, /grid-template-columns: subgrid;/);
  assert.match(rowRule, /grid-template-columns: subgrid;/);
  assert.match(telemetryPanelSource, /layout\.append\(legend, target\);/);
  assert.match(legendSeriesRule, /grid-column: 2;/);
  assert.match(
    legendSeriesRule,
    /grid-template-columns: minmax\(0, 1fr\);/,
  );
  assert.match(barsRule, /grid-template-columns: minmax\(0, 1fr\);/);
  assert.match(barsRule, /gap: 0;/);
  assert.match(
    styles,
    /\.action-comparison-track \{[^}]*border-radius: 0;/,
  );
  assert.match(
    styles,
    /\.action-comparison-bar\.step \.action-comparison-track \{\s*align-self: end;/,
  );
  assert.match(
    styles,
    /\.action-comparison-bar\.episode \.action-comparison-track \{\s*align-self: start;/,
  );
  assert.doesNotMatch(
    styles,
    /\.action-comparison-bar\.(?:step|episode) \.action-comparison-track \{[^}]*border-radius:/,
  );
  assert.match(telemetryPanelSource, /class="action-comparison-legend-series"/);
  assert.match(
    telemetryPanelSource,
    /<span class="step">Step action probability<\/span>\s*<span class="episode">Episode action frequency<\/span>/,
  );
  assert.match(
    telemetryPanelSource,
    /actionComparisonBar\(row\.name, "step", row\.stepProbability\),\s*actionComparisonBar\(row\.name, "episode", row\.episodeProbability\)/,
  );
});

test("policy distributions omit the redundant action summary caption", () => {
  assert.doesNotMatch(
    telemetryPanelSource,
    /executed actions? in the retained episode|setActionComparisonCaption/,
  );
});

test("the built-in policy panel uses the decision-first layout only for its canonical blocks", () => {
  const definition = {
    id: "policy",
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
  };

  assert.equal(policyDecisionLayoutEnabled(definition), true);
  assert.equal(policyDecisionLayoutEnabled({ ...definition, id: "custom" }), false);
  assert.equal(policyDecisionLayoutEnabled({
    ...definition,
    config: { blocks: definition.config.blocks.slice(1) },
  }), false);
});

test("policy decision presentation keeps selection, probability, and episode frequency distinct", () => {
  const entries = ["noop", "button", "right", "left"].map((label, value) => ({
    value,
    semantic_id: label,
    label,
  }));
  const snapshot = {
    policy: {
      algorithm_id: "ppo",
      introspection: [
        "actor_distribution",
        "state_value",
        "entropy",
        "selected_action_log_probability",
      ],
    },
    transition: {
      executed_action: 3,
      decision: {
        action_selection_mode: "stochastic",
        selected_action: 3,
        probabilities: [0.041, 0.165, 0.014, 0.78],
        value: 3.7437,
        entropy: 0.6828,
        log_probability: -0.2486,
      },
    },
    session: {
      action_contract: {
        policy: {
          space: { type: "discrete", n: 4, start: 0 },
          semantics: { status: "available", encoding: "explicit", entries },
        },
      },
    },
  };
  const history = [
    ...Array.from({ length: 10 }, () => ({ executed_action: 0 })),
    ...Array.from({ length: 26 }, () => ({ executed_action: 1 })),
    ...Array.from({ length: 22 }, () => ({ executed_action: 2 })),
    ...Array.from({ length: 42 }, () => ({ executed_action: 3 })),
  ];

  const presentation = policyDecisionPresentation(snapshot, history, {});

  assert.equal(presentation.discrete, true);
  assert.equal(presentation.action, "left");
  assert.equal(presentation.mode, "stochastic");
  assert.equal(presentation.rank, 1);
  assert.equal(presentation.choiceCount, 4);
  assert.equal(presentation.stepProbability, 0.78);
  assert.equal(presentation.selectedIsHighest, true);
  assert.deepEqual(
    presentation.rows.map(({ selected, highest }) => ({ selected, highest })),
    [
      { selected: false, highest: false },
      { selected: false, highest: false },
      { selected: false, highest: false },
      { selected: true, highest: true },
    ],
  );
  assert.deepEqual(
    presentation.rows.map(({ episodeProbability }) => episodeProbability),
    [0.1, 0.26, 0.22, 0.42],
  );
  assert.deepEqual(
    presentation.stats.map(({ label, value }) => [label, value]),
    [["V(s)", "3.7437"], ["Entropy", "0.6828"], ["Log p", "-0.2486"]],
  );
});

test("policy decision table omits the redundant standalone series legend", () => {
  assert.doesNotMatch(telemetryPanelSource, /policy-decision-legend/);
  assert.doesNotMatch(styles, /\.policy-decision-legend/);
  assert.match(
    telemetryPanelSource,
    /\["This step", "step"\],\s*\["Episode frequency", "episode"\]/,
  );
});

test("policy decision rank follows the action-selection mode", () => {
  assert.match(
    telemetryPanelSource,
    /modeLine\.append\(mode, rank\);\s*hero\.append\(heroLine, modeLine\);/,
  );
  assert.match(
    styles,
    /\.policy-decision-mode-line \{[^}]*display: flex;[^}]*gap: \.6rem;/,
  );
  assert.doesNotMatch(
    styles.match(/\.policy-decision-rank \{([^}]*)\}/)?.[1] || "",
    /margin-left: auto;/,
  );
});

test("policy decision hero omits the redundant step probability label", () => {
  assert.doesNotMatch(telemetryPanelSource, /policy-decision-probability-label/);
  assert.doesNotMatch(telemetryPanelSource, /probabilityLabel/);
  assert.doesNotMatch(styles, /\.policy-decision-probability-label/);
  assert.match(telemetryPanelSource, /heroLine\.append\(action, probability\);/);
});

test("policy decision rank orders current-step choices and preserves ties", () => {
  const rows = [
    { name: "first", stepProbability: 0.5 },
    { name: "tied", stepProbability: 0.25 },
    { name: "selected", stepProbability: 0.25 },
    { name: "last", stepProbability: 0 },
  ];

  assert.equal(policyDecisionRank(rows, rows[2]), 2);
  assert.equal(policyDecisionRank(rows, { stepProbability: null }), null);
  assert.equal(ordinal(1), "1st");
  assert.equal(ordinal(2), "2nd");
  assert.equal(ordinal(3), "3rd");
  assert.equal(ordinal(11), "11th");
  assert.equal(ordinal(23), "23rd");
});

test("policy decision colors selected and highest-probability actions independently", () => {
  const snapshot = {
    policy: {
      introspection: ["actor_distribution"],
    },
    transition: {
      decision: {
        action_selection_mode: "stochastic",
        selected_action: 2,
        probabilities: [0.1, 0.7, 0.2],
      },
    },
    session: {
      action_contract: {
        policy: {
          space: { type: "discrete", n: 3, start: 0 },
        },
      },
    },
  };

  const presentation = policyDecisionPresentation(snapshot, [], {});

  assert.equal(presentation.selectedIsHighest, false);
  assert.deepEqual(
    presentation.rows.map(({ selected, highest }) => ({ selected, highest })),
    [
      { selected: false, highest: false },
      { selected: false, highest: true },
      { selected: true, highest: false },
    ],
  );
});

test("policy decision marks every tied probability maximum", () => {
  const snapshot = {
    policy: {
      introspection: ["actor_distribution"],
    },
    transition: {
      decision: {
        action_selection_mode: "stochastic",
        selected_action: 1,
        probabilities: [0.5, 0.5, 0],
      },
    },
    session: {
      action_contract: {
        policy: {
          space: { type: "discrete", n: 3, start: 0 },
        },
      },
    },
  };

  const presentation = policyDecisionPresentation(snapshot, [], {});

  assert.equal(presentation.selectedIsHighest, true);
  assert.deepEqual(
    presentation.rows.map(({ selected, highest }) => ({ selected, highest })),
    [
      { selected: false, highest: true },
      { selected: true, highest: true },
      { selected: false, highest: false },
    ],
  );
});

test("policy decision does not fabricate a maximum for invalid probabilities", () => {
  const snapshot = {
    transition: { executed_action: 0 },
    session: {
      action_contract: {
        policy: {
          space: { type: "discrete", n: 3, start: 0 },
        },
      },
    },
  };
  const presentation = actionComparisonPresentation(snapshot, [], {
    selected_action: 0,
    probabilities: [0.5, 1.5, 0.5],
  });

  assert.equal(presentation.step.status, "protocol-error");
  assert.deepEqual(presentation.rows.map(({ highest }) => highest), [null, null, null]);
});

test("policy decision color system renders separate selected and highest stripes", () => {
  assert.match(
    telemetryPanelSource,
    /row\.highest \? "highest" : ""/,
  );
  assert.match(
    telemetryPanelSource,
    /target\.classList\.toggle\(\s*"selected-is-highest"/,
  );
  assert.match(
    telemetryPanelSource,
    /target\.classList\.toggle\(\s*"selected-below-highest"/,
  );
  assert.match(
    styles,
    /\.policy-decision-comparison-row\.selected\.highest \{[^}]*inset 3px 0 var\(--color-evaluation-text\)[^}]*inset 6px 0 var\(--color-series-amber\)/,
  );
});

test("reward totals show pre-clip and post-clip values without a formula", () => {
  assert.doesNotMatch(telemetryPanelSource, /block\.title \|\| "Reward ledger"/);
  assert.doesNotMatch(
    telemetryPanelSource,
    /Signed contribution uses \|final reward\|/,
  );
  assert.match(telemetryPanelSource, /const foot = appendFoot\(section, block\.foot\);/);
  assert.match(telemetryPanelSource, /foot\?\.classList\.toggle\(/);
  assert.match(telemetryPanelSource, /classList\.toggle\("titleless", !block\.title\)/);
  assert.match(styles, /\.reward-analysis-toolbar\.titleless \{ justify-content: flex-end; \}/);
  assert.doesNotMatch(telemetryPanelSource, /reward-transform-strip/);
  assert.doesNotMatch(telemetryPanelSource, /reward-ledger-summary-detail/);
  assert.deepEqual(
    rewardSummaryCards({
      positive: 4,
      negative: -2,
      preclip: 2,
      final: 2,
    }),
    [
      ["Bonuses", 4, "positive"],
      ["Penalties", -2, "negative"],
      ["Pre-clip", 2, "preclip"],
      ["Post-clip", 2, "postclip"],
    ],
  );
});

test("namespace telemetry tables use the panel as their only scroll container", () => {
  assert.match(telemetryPanelSource, /table\.className = "telemetry-namespace-table";/);
  const rule = styles.match(/\.telemetry-namespace-table \{([^}]*)\}/)?.[1] || "";
  assert.match(rule, /margin-top: \.55rem;/);
  assert.doesNotMatch(rule, /(?:max-height|overflow)\s*:/);
});

test("action labels fit one content-sized column and retain their full tooltip", () => {
  const rule = styles.match(/\.action-comparison-label \{([^}]*)\}/)?.[1] || "";
  const policyRule = styles.match(/\.policy-decision-action-label \{([^}]*)\}/)?.[1] || "";
  assert.match(rule, /margin-left: \.3rem;/);
  assert.match(policyRule, /margin-left: \.3rem;/);
  assert.match(rule, /white-space: nowrap;/);
  assert.doesNotMatch(rule, /overflow: hidden;/);
  assert.doesNotMatch(rule, /text-overflow: ellipsis;/);
  assert.doesNotMatch(rule, /overflow-wrap/);
  assert.match(telemetryPanelSource, /label\.title = row\.name;/);
});

test("dynamic metric names round-trip without path ambiguity", () => {
  const name = "coins / bonus%";
  const encoded = encodeDynamicSegment(name);
  assert.equal(decodeDynamicSegment(encoded), name);
  const key = dynamicDescriptorKey("signal", name);
  const descriptor = descriptorFor(key);
  assert.equal(descriptor.name, name);
  assert.equal(descriptor.namespace, "signal");
});

test("catalog discovers signal and reward-component descriptors", () => {
  const snapshot = {
    transition: {
      signals: { x_pos: 12 },
      reward: { components: { progress: 1.25 } },
    },
  };
  const catalog = descriptorCatalog(snapshot, [
    { signals: { coins: 3 }, components: { terminal: -1 } },
  ]);
  assert.ok(catalog.has(dynamicDescriptorKey("signal", "x_pos")));
  assert.ok(catalog.has(dynamicDescriptorKey("signal", "coins")));
  assert.ok(catalog.has(dynamicDescriptorKey("reward-component", "progress")));
  assert.ok(catalog.has(dynamicDescriptorKey("reward-component", "terminal")));
});

test("action descriptors distinguish policy-selected and executed action", () => {
  const point = { policy_action: 2, executed_action: 1 };
  assert.equal(descriptorFor("action/policy").shortLabel, "Action");
  assert.equal(
    descriptorValue(descriptorFor("action/policy"), { point }),
    2,
  );
  assert.equal(
    descriptorValue(descriptorFor("action/executed"), { point }),
    1,
  );
});

test("action history normalizes vector scalars and never renders undefined selection", () => {
  assert.equal(scalarActionIndex([1]), 1);
  assert.equal(histogramSelectedLabel(["noop", "move left"], null), null);
  assert.equal(histogramSelectedLabel(["noop", "move left"], 1), "move left");
});

test("action comparison aligns episode frequencies with selected-step probabilities", () => {
  const entries = ["noop", "left", "right", "attack"].map((label, value) => ({
    value,
    semantic_id: label,
    label,
  }));
  const snapshot = {
    transition: { executed_action: 3 },
    session: {
      action_contract: {
        policy: {
          space: { type: "discrete", n: 4, start: 0 },
          semantics: { status: "available", encoding: "explicit", entries },
        },
      },
    },
  };
  const history = [
    { executed_action: 0 },
    { executed_action: [1] },
    { executed_action: 1 },
    { executed_action: 3 },
  ];
  const presentation = actionComparisonPresentation(snapshot, history, {
    probabilities: [0.1, 0.6, 0.2, 0.1],
    selected_action: 1,
  });

  assert.equal(presentation.history.status, "available");
  assert.equal(presentation.history.sampleCount, 4);
  assert.deepEqual(
    presentation.rows.map((row) => row.episodeProbability),
    [0.25, 0.5, 0, 0.25],
  );
  assert.deepEqual(
    presentation.rows.map((row) => row.stepProbability),
    [0.1, 0.6, 0.2, 0.1],
  );
  assert.equal(presentation.rows[1].selected, true);
  assert.equal(presentation.rows[3].executed, true);
});

test("legal-tuple action comparison aligns the joint categorical support", () => {
  const legalTuples = [
    [0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
  ];
  const legalEntries = ["noop", "move_forward", "attack"].map(
    (semanticId, index) => ({
      value: legalTuples[index],
      semantic_id: semanticId,
      label: semanticId.replaceAll("_", " "),
    }),
  );
  const snapshot = {
    transition: { executed_action: legalTuples[2] },
    session: {
      action_contract: {
        policy: {
          space: { type: "multi_discrete", legal_tuples: legalTuples },
          semantics: {
            status: "available",
            encoding: "components",
            legal_entries: legalEntries,
          },
        },
      },
    },
  };
  const presentation = actionComparisonPresentation(
    snapshot,
    [
      { executed_action: legalTuples[0] },
      { executed_action: legalTuples[1] },
      { executed_action: legalTuples[1] },
    ],
    { probabilities: [0.1, 0.7, 0.2], selected_action: 1 },
  );

  assert.equal(formatActionValue(legalTuples[1], snapshot), "move forward");
  assert.deepEqual(discreteActionLabels(snapshot, 3), ["noop", "move forward", "attack"]);
  assert.deepEqual(
    presentation.rows.map((row) => row.episodeProbability),
    [1 / 3, 2 / 3, 0],
  );
  assert.equal(presentation.rows[1].selected, true);
  assert.equal(presentation.rows[2].executed, true);
});

test("action comparison keeps the episode aggregate fixed across inspected steps", () => {
  const snapshot = {
    transition: { executed_action: 0 },
    session: {
      action_names: ["noop", "move"],
      action_contract: { policy: { space: { type: "discrete", n: 2, start: 0 } } },
    },
  };
  const history = [{ executed_action: 0 }, { executed_action: 1 }];
  const first = actionComparisonPresentation(snapshot, history, {
    probabilities: [0.8, 0.2],
    selected_action: 0,
  });
  const inspected = actionComparisonPresentation(snapshot, history, {
    probabilities: [0.25, 0.75],
    selected_action: 1,
  });

  assert.deepEqual(
    first.rows.map((row) => row.episodeProbability),
    inspected.rows.map((row) => row.episodeProbability),
  );
  assert.notDeepEqual(
    first.rows.map((row) => row.stepProbability),
    inspected.rows.map((row) => row.stepProbability),
  );
});

test("action comparison exposes empty and contract-incomparable history", () => {
  const snapshot = {
    transition: { executed_action: 1 },
    session: {
      action_names: ["noop", "move"],
      action_contract: { policy: { space: { type: "discrete", n: 2, start: 0 } } },
    },
  };
  const decision = { probabilities: [0.4, 0.6], selected_action: 1 };
  const empty = actionComparisonPresentation(snapshot, [], decision);
  assert.equal(empty.history.status, "not-yet-observed");
  assert.ok(empty.rows.every((row) => row.episodeProbability === null));

  const incomparable = actionComparisonPresentation(
    snapshot,
    [{ executed_action: ["A", "RIGHT"] }],
    decision,
  );
  assert.equal(incomparable.history.status, "contract-incomparable");
  assert.match(incomparable.history.message, /1 of 1 executed actions/);
  assert.ok(incomparable.rows.every((row) => row.episodeProbability === null));
});

test("numeric series preserve gaps and unit compatibility is explicit", () => {
  const series = seriesForMetric("reward/shaped", [
    { reward_shaped: 1 },
    { reward_shaped: null },
    { reward_shaped: -0.5 },
  ]);
  assert.equal(series[0], 1);
  assert.ok(Number.isNaN(series[1]));
  assert.equal(series[2], -0.5);
  const catalog = descriptorCatalog(null, []);
  assert.equal(
    compatibleMetricKeys(["reward/provider", "reward/shaped"], catalog),
    true,
  );
  assert.equal(
    compatibleMetricKeys(["reward/shaped", "reward/return"], catalog),
    false,
  );
  assert.ok(metricOptions(catalog, "histogram").some(
    (descriptor) => descriptor.key === "action/executed",
  ));
});

test("the chart cursor remains on the newest live transition", () => {
  const history = [{ sequence: 4 }, { sequence: 5 }];
  assert.equal(
    cursorIndex(history, {
      inspection: false,
      selectedSequence: 5,
    }),
    1,
  );
});

test("line legends show each series value at the chart cursor", () => {
  const descriptors = [
    descriptorFor("reward/provider"),
    descriptorFor("reward/shaped"),
  ];
  const history = [
    { sequence: 4, reward_provider: 0.25, reward_shaped: 0.5 },
    { sequence: 5, reward_provider: null, reward_shaped: -1 },
  ];
  assert.deepEqual(
    lineLegendPresentation(descriptors, history, { selectedSequence: 4 }),
    [
      { key: "reward/provider", value: "0.250" },
      { key: "reward/shaped", value: "0.500" },
    ],
  );
  assert.deepEqual(
    lineLegendPresentation(descriptors, history, { selectedSequence: 5 }),
    [
      { key: "reward/provider", value: "—" },
      { key: "reward/shaped", value: "-1.000" },
    ],
  );
  assert.deepEqual(
    lineLegendPresentation(descriptors, history, { selectedSequence: null }),
    [
      { key: "reward/provider", value: "—" },
      { key: "reward/shaped", value: "—" },
    ],
  );
});

test("hovered line-chart samples drive every legend value", () => {
  const descriptors = [
    descriptorFor("reward/provider"),
    descriptorFor("reward/shaped"),
  ];
  const history = [
    { reward_provider: 0.25, reward_shaped: 0.5 },
    { reward_provider: 1, reward_shaped: -1 },
  ];
  assert.deepEqual(
    lineLegendPresentationAtIndex(descriptors, history, 0),
    [
      { key: "reward/provider", value: "0.250" },
      { key: "reward/shaped", value: "0.500" },
    ],
  );
});

test("step reward legends use native and shaped reward labels", () => {
  assert.deepEqual(
    ["reward/provider", "reward/shaped"].map((key) => (
      lineLegendPrefix(descriptorFor(key))
    )),
    ["Native R = ", "Shaped R = "],
  );
});

test("value error legends identify the signed critic error", () => {
  assert.equal(
    lineLegendPrefix(descriptorFor("policy/value-error")),
    "V(s) − G(s) = ",
  );
});

test("step zero does not borrow telemetry from a later transition", () => {
  const history = [{ sequence: 1, policy_action: 3 }];
  assert.equal(
    selectedPoint(history, { transition: null }, { selectedSequence: 0 }),
    null,
  );
  assert.equal(
    selectedPoint(history, { transition: null }, {}),
    history[0],
  );
});

test("the initial observation reports the selected sampling mode", () => {
  const snapshot = {
    driver: "policy",
    policy: {
      action_selection: {
        requested_mode: "stochastic",
      },
    },
    session: {
      sampling_mode: "stochastic",
    },
    transition: null,
  };
  assert.equal(
    descriptorValue(descriptorFor("policy/mode"), { snapshot }),
    "stochastic",
  );
});

test("the final chart cursor stays inside the canvas clipping edge", () => {
  const plot = { left: 20, right: 200 };
  assert.equal(lineCursorX(plot, 0, 5), 21);
  assert.equal(lineCursorX(plot, 4, 5), 199);
});

test("line-chart pointer positions select the nearest sample", () => {
  const plot = { left: 20, right: 200 };
  assert.equal(lineCursorIndex(plot, 20, 5), 0);
  assert.equal(lineCursorIndex(plot, 66, 5), 1);
  assert.equal(lineCursorIndex(plot, 110, 5), 2);
  assert.equal(lineCursorIndex(plot, 200, 5), 4);
  assert.equal(lineCursorIndex(plot, 500, 5), 4);
});

test("line-chart pointer positions resolve to retained playback sequences", () => {
  const history = [
    { sequence: 40 },
    { sequence: 44 },
    { sequence: 51 },
  ];
  const plot = { left: 20, right: 200 };
  assert.equal(lineCursorSequence(history, plot, 20, history.length), 40);
  assert.equal(lineCursorSequence(history, plot, 110, history.length), 44);
  assert.equal(lineCursorSequence(history, plot, 200, history.length), 51);
  assert.equal(lineCursorSequence(history, null, 110, history.length), null);
});

test("clicking a line chart inspects its nearest retained playback sequence", () => {
  assert.match(
    telemetryPanelSource,
    /canvas\.addEventListener\("click", \(event\) => \{[\s\S]*?lineCursorSequence\([\s\S]*?services\.inspectSequence\?\.\(sequence\);/,
  );
  assert.match(
    telemetryPanelSource,
    /if \(block\.kind === "line"\) return makeLineBlock\(block, services\);/,
  );
});

test("the timeline shows the displayed episode and step across a boundary", () => {
  assert.equal(
    displayedEpisode({ session: { episode: 3 }, transition: null }),
    3,
  );
  assert.equal(
    displayedEpisode({ session: { episode: 4 }, transition: { episode: 3 } }),
    3,
  );
  assert.equal(
    displayedStep({ session: { step: 0 }, transition: null }),
    0,
  );
  assert.equal(
    displayedStep({ session: { step: 4 }, transition: { step: 4 } }),
    4,
  );
  assert.equal(
    displayedStep({ session: { step: 0 }, transition: { step: 116 } }),
    116,
  );
  assert.equal(
    timelineLabel({
      sequence: 763,
      session: { episode: 4, step: 1228 },
      transition: { episode: 3, step: 763 },
    }),
    "EPISODE 3 · STEP 763",
  );
});

test("policy descriptors distinguish pending and incomparable data", () => {
  const ppoStart = {
    policy: {
      algorithm_id: "ppo",
      introspection: ["state_value"],
    },
    transition: null,
    session: { critic_comparison: { reasons: [] } },
  };
  const pending = descriptorAvailability(
    descriptorFor("policy/value"),
    { snapshot: ppoStart },
  );
  assert.equal(pending.status, "not-yet-observed");
  assert.equal(pending.message, "N/A");

  const incomparable = {
    ...ppoStart,
    transition: { decision: { value: 2 } },
    session: {
      critic_comparison: {
        reasons: ["active policy environment differs from training"],
      },
    },
  };
  assert.equal(
    descriptorAvailability(
      descriptorFor("policy/realized-return"),
      { snapshot: incomparable },
    ).status,
    "contract-incomparable",
  );
});

test("line charts recognize retained post-episode diagnostics", () => {
  const snapshot = {
    policy: {
      algorithm_id: "ppo",
      introspection: ["state_value"],
    },
    transition: {
      boundary: true,
      decision: { value: -14.153 },
    },
    session: { critic_comparison: { reasons: [] } },
  };
  const history = [{
    value: -14.153,
    realized_return: -3.5,
    value_error: -10.653,
  }];
  const descriptors = [
    descriptorFor("policy/value"),
    descriptorFor("policy/realized-return"),
    descriptorFor("policy/value-error"),
  ];

  assert.deepEqual(
    lineBlockAvailability(descriptors, snapshot, history),
    { unavailable: null, status: "available" },
  );
  assert.deepEqual(
    lineLegendPresentationAtIndex(descriptors, history, 0),
    [
      { key: "policy/value", value: "-14.1530" },
      { key: "policy/realized-return", value: "-3.500" },
      { key: "policy/value-error", value: "-10.653" },
    ],
  );

  assert.deepEqual(
    lineBlockAvailability(
      descriptors,
      {
        ...snapshot,
        transition: { boundary: false, decision: { value: -14.153 } },
      },
      [{ value: -14.153 }],
    ),
    { unavailable: null, status: "available" },
  );

  assert.equal(
    lineBlockAvailability(descriptors, snapshot, [{ value: -14.153 }]).status,
    "protocol-error",
  );
});

test("distribution blocks omit diagnostics unsupported by the active policy", () => {
  assert.equal(distributionBlockVisible("unsupported"), false);
  assert.equal(distributionBlockVisible("available"), true);
  assert.equal(distributionBlockVisible("not-yet-observed"), true);
  assert.equal(distributionBlockVisible("protocol-error"), true);
});

test("line charts omit protocol-error labels without hiding contract warnings", () => {
  assert.deepEqual(
    lineBlockFootPresentation({}, {
      status: "protocol-error",
      message: "Protocol error: metric was declared but not supplied.",
    }),
    { text: "", warning: false },
  );
  assert.deepEqual(
    lineBlockFootPresentation({}, {
      status: "contract-incomparable",
      message: "Contract-incomparable: action sampling differs.",
    }),
    {
      text: "Contract-incomparable: action sampling differs.",
      warning: true,
    },
  );
  assert.deepEqual(
    lineBlockFootPresentation({}, null, {
      message: "Truncated episode: G(s) includes the final state's V(s) as a bootstrap.",
    }),
    {
      text: "Truncated episode: G(s) includes the final state's V(s) as a bootstrap.",
      warning: true,
    },
  );
});

test("bootstrapped truncated returns remain available with an explanatory warning", () => {
  const descriptors = [
    descriptorFor("policy/value"),
    descriptorFor("policy/realized-return"),
    descriptorFor("policy/value-error"),
  ];
  const result = lineBlockAvailability(
    descriptors,
    {
      policy: { introspection: ["state_value"] },
      transition: { boundary: true },
      session: { critic_comparison: { reasons: [] } },
    },
    [{
      value: 1.0,
      realized_return: 2.0,
      value_error: -1.0,
      realized_return_bootstrapped: true,
    }],
  );

  assert.equal(result.status, "available");
  assert.equal(result.unavailable, null);
  assert.match(result.notice.message, /final state's V\(s\)/);
});

test("the default policy distribution omits its redundant heading", () => {
  const descriptor = descriptorFor("policy/distribution");
  assert.equal(
    distributionBlockTitle({ metric: "policy/distribution" }, descriptor),
    "",
  );
  assert.equal(
    distributionBlockTitle(
      { metric: "policy/distribution", title: "Action probabilities" },
      descriptor,
    ),
    "Action probabilities",
  );
});

test("stats omit unsupported summaries", () => {
  assert.equal(statsBlockFoot({
    metrics: ["policy/program"],
  }, {}), "");
});

test("ViZDoom actions use the structured runtime contract in every telemetry view", () => {
  const entries = ["noop", "move_left", "move_right", "attack"].map(
    (semanticId, value) => ({
      value,
      semantic_id: semanticId,
      label: semanticId.replaceAll("_", " "),
    }),
  );
  const snapshot = {
    session: {
      action_contract: {
        schema_version: 1,
        policy: {
          space: { type: "discrete", n: 4, start: 0 },
          semantics: {
            status: "available",
            encoding: "explicit",
            entries,
          },
        },
      },
    },
  };

  assert.equal(formatActionValue(1, snapshot), "move left");
  assert.deepEqual(
    discreteActionLabels(snapshot, 4),
    ["noop", "move left", "move right", "attack"],
  );
});

test("missing action semantics are explicit instead of fabricated labels", () => {
  const snapshot = {
    session: {
      action_contract: {
        policy: {
          space: { type: "discrete", n: 2, start: 0 },
          semantics: {
            status: "unavailable",
            reason: "provider did not declare meanings",
          },
        },
      },
    },
  };

  assert.equal(
    formatActionValue(1, snapshot),
    "raw action 1 · semantics unavailable: provider did not declare meanings",
  );
});

test("truncated action entries use the server-resolved names instead of raw indices", () => {
  const snapshot = {
    session: {
      action_names: ["noop", "turn_left", "turn_right", "move_forward"],
      action_contract: {
        policy: {
          space: { type: "discrete", n: 4, start: 0 },
          semantics: {
            status: "available",
            encoding: "explicit",
            entries: [
              { value: "<int>", semantic_id: "<str>", label: "<str>" },
            ],
          },
        },
      },
    },
  };

  assert.equal(formatActionValue(2, snapshot), "turn right");
  assert.deepEqual(
    discreteActionLabels(snapshot, 4),
    ["noop", "turn left", "turn right", "move forward"],
  );
});

test("malformed available action semantics fail visibly when no exact fallback exists", () => {
  const snapshot = {
    session: {
      action_contract: {
        policy: {
          space: { type: "discrete", n: 2, start: 0 },
          semantics: {
            status: "available",
            encoding: "explicit",
            entries: [{ value: "<int>", label: "<str>" }],
          },
        },
      },
    },
  };

  assert.equal(
    formatActionValue(1, snapshot),
    "raw action 1 · semantics unavailable: "
      + "the declared action semantics do not describe this value",
  );
});

test("component action spaces render only complete declared semantics", () => {
  const snapshot = {
    session: {
      action_contract: {
        policy: {
          space: { type: "multi_binary" },
          semantics: {
            status: "available",
            encoding: "components",
            components: [
              { index: 0, semantic_id: "attack", label: "attack" },
              { index: 1, semantic_id: "left", label: "left" },
            ],
          },
        },
      },
    },
  };

  assert.equal(formatActionValue([1, 0], snapshot), "attack");
  assert.match(
    formatActionValue([1], snapshot),
    /declared action semantics do not describe this value/,
  );
});

test("value comparison omits its redundant explanatory label", () => {
  const presentation = lineBlockFootPresentation(
    {
      metrics: ["policy/value", "policy/realized-return"],
    },
    null,
  );

  assert.equal(presentation.text, "");
  assert.equal(presentation.warning, false);
});
