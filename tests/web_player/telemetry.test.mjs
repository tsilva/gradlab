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
  lineLegendPrefix,
  lineLegendPresentation,
  lineLegendPresentationAtIndex,
  lineBlockFootPresentation,
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

test("policy distributions omit the redundant action summary caption", () => {
  assert.doesNotMatch(
    telemetryPanelSource,
    /executed actions? in the retained episode|setActionComparisonCaption/,
  );
});

test("namespace telemetry tables use the panel as their only scroll container", () => {
  assert.match(telemetryPanelSource, /table\.className = "telemetry-namespace-table";/);
  const rule = styles.match(/\.telemetry-namespace-table \{([^}]*)\}/)?.[1] || "";
  assert.match(rule, /margin-top: \.55rem;/);
  assert.doesNotMatch(rule, /(?:max-height|overflow)\s*:/);
});

test("long action labels truncate on one line and retain their full tooltip", () => {
  const rule = styles.match(/\.action-comparison-label \{([^}]*)\}/)?.[1] || "";
  assert.match(rule, /overflow: hidden;/);
  assert.match(rule, /text-overflow: ellipsis;/);
  assert.match(rule, /white-space: nowrap;/);
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

test("value comparison foot explains returns and finite-horizon aliasing", () => {
  const presentation = lineBlockFootPresentation(
    {
      metrics: ["policy/value", "policy/realized-return"],
    },
    null,
    {
      session: {
        termination_conditions: [
          {
            id: "limit:max_episode_steps",
            enabled: true,
            value: 512,
          },
        ],
      },
    },
  );

  assert.match(presentation.text, /not its success flag or cumulative episode return/);
  assert.match(presentation.text, /Finite horizon: 512 policy steps/);
  assert.match(presentation.text, /remaining time is absent from the policy observation/);
  assert.equal(presentation.warning, false);
});
