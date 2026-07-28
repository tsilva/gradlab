import assert from "node:assert/strict";
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
  cursorIndex,
  distributionBlockVisible,
  histogramSelectedLabel,
  selectedPoint,
  unsupportedStatLabelsVisible,
} from "../../src/gradlab/web_player/panels/telemetry-panel.js";
import {
  displayedStep,
  lineCursorX,
  timelineLabel,
} from "../../src/gradlab/web_player/panels/shared.js";
import {
  discreteActionLabels,
  formatActionValue,
  scalarActionIndex,
} from "../../src/gradlab/web_player/panels/action-contract.js";

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

test("the timeline shows the displayed transition step across a boundary", () => {
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
      session: { step: 1228 },
      transition: { step: 763 },
    }),
    "STEP 763",
  );
});

test("policy descriptors distinguish unsupported, pending, and incomparable data", () => {
  const dqn = {
    policy: {
      algorithm_id: "dqn",
      introspection: ["action_value"],
    },
    transition: {
      decision: {
        selected_q_value: 1.25,
      },
    },
    session: { critic_comparison: { reasons: [] } },
  };
  assert.equal(
    descriptorAvailability(descriptorFor("policy/value"), { snapshot: dqn }).status,
    "unsupported",
  );
  assert.equal(
    descriptorAvailability(
      descriptorFor("policy/selected-q-value"),
      { snapshot: dqn },
    ).status,
    "available",
  );

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

test("Go-Explore stats omit unsupported critic and probability labels", () => {
  assert.equal(unsupportedStatLabelsVisible({
    policy: {
      algorithm_id: "action-program",
      provenance: { search_algorithm_id: "go-explore" },
    },
  }), false);
  assert.equal(unsupportedStatLabelsVisible({
    policy: {
      algorithm_id: "dqn",
      provenance: {},
    },
  }), true);
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
