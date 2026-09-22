import { lineCursorIndex } from "./shared.js";
import { chartPoints } from "./chart-status.js";
import {
  descriptorAvailability,
  descriptorFor,
  descriptorValue,
  formatTelemetryValue,
  seriesForMetric,
} from "./telemetry.js";
import {
  scalarActionIndex,
  discreteActionLabels,
  formatActionValue,
} from "./action-contract.js";

const POLICY_DECISION_STATS = Object.freeze([
  "policy/mode",
  "action/policy",
  "policy/value",
  "policy/entropy",
  "policy/log-probability",
  "policy/program",
]);
const POLICY_DECISION_FOOTER_METRICS = Object.freeze([
  "policy/value",
  "policy/entropy",
  "policy/log-probability",
]);

export function selectedPoint(history, snapshot, view) {
  const sequence = view?.selectedSequence ?? snapshot?.transition?.sequence;
  if (sequence !== null && sequence !== undefined) {
    return history.find(
      (point) => Number(point.sequence) === Number(sequence),
    ) || null;
  }
  return history.at(-1) || null;
}


export function cursorIndex(history, view) {
  if (view?.selectedSequence === null || view?.selectedSequence === undefined) {
    return null;
  }
  const index = history.findIndex(
    (point) => Number(point.sequence) === Number(view.selectedSequence),
  );
  return index < 0 ? null : index;
}


export function lineLegendPresentation(descriptors, history, view) {
  // Reconstructed annotations may arrive after the selected live transition.
  // Use only an exact cursor match; never substitute a nearby sampled step.
  const recorded = chartPoints(history, view);
  if (cursorIndex(recorded, view) !== null) history = recorded;
  return lineLegendPresentationAtIndex(
    descriptors,
    history,
    cursorIndex(history, view),
  );
}


export function lineLegendPresentationAtIndex(descriptors, history, index) {
  return descriptors.map((descriptor) => ({
    key: descriptor.key,
    value: index === null
      ? "—"
      : formatTelemetryValue(
        seriesForMetric(descriptor.key, history)[index],
        descriptor,
      ),
  }));
}


export function lineCursorSequence(history, plot, x, pointCount) {
  const index = lineCursorIndex(plot, x, pointCount);
  if (index === null) return null;
  const sequence = history[index]?.sequence;
  return sequence === null || sequence === undefined ? null : sequence;
}


function renderedValue(value, descriptor, snapshot) {
  if (descriptor?.type === "categorical" && value !== null && value !== undefined) {
    return formatActionValue(value, snapshot);
  }
  return formatTelemetryValue(value, descriptor);
}


export function lineLegendPrefix(descriptor) {
  return `${descriptor.shortLabel} = `;
}


export function statsBlockFoot(block, snapshot) {
  return block.foot || "";
}

/** @param {any} block @param {any} unavailable @param {any} notice */

/** @param {any} block @param {any} unavailable @param {any} notice */
export function lineBlockFootPresentation(block, unavailable, notice = null) {
  const visibleUnavailable = unavailable?.status === "protocol-error"
    ? null
    : unavailable;
  const visibleMessage = visibleUnavailable || notice;
  return {
    text: visibleMessage?.message || block.foot || "",
    warning: Boolean(visibleMessage),
  };
}


export function lineBlockAvailability(descriptors, snapshot, history) {
  const availabilities = descriptors.map((descriptor) => {
    const point = history.find((candidate) => {
      const value = descriptor?.history ? descriptor.history(candidate) : null;
      return value !== null && value !== undefined && value !== "";
    });
    const availability = descriptorAvailability(descriptor, { snapshot, point });
    if (
      availability.status === "protocol-error"
      && descriptor?.phase === "post-episode"
      && !snapshot?.transition?.boundary
    ) {
      return { status: "not-yet-observed", message: "N/A" };
    }
    return availability;
  });
  const unavailable = availabilities.find(
    (availability) => availability.status !== "available"
      && availability.status !== "not-yet-observed",
  ) || null;
  const observed = descriptors.some((descriptor) => (
    seriesForMetric(descriptor.key, history).some(Number.isFinite)
  ));
  const notice = (
    descriptors.some((descriptor) => (
      ["policy/realized-return", "policy/value-error"].includes(descriptor?.key)
    ))
    && history.some((point) => point?.realized_return_bootstrapped === true)
  )
    ? { message: "Truncated episode: G(s) includes the final state's V(s) as a bootstrap." }
    : null;
  return {
    unavailable,
    status: unavailable?.status || (observed ? "available" : "not-yet-observed"),
    ...(notice ? { notice } : {}),
  };
}


export function policyDecisionLayoutEnabled(definition) {
  const blocks = definition?.config?.blocks;
  if (definition?.id !== "policy" || !Array.isArray(blocks) || blocks.length !== 2) {
    return false;
  }
  const [stats, distribution] = blocks;
  return stats?.kind === "stats"
    && Array.isArray(stats.metrics)
    && stats.metrics.length === POLICY_DECISION_STATS.length
    && stats.metrics.every((metric, index) => metric === POLICY_DECISION_STATS[index])
    && distribution?.kind === "distribution"
    && distribution.metric === "policy/distribution";
}


function policyDecisionMetric(key, snapshot, point) {
  const descriptor = descriptorFor(key);
  const availability = descriptorAvailability(descriptor, { snapshot, point });
  return {
    availability,
    label: descriptor?.shortLabel || key,
    value: availability.status === "available"
      ? renderedValue(
        descriptorValue(descriptor, { snapshot, point }),
        descriptor,
        snapshot,
      )
      : availability.message,
  };
}


export function policyDecisionRank(rows, selected) {
  if (!selected || selected.stepProbability === null) return null;
  return 1 + rows.filter((row) => (
    row.stepProbability !== null
      && row.stepProbability > selected.stepProbability
  )).length;
}


export function ordinal(value) {
  if (!Number.isInteger(value) || value < 1) return "—";
  const remainder = value % 100;
  if (remainder >= 11 && remainder <= 13) return `${value}th`;
  return `${value}${({ 1: "st", 2: "nd", 3: "rd" })[value % 10] || "th"}`;
}


export function policyDecisionPresentation(snapshot, history, view) {
  const point = selectedPoint(history, snapshot, view);
  const descriptor = descriptorFor("policy/distribution");
  const availability = descriptorAvailability(descriptor, { snapshot });
  const decision = availability.status === "available"
    ? descriptorValue(descriptor, { snapshot })
    : null;
  const comparison = actionComparisonPresentation(snapshot, history, decision);
  if (!comparison) {
    return { discrete: false, availability };
  }
  const selected = comparison.rows.find((row) => row.selected) || null;
  const actionMetric = policyDecisionMetric("action/policy", snapshot, point);
  const modeMetric = policyDecisionMetric("policy/mode", snapshot, point);
  const semantics = snapshot?.session?.action_contract?.policy?.semantics;
  const footMessages = [];
  if (semantics?.status === "unavailable") {
    footMessages.push(`Action semantics unavailable: ${
      semantics.reason || "the provider did not declare them"
    }.`);
  }
  for (const state of [comparison.step]) {
    if (state.message) footMessages.push(state.message);
  }
  return {
    discrete: true,
    action: selected?.name || actionMetric.value,
    effectiveAction: snapshot?.transition?.effective_action == null
      ? "Unavailable · not recorded"
      : formatActionValue(snapshot.transition.effective_action, snapshot),
    overrideRuleId: snapshot?.transition?.action_override_rule_id || null,
    environmentActionNote: snapshot?.session?.playback_contract?.environment_action_note || null,
    mode: modeMetric.value,
    rank: policyDecisionRank(comparison.rows, selected),
    choiceCount: comparison.rows.length,
    stepProbability: selected?.stepProbability ?? null,
    selectedIsHighest: selected?.highest ?? null,
    rows: comparison.rows,
    stats: POLICY_DECISION_FOOTER_METRICS
      .map((key) => policyDecisionMetric(key, snapshot, point))
      .filter(({ availability: metricAvailability }) => (
        metricAvailability.status !== "unsupported"
      )),
    foot: footMessages.join(" "),
    warning: [comparison.history.status, comparison.step.status]
      .some((status) => ["contract-incomparable", "protocol-error"].includes(status)),
  };
}


export function distributionBlockVisible(status) {
  return status !== "unsupported";
}


export function distributionBlockTitle(block, descriptor) {
  if (block.metric === "policy/distribution" && !block.title) return "";
  return block.title || descriptor?.label || "Distribution";
}


function discreteActionOffset(value, start, count, snapshot = null) {
  const legalTuples = snapshot?.session?.action_contract?.policy?.space?.legal_tuples;
  if (Array.isArray(value) && value.flat(Infinity).length > 1 && Array.isArray(legalTuples)) {
    const selected = value.flat(Infinity).map(Number);
    const legalIndex = legalTuples.findIndex((tuple) => {
      const candidate = Array.isArray(tuple) ? tuple.flat(Infinity).map(Number) : [];
      return candidate.length === selected.length
        && candidate.every((item, index) => item === selected[index]);
    });
    return legalIndex >= 0 && legalIndex < count ? legalIndex : null;
  }
  const action = scalarActionIndex(value);
  const offset = action === null ? null : action - start;
  return Number.isInteger(offset) && offset >= 0 && offset < count
    ? offset
    : null;
}


function probabilityValue(value) {
  if (value === null || value === undefined || value === "") return null;
  const probability = Number(value);
  return Number.isFinite(probability) && probability >= 0 && probability <= 1
    ? probability
    : null;
}


export function actionComparisonPresentation(snapshot, history, decision) {
  if (!Array.isArray(decision?.probabilities)) return null;
  const count = decision.probabilities.length;
  const policySpace = snapshot?.session?.action_contract?.policy?.space;
  const start = Array.isArray(policySpace?.legal_tuples)
    ? 0
    : Number(policySpace?.start || 0);
  const names = discreteActionLabels(snapshot, count);
  const stepProbabilities = decision.probabilities.map(probabilityValue);
  const invalidStepValues = stepProbabilities.filter((value) => value === null).length;
  const highestStepProbability = count && !invalidStepValues
    ? Math.max(...stepProbabilities)
    : null;
  const cursor = snapshot?.transition;
  const lastStep = Number(cursor?.step);
  const firstStep = Math.max(1, lastStep - 63);
  const points = [...new Map((history || []).filter((point) => (
    Number(point.episode) === Number(cursor?.episode)
    && Number(point.step) >= firstStep && Number(point.step) <= lastStep
  )).map((point) => [Number(point.step), point])).values()]
    .sort((a, b) => Number(a.step) - Number(b.step));
  const complete = Number.isInteger(lastStep) && lastStep >= 1
    && points.length === lastStep - firstStep + 1
    && points.every((point, index) => Number(point.step) === firstStep + index);
  function frequencies(field, eligible, label) {
    const population = points.filter(eligible);
    const counts = Array.from({ length: count }, () => 0);
    let missing = 0;
    let unmappable = 0;
    for (const point of population) {
      if (point[field] == null || (field === "policy_action" && point.action_source !== "policy")) {
        missing += 1;
        continue;
      }
      const offset = discreteActionOffset(point[field], start, count, snapshot);
      if (offset === null) unmappable += 1;
      else counts[offset] += 1;
    }
    const sampleCount = population.length - missing;
    const status = !complete ? "partial-history" : unmappable ? "contract-incomparable"
      : missing ? "unavailable" : !sampleCount ? "not-yet-observed" : "available";
    const message = `${label}: n=${sampleCount}` + (unmappable
      ? `; ${unmappable} of ${population.length} actions do not map to the policy action space.`
      : missing ? `; ${missing} actions not recorded.` : ".");
    return { sampleCount, status, message, values: counts.map((value) => status === "available" ? value / sampleCount : null) };
  }
  const policy = frequencies("policy_action", (point) => point.action_source !== "human", "Policy choices");
  const environment = frequencies("effective_action", () => true, "Environment actions");
  const historyStatus = !complete ? "partial-history"
    : [policy, environment].find((item) => item.status === "contract-incomparable")?.status || "available";
  const historyMessage = [
    !complete ? "Complete window unavailable." : "",
    ...[policy, environment]
      .filter((state) => state.status !== "available")
      .map((state) => state.message),
  ].filter(Boolean).join(" ");
  const selectedIndex = discreteActionOffset(decision.selected_action, start, count);
  const executedIndex = discreteActionOffset(
    snapshot?.transition?.executed_action,
    start,
    count,
    snapshot,
  );
  return {
    history: {
      sampleCount: points.length,
      firstStep, lastStep, policy, environment,
      status: historyStatus,
      message: historyMessage,
    },
    step: {
      status: invalidStepValues ? "protocol-error" : "available",
      message: invalidStepValues
        ? `Protocol error: ${invalidStepValues} policy probabilities are outside 0–100%.`
        : "",
    },
    rows: names.map((name, index) => ({
      name,
      policyFrequency: policy.values[index],
      environmentFrequency: environment.values[index],
      stepProbability: stepProbabilities[index],
      selected: index === selectedIndex,
      highest: highestStepProbability === null
        ? null
        : stepProbabilities[index] === highestStepProbability,
      executed: index === executedIndex,
    })),
  };
}


export function histogramSelectedLabel(names, highlightIndex) {
  return Number.isInteger(highlightIndex)
    && highlightIndex >= 0
    && highlightIndex < names.length
    ? names[highlightIndex]
    : null;
}


export function rewardSummaryCards(presentation) {
  return [
    ["Bonuses", presentation.positive, "positive"],
    ["Penalties", presentation.negative, "negative"],
    ["Pre-clip", presentation.preclip, "preclip"],
    ["Post-clip", presentation.final, "postclip"],
  ];
}


export function createTelemetryRenderer(blocks, beforeRender = () => {}) {
  let context = { snapshot: null, history: [], view: {} };
  const renderBlocks = () => {
    beforeRender(context);
    blocks.forEach((block) => block.render(context));
  };
  const update = (snapshot, history, view) => {
    view ||= {};
    // Snapshot presentation already supplies the matching history. Its queued
    // history callback must not rebuild the same tables and charts a second time.
    if (snapshot === context.snapshot && history === context.history
        && Object.keys(view).length === Object.keys(context.view).length
        && Object.entries(view).every(([key, value]) => context.view[key] === value)) return;
    context = { snapshot, history, view };
    renderBlocks();
  };
  return {
    render(snapshot, view = context.view) {
      update(snapshot, view?.history ?? context.history, view);
    },
    renderHistory(history, snapshot = context.snapshot, view = context.view) {
      update(snapshot, history, view);
    },
    resize: renderBlocks,
  };
}
