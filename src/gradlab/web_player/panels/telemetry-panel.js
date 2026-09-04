import {
  createPanel,
  drawHistogram,
  drawLines,
  lineCursorIndex,
  setStats,
  themeColor,
} from "./shared.js";
import {
  descriptorCatalog,
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
import {
  magnitudeShareLabel,
  rewardBreakdownPresentation,
  signedContributionLabel,
} from "./reward-breakdown.js";

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

function setLegend(target, descriptors) {
  const values = new Map();
  target.replaceChildren(...descriptors.map((descriptor) => {
    const item = document.createElement("span");
    const value = document.createElement("strong");
    value.className = "legend-value";
    value.textContent = "—";
    item.append(lineLegendPrefix(descriptor), value);
    item.style.setProperty(
      "--legend-color",
      themeColor(descriptor.color || "chartBar"),
    );
    values.set(descriptor.key, value);
    return item;
  }));
  return values;
}

function appendHeading(section, title) {
  const heading = document.createElement("div");
  heading.className = "chart-heading";
  const label = document.createElement("span");
  label.textContent = title;
  heading.append(label);
  section.append(heading);
}

function appendFoot(section, value, { force = false } = {}) {
  if (!value && !force) return null;
  const foot = document.createElement("p");
  foot.className = "panel-foot";
  foot.textContent = value;
  section.append(foot);
  return foot;
}

export function statsBlockFoot(block, snapshot) {
  return block.foot || "";
}

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

function makeStatsBlock(block) {
  const section = document.createElement("section");
  section.className = "telemetry-block telemetry-stats";
  if (block.title) appendHeading(section, block.title);
  const target = document.createElement("div");
  target.className = "stat-grid";
  section.append(target);
  const foot = appendFoot(section, block.foot, { force: true });
  return {
    element: section,
    render({ snapshot, history, view }) {
      const point = selectedPoint(history, snapshot, view);
      const rows = block.metrics.map((key) => {
        const descriptor = descriptorFor(key);
        const availability = descriptorAvailability(
          descriptor,
          { snapshot, point },
        );
        return {
          availability,
          descriptor,
          row: [
          descriptor?.shortLabel || key,
          availability.status === "available"
            ? renderedValue(
              descriptorValue(descriptor, { snapshot, point }),
              descriptor,
              snapshot,
            )
            : availability.message,
          ],
        };
      });
      setStats(
        target,
        rows
          .filter(({ availability }) => availability.status !== "unsupported")
          .map(({ row }) => row),
      );
      foot.textContent = statsBlockFoot(block, snapshot);
      foot.hidden = !foot.textContent;
    },
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
  for (const state of [comparison.history, comparison.step]) {
    if (state.message) footMessages.push(state.message);
  }
  return {
    discrete: true,
    action: selected?.name || actionMetric.value,
    mode: modeMetric.value,
    rank: policyDecisionRank(comparison.rows, selected),
    choiceCount: comparison.rows.length,
    stepProbability: selected?.stepProbability ?? null,
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

function makeLineBlock(block, services) {
  const section = document.createElement("section");
  section.className = "telemetry-block telemetry-plot";
  const descriptors = block.metrics.map(descriptorFor).filter(Boolean);
  const canvas = document.createElement("canvas");
  canvas.className = "chart telemetry-chart";
  canvas.setAttribute(
    "aria-label",
    `${descriptors.map((item) => item.label).join(" and ")} history`,
  );
  const legend = document.createElement("div");
  legend.className = "legend";
  section.append(canvas, legend);
  const foot = appendFoot(section, block.foot, { force: true });
  const legendValues = setLegend(legend, descriptors);
  let currentContext = { snapshot: null, history: [], view: {} };
  let chartGeometry = null;
  let hoverX = null;

  const renderChart = ({ history, view }) => {
    const series = descriptors.map((descriptor) => ({
      values: seriesForMetric(descriptor.key, history),
      color: themeColor(descriptor.color || "chartBar"),
    }));
    const defaultIndex = cursorIndex(history, view);
    const hoveredIndex = hoverX === null
      ? null
      : lineCursorIndex(chartGeometry?.plot, hoverX, chartGeometry?.pointCount);
    let displayedIndex = hoveredIndex ?? defaultIndex;
    chartGeometry = drawLines(canvas, series, { cursorIndex: displayedIndex });
    const correctedIndex = hoverX === null
      ? null
      : lineCursorIndex(chartGeometry?.plot, hoverX, chartGeometry?.pointCount);
    if (correctedIndex !== null && correctedIndex !== displayedIndex) {
      displayedIndex = correctedIndex;
      chartGeometry = drawLines(canvas, series, { cursorIndex: displayedIndex });
    }
    lineLegendPresentationAtIndex(descriptors, history, displayedIndex)
      .forEach(({ key, value }) => {
        const target = legendValues.get(key);
        if (target) target.textContent = value;
      });
  };

  canvas.addEventListener("pointermove", (event) => {
    const bounds = canvas.getBoundingClientRect();
    if (!(bounds.width > 0)) return;
    hoverX = (event.clientX - bounds.left) * (canvas.clientWidth / bounds.width);
    renderChart(currentContext);
  });
  canvas.addEventListener("pointerleave", () => {
    hoverX = null;
    renderChart(currentContext);
  });
  canvas.addEventListener("click", (event) => {
    const bounds = canvas.getBoundingClientRect();
    if (!(bounds.width > 0)) return;
    const x = (event.clientX - bounds.left) * (canvas.clientWidth / bounds.width);
    const sequence = lineCursorSequence(
      currentContext.history,
      chartGeometry?.plot,
      x,
      chartGeometry?.pointCount,
    );
    if (sequence !== null) services.inspectSequence?.(sequence);
  });

  return {
    element: section,
    render({ snapshot, history, view }) {
      currentContext = { snapshot, history, view };
      const availability = lineBlockAvailability(descriptors, snapshot, history);
      if (foot) {
        const presentation = lineBlockFootPresentation(
          block,
          availability.unavailable,
          availability.notice,
        );
        foot.textContent = presentation.text;
        foot.hidden = !foot.textContent;
        foot.classList.toggle("warning", presentation.warning);
      }
      section.dataset.telemetryStatus = availability.status;
      renderChart(currentContext);
    },
  };
}

function histogramValues(descriptor, history) {
  if (!descriptor?.history) return [];
  return history
    .map((point) => descriptor.history(point))
    .filter((value) => value !== null && value !== undefined);
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
  const executedActions = (history || [])
    .map((point) => point?.executed_action)
    .filter((value) => value !== null && value !== undefined);
  const counts = Array.from({ length: count }, () => 0);
  let unmappable = 0;
  executedActions.forEach((value) => {
    const offset = discreteActionOffset(value, start, count, snapshot);
    if (offset === null) unmappable += 1;
    else counts[offset] += 1;
  });
  const historyStatus = !executedActions.length
    ? "not-yet-observed"
    : unmappable
      ? "contract-incomparable"
      : "available";
  const historyMessage = historyStatus === "not-yet-observed"
    ? "No executed actions have been retained for this episode yet."
    : historyStatus === "contract-incomparable"
      ? `Episode action history is contract-incomparable: ${unmappable} of ${
        executedActions.length
      } executed actions do not map to this discrete policy distribution.`
      : "";
  const episodeProbabilities = historyStatus === "available"
    ? counts.map((value) => value / executedActions.length)
    : counts.map(() => null);
  const selectedIndex = discreteActionOffset(decision.selected_action, start, count);
  const executedIndex = discreteActionOffset(
    snapshot?.transition?.executed_action,
    start,
    count,
    snapshot,
  );
  return {
    history: {
      sampleCount: executedActions.length,
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
      episodeProbability: episodeProbabilities[index],
      stepProbability: stepProbabilities[index],
      selected: index === selectedIndex,
      executed: index === executedIndex,
    })),
  };
}

function formatProbability(value) {
  return value === null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function actionComparisonTrack(name, series, value) {
  const track = document.createElement("div");
  track.className = `action-comparison-track ${series}`;
  track.setAttribute("role", "progressbar");
  track.setAttribute("aria-label", `${name} ${
    series === "episode" ? "episode action frequency" : "step action probability"
  }`);
  track.setAttribute("aria-valuemin", "0");
  track.setAttribute("aria-valuemax", "100");
  const fill = document.createElement("div");
  fill.className = "action-comparison-fill";
  fill.style.width = `${value === null ? 0 : value * 100}%`;
  track.append(fill);
  if (value === null) track.setAttribute("aria-valuetext", "Unavailable");
  else track.setAttribute("aria-valuenow", String(value * 100));
  return track;
}

function actionComparisonBar(name, series, value) {
  const item = document.createElement("div");
  item.className = `action-comparison-bar ${series}`;
  const track = actionComparisonTrack(name, series, value);
  const amount = document.createElement("span");
  amount.className = "action-comparison-amount";
  amount.textContent = formatProbability(value);
  item.append(track, amount);
  return item;
}

function actionComparisonRow(row) {
  const item = document.createElement("div");
  item.className = [
    "action-comparison-row",
    row.selected ? "selected" : "",
    row.executed ? "executed" : "",
  ].filter(Boolean).join(" ");
  const label = document.createElement("span");
  label.className = "action-comparison-label";
  label.textContent = row.name;
  label.title = row.name;
  const bars = document.createElement("div");
  bars.className = "action-comparison-bars";
  bars.append(
    actionComparisonBar(row.name, "step", row.stepProbability),
    actionComparisonBar(row.name, "episode", row.episodeProbability),
  );
  item.append(label, bars);
  return item;
}

function policyDecisionComparisonRow(row) {
  const item = document.createElement("div");
  item.className = [
    "policy-decision-comparison-row",
    row.selected ? "selected" : "",
    row.executed ? "executed" : "",
  ].filter(Boolean).join(" ");
  item.setAttribute("role", "row");
  const label = document.createElement("span");
  label.className = "policy-decision-action-label";
  label.textContent = row.name;
  label.title = row.name;
  label.setAttribute("role", "rowheader");
  const bars = document.createElement("div");
  bars.className = "policy-decision-bars";
  bars.setAttribute("role", "cell");
  bars.append(
    actionComparisonTrack(row.name, "step", row.stepProbability),
    actionComparisonTrack(row.name, "episode", row.episodeProbability),
  );
  const step = document.createElement("span");
  step.className = "policy-decision-amount step";
  step.textContent = formatProbability(row.stepProbability);
  step.setAttribute("role", "cell");
  const episode = document.createElement("span");
  episode.className = "policy-decision-amount episode";
  episode.textContent = formatProbability(row.episodeProbability);
  episode.setAttribute("role", "cell");
  item.append(label, bars, step, episode);
  return item;
}

function makePolicyDecisionBlock(statsBlock, distributionBlock) {
  const section = document.createElement("section");
  section.className = "telemetry-block policy-decision-content";

  const discrete = document.createElement("div");
  discrete.className = "policy-decision-discrete";
  const hero = document.createElement("div");
  hero.className = "policy-decision-hero";
  const heroLine = document.createElement("div");
  heroLine.className = "policy-decision-hero-line";
  const action = document.createElement("strong");
  action.className = "policy-decision-action";
  const probability = document.createElement("strong");
  probability.className = "policy-decision-probability";
  const rank = document.createElement("span");
  rank.className = "policy-decision-rank";
  heroLine.append(action, probability);
  const modeLine = document.createElement("div");
  modeLine.className = "policy-decision-mode-line";
  const mode = document.createElement("span");
  mode.className = "policy-decision-mode";
  modeLine.append(mode, rank);
  hero.append(heroLine, modeLine);

  const comparison = document.createElement("div");
  comparison.className = "policy-decision-comparison";
  comparison.setAttribute("role", "table");
  comparison.setAttribute(
    "aria-label",
    "Step action probability compared with retained episode action frequency",
  );
  const comparisonHeader = document.createElement("div");
  comparisonHeader.className = "policy-decision-comparison-header";
  comparisonHeader.setAttribute("role", "row");
  for (const [label, className] of [
    ["Action", "action"],
    ["", "bars"],
    ["This step", "step"],
    ["Episode frequency", "episode"],
  ]) {
    const cell = document.createElement("span");
    cell.className = className;
    cell.textContent = label;
    cell.setAttribute("role", "columnheader");
    comparisonHeader.append(cell);
  }
  const comparisonRows = document.createElement("div");
  comparisonRows.className = "policy-decision-comparison-rows";
  comparisonRows.setAttribute("role", "rowgroup");
  comparison.append(comparisonHeader, comparisonRows);

  const footerStats = document.createElement("div");
  footerStats.className = "policy-decision-stats";
  const foot = appendFoot(discrete, "", { force: true });
  discrete.append(hero, comparison, footerStats);
  discrete.append(foot);

  const fallback = document.createElement("div");
  fallback.className = "policy-decision-fallback";
  const fallbackBlocks = [
    makeStatsBlock(statsBlock),
    makeDistributionBlock(distributionBlock),
  ];
  fallback.append(...fallbackBlocks.map((block) => block.element));
  section.append(discrete, fallback);

  return {
    element: section,
    render(context) {
      const presentation = policyDecisionPresentation(
        context.snapshot,
        context.history,
        context.view,
      );
      discrete.hidden = !presentation.discrete;
      fallback.hidden = presentation.discrete;
      if (!presentation.discrete) {
        delete section.dataset.telemetryStatus;
        fallbackBlocks.forEach((block) => block.render(context));
        return;
      }
      section.dataset.telemetryStatus = "available";
      action.textContent = presentation.action;
      mode.textContent = presentation.mode;
      probability.textContent = formatProbability(presentation.stepProbability);
      rank.textContent = presentation.rank === null
        ? "Rank unavailable"
        : `${ordinal(presentation.rank)} of ${presentation.choiceCount} choices`;
      rank.title = presentation.rank === null
        ? "The selected action cannot be ranked because its probability is unavailable."
        : `Selected action ranks ${ordinal(presentation.rank)} of ${
          presentation.choiceCount
        } by this step's action probabilities.`;
      comparisonRows.replaceChildren(
        ...presentation.rows.map(policyDecisionComparisonRow),
      );
      footerStats.replaceChildren(...presentation.stats.map((stat) => {
        const item = document.createElement("div");
        item.className = "policy-decision-stat";
        const label = document.createElement("span");
        label.textContent = stat.label;
        const value = document.createElement("strong");
        value.textContent = stat.value;
        item.append(label, value);
        return item;
      }));
      const blockFoot = [statsBlock.foot, distributionBlock.foot, presentation.foot]
        .filter(Boolean)
        .join(" ");
      foot.textContent = blockFoot;
      foot.hidden = !blockFoot;
      foot.classList.toggle("warning", presentation.warning);
    },
  };
}

export function histogramSelectedLabel(names, highlightIndex) {
  return Number.isInteger(highlightIndex)
    && highlightIndex >= 0
    && highlightIndex < names.length
    ? names[highlightIndex]
    : null;
}

function makeHistogramBlock(block) {
  const section = document.createElement("section");
  section.className = "telemetry-block telemetry-plot";
  const descriptor = descriptorFor(block.metric);
  appendHeading(section, block.title || descriptor?.label || "Histogram");
  const canvas = document.createElement("canvas");
  canvas.className = "chart telemetry-chart";
  canvas.setAttribute("aria-label", `${descriptor?.label || "Metric"} histogram`);
  const caption = document.createElement("p");
  caption.className = "panel-foot";
  section.append(canvas, caption);
  appendFoot(section, block.foot);
  return {
    element: section,
    render({ snapshot, history, view }) {
      const values = histogramValues(descriptor, history);
      const actionIndices = values.map(scalarActionIndex);
      const numeric = actionIndices.every((value) => value !== null && value >= 0);
      const names = numeric
        ? discreteActionLabels(snapshot)
        : [...new Set(values.map(String))].sort();
      if (numeric) {
        const maximum = Math.max(-1, ...actionIndices);
        while (names.length <= maximum) {
          names.push(formatActionValue(names.length, snapshot));
        }
      }
      if (!names.length) names.push("—");
      const counts = Array.from({ length: names.length }, () => 0);
      values.forEach((value, valueIndex) => {
        const index = numeric
          ? actionIndices[valueIndex]
          : names.indexOf(String(value));
        if (index >= 0) counts[index] = (counts[index] || 0) + 1;
      });
      const point = view?.inspection
        ? selectedPoint(history, snapshot, view)
        : null;
      const selected = point && descriptor?.history ? descriptor.history(point) : null;
      const highlightIndex = selected === null || selected === undefined
        ? null
        : (
          numeric
            ? scalarActionIndex(selected)
            : names.indexOf(String(selected))
        );
      const selectedLabel = histogramSelectedLabel(names, highlightIndex);
      drawHistogram(canvas, counts, names, {
        highlightIndex: selectedLabel === null ? null : highlightIndex,
      });
      caption.textContent = values.length
        ? `${values.length} values in the retained episode${
          selectedLabel === null ? "." : ` · selected ${selectedLabel}.`
        }`
        : `No ${descriptor?.shortLabel.toLowerCase() || "metric"} values observed.`;
    },
  };
}

function makeDistributionBlock(block) {
  const section = document.createElement("section");
  section.className = "telemetry-block telemetry-distribution";
  const descriptor = descriptorFor(block.metric);
  const title = distributionBlockTitle(block, descriptor);
  if (title) appendHeading(section, title);
  const target = document.createElement("div");
  target.className = "action-probabilities empty-state";
  const legend = document.createElement("div");
  legend.className = "action-comparison-legend";
  legend.innerHTML = `
    <div class="action-comparison-legend-series">
      <span class="step">Step action probability</span>
      <span class="episode">Episode action frequency</span>
    </div>
  `;
  const layout = document.createElement("div");
  layout.className = "action-comparison-layout";
  layout.append(legend, target);
  section.append(layout);
  const foot = appendFoot(section, block.foot, { force: true });
  return {
    element: section,
    render({ snapshot, history }) {
      const availability = descriptorAvailability(descriptor, { snapshot });
      const semantics = snapshot?.session?.action_contract?.policy?.semantics;
      const footMessages = [];
      if (block.foot) footMessages.push(block.foot);
      if (semantics?.status === "unavailable") {
        footMessages.push(`Action semantics unavailable: ${
          semantics.reason || "the provider did not declare them"
        }.`);
      }
      legend.hidden = true;
      foot.classList.remove("warning");
      section.hidden = !distributionBlockVisible(availability.status);
      if (
        availability.status !== "available"
        && availability.status !== "not-yet-observed"
      ) {
        target.className = `action-probabilities empty-state ${availability.status}`;
        target.textContent = availability.message;
        section.dataset.telemetryStatus = availability.status;
        foot.textContent = footMessages.join(" ");
        foot.hidden = !foot.textContent;
        return;
      }
      const decision = descriptorValue(descriptor, { snapshot });
      if (!decision) {
        target.className = "action-probabilities empty-state";
        target.textContent = availability.message || "N/A";
        section.dataset.telemetryStatus = "not-yet-observed";
        foot.textContent = footMessages.join(" ");
        foot.hidden = !foot.textContent;
        return;
      }
      if (!Array.isArray(decision.probabilities)) {
        target.className = "distribution-summary";
        target.textContent = [
          `Executed ${formatActionValue(snapshot?.transition?.executed_action, snapshot)}`,
          `mean ${JSON.stringify(decision.mean)}`,
          `std ${JSON.stringify(decision.stddev)}`,
        ].join(" · ");
        section.dataset.telemetryStatus = "available";
        foot.textContent = footMessages.join(" ");
        foot.hidden = !foot.textContent;
        return;
      }
      const presentation = actionComparisonPresentation(snapshot, history, decision);
      target.className = "action-comparison";
      target.replaceChildren(...presentation.rows.map(actionComparisonRow));
      legend.hidden = false;
      for (const state of [presentation.history, presentation.step]) {
        if (state.message) footMessages.push(state.message);
        if (["contract-incomparable", "protocol-error"].includes(state.status)) {
          foot.classList.add("warning");
        }
      }
      foot.textContent = footMessages.join(" ");
      foot.hidden = !foot.textContent;
      section.dataset.telemetryStatus = "available";
    },
  };
}

function namespaceDescriptors(namespace, snapshot, history) {
  return [...descriptorCatalog(snapshot, history).values()]
    .filter((descriptor) => descriptor.namespace === namespace)
    .sort((left, right) => left.label.localeCompare(right.label));
}

function makeNamespaceBlock(block, definition, services) {
  const section = document.createElement("section");
  section.className = "telemetry-block telemetry-namespace";
  const toolbar = document.createElement("div");
  toolbar.className = "signal-toolbar";
  const label = document.createElement("label");
  const caption = document.createElement("span");
  caption.className = "signal-toolbar-label";
  caption.textContent = block.namespace === "signal"
    ? "Chart signal"
    : "Chart reward component";
  const select = document.createElement("select");
  select.setAttribute("aria-label", caption.textContent);
  label.append(caption, select);
  toolbar.append(label);
  const canvas = document.createElement("canvas");
  canvas.className = "chart telemetry-chart";
  canvas.setAttribute("aria-label", `${caption.textContent} history`);
  const table = document.createElement("div");
  table.className = "telemetry-namespace-table";
  const tableElement = document.createElement("table");
  const body = document.createElement("tbody");
  tableElement.append(body);
  table.append(tableElement);
  section.append(toolbar, canvas, table);
  appendFoot(
    section,
    block.foot || (
      block.namespace === "signal"
        ? "Post-action environment signals."
        : "Post-action reward components."
    ),
  );
  let selected = block.metric || "";
  let currentContext = { snapshot: null, history: [], view: {} };

  const render = ({ snapshot, history, view }) => {
    currentContext = { snapshot, history, view };
    const descriptors = namespaceDescriptors(block.namespace, snapshot, history);
    if (!descriptors.some((descriptor) => descriptor.key === selected)) {
      selected = descriptors[0]?.key || "";
    }
    const prior = select.value;
    select.replaceChildren();
    if (!descriptors.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "No metrics observed";
      select.append(option);
      select.disabled = true;
    } else {
      select.disabled = false;
      select.append(...descriptors.map((descriptor) => {
        const option = document.createElement("option");
        option.value = descriptor.key;
        option.textContent = descriptor.label;
        return option;
      }));
      select.value = descriptors.some((descriptor) => descriptor.key === selected)
        ? selected
        : prior;
    }
    const descriptor = descriptorFor(selected);
    drawLines(canvas, [{
      values: descriptor ? seriesForMetric(descriptor.key, history) : [],
      color: themeColor(descriptor?.color || "chartHighlight"),
    }], { cursorIndex: cursorIndex(history, view) });
    const point = selectedPoint(history, snapshot, view);
    body.replaceChildren(...descriptors.map((item) => {
      const row = document.createElement("tr");
      const key = document.createElement("td");
      key.textContent = item.label;
      const value = document.createElement("td");
      value.textContent = renderedValue(
        descriptorValue(item, { snapshot, point }),
        item,
        snapshot,
      );
      row.append(key, value);
      return row;
    }));
  };

  select.addEventListener("change", () => {
    selected = select.value;
    const blocks = definition.config.blocks.map((candidate) => (
      candidate === block ? { ...candidate, metric: selected } : candidate
    ));
    services.updatePanelConfig?.(definition.id, { blocks });
    render(currentContext);
  });

  return { element: section, render };
}

function rewardNumber(value, { signed = false } = {}) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
  const number = Number(value);
  const prefix = signed && number > 0 ? "+" : "";
  return `${prefix}${number.toLocaleString(undefined, {
    maximumFractionDigits: 4,
    minimumFractionDigits: Math.abs(number) > 0 && Math.abs(number) < 0.001 ? 4 : 0,
  })}`;
}

export function rewardSummaryCards(presentation) {
  return [
    ["Bonuses", presentation.positive, "positive"],
    ["Penalties", presentation.negative, "negative"],
    ["Pre-clip", presentation.preclip, "preclip"],
    ["Post-clip", presentation.final, "postclip"],
  ];
}

function rewardLedgerRow(row, maxMagnitude) {
  const tr = document.createElement("tr");
  const component = document.createElement("th");
  component.scope = "row";
  const identity = document.createElement("div");
  identity.className = "reward-ledger-identity";
  const sign = document.createElement("span");
  const signName = row.impact > 0 ? "positive" : row.impact < 0 ? "negative" : "zero";
  sign.className = `reward-sign ${signName}`;
  sign.textContent = row.impact > 0 ? "+" : row.impact < 0 ? "−" : "0";
  sign.setAttribute("aria-label", `${signName} impact`);
  const label = document.createElement("span");
  label.textContent = row.label;
  identity.append(sign, label);
  const bar = document.createElement("div");
  bar.className = "reward-zero-bar";
  bar.setAttribute("aria-hidden", "true");
  const fill = document.createElement("span");
  fill.className = row.impact > 0 ? "positive" : row.impact < 0 ? "negative" : "zero";
  fill.style.setProperty(
    "--reward-bar-size",
    `${maxMagnitude > 0 ? (50 * Math.abs(row.impact)) / maxMagnitude : 0}%`,
  );
  bar.append(fill);
  component.append(identity, bar);
  const raw = document.createElement("td");
  raw.textContent = rewardNumber(row.raw, { signed: true });
  const impact = document.createElement("td");
  impact.textContent = rewardNumber(row.impact, { signed: true });
  const contribution = document.createElement("td");
  contribution.textContent = signedContributionLabel(row.signedContribution);
  const activity = document.createElement("td");
  activity.textContent = magnitudeShareLabel(row.magnitudeShare);
  tr.className = `reward-ledger-row ${row.kind}`;
  tr.append(component, raw, impact, contribution, activity);
  return tr;
}

function makeRewardBreakdownBlock(block, definition, services) {
  const section = document.createElement("section");
  section.className = "telemetry-block telemetry-reward-breakdown";
  const toolbar = document.createElement("div");
  toolbar.className = "reward-analysis-toolbar";
  toolbar.classList.toggle("titleless", !block.title);
  if (block.title) {
    const heading = document.createElement("span");
    heading.className = "chart-heading";
    heading.textContent = block.title;
    toolbar.append(heading);
  }
  const scopeLabel = document.createElement("label");
  scopeLabel.append("Scope ");
  const scope = document.createElement("select");
  scope.setAttribute("aria-label", "Reward analysis scope");
  scope.append(
    Object.assign(document.createElement("option"), { value: "step", textContent: "Selected step" }),
    Object.assign(document.createElement("option"), { value: "episode", textContent: "Episode to cursor" }),
  );
  scope.value = block.scope === "episode" ? "episode" : "step";
  scopeLabel.append(scope);
  toolbar.append(scopeLabel);

  const state = document.createElement("div");
  state.className = "reward-analysis-state empty-state";
  const content = document.createElement("div");
  content.className = "reward-analysis-content";
  const scroll = document.createElement("div");
  scroll.className = "table-scroll reward-ledger-scroll";
  const table = document.createElement("table");
  table.className = "reward-ledger-table";
  const head = document.createElement("thead");
  const header = document.createElement("tr");
  ["Component", "Raw", "Final impact", "Signed contribution", "Activity share"]
    .forEach((value) => {
      const cell = document.createElement("th");
      cell.scope = "col";
      cell.textContent = value;
      header.append(cell);
    });
  head.append(header);
  const body = document.createElement("tbody");
  table.append(head, body);
  scroll.append(table);
  const summary = document.createElement("div");
  summary.className = "reward-ledger-summary";
  content.append(scroll, summary);
  const foot = appendFoot(section, block.foot);
  section.prepend(toolbar);
  section.insertBefore(state, foot);
  section.insertBefore(content, foot);
  let currentContext = { snapshot: null, history: [], view: {} };

  const render = ({ snapshot, history, view }) => {
    currentContext = { snapshot, history, view };
    const presentation = rewardBreakdownPresentation({
      snapshot,
      history,
      view,
      scope: scope.value,
    });
    section.dataset.telemetryStatus = presentation.status;
    const available = presentation.status === "available";
    state.hidden = available;
    content.hidden = !available;
    foot?.classList.toggle(
      "warning",
      ["protocol-error", "partial-history"].includes(presentation.status),
    );
    if (!available) {
      state.className = `reward-analysis-state empty-state ${presentation.status}`;
      state.textContent = presentation.message;
      return;
    }
    const maxMagnitude = Math.max(
      0,
      ...presentation.rows.map((row) => Math.abs(row.impact)),
    );
    body.replaceChildren(...presentation.rows.map(
      (row) => rewardLedgerRow(row, maxMagnitude),
    ));
    const cards = rewardSummaryCards(presentation).map(([label, value, kind]) => {
      const card = document.createElement("div");
      card.className = kind;
      const name = document.createElement("span");
      name.textContent = label;
      const number = document.createElement("strong");
      number.textContent = rewardNumber(value, { signed: true });
      card.append(name, number);
      return card;
    });
    summary.replaceChildren(...cards);
  };

  scope.addEventListener("change", () => {
    const blocks = definition.config.blocks.map((candidate) => (
      candidate === block ? { ...candidate, scope: scope.value } : candidate
    ));
    services.updatePanelConfig?.(definition.id, { blocks });
    render(currentContext);
  });
  return { element: section, render };
}

function makeBlock(block, definition, services) {
  if (block.kind === "stats") return makeStatsBlock(block);
  if (block.kind === "line") return makeLineBlock(block, services);
  if (block.kind === "histogram") return makeHistogramBlock(block);
  if (block.kind === "distribution") return makeDistributionBlock(block);
  if (block.kind === "reward-breakdown") {
    return makeRewardBreakdownBlock(block, definition, services);
  }
  return makeNamespaceBlock(block, definition, services);
}

export function mount({ definition, services }) {
  const element = createPanel({
    id: definition.id,
    label: definition.label,
  });
  const policyDecision = policyDecisionLayoutEnabled(definition);
  element.classList.toggle("policy-decision-panel", policyDecision);
  const target = document.createElement("div");
  target.className = "telemetry-blocks";
  element.append(target);
  const blocks = policyDecision
    ? [makePolicyDecisionBlock(
      definition.config.blocks[0],
      definition.config.blocks[1],
    )]
    : definition.config.blocks.map(
      (block) => makeBlock(block, definition, services),
    );
  target.replaceChildren(...blocks.map((block) => block.element));
  let context = { snapshot: null, history: [], view: {} };

  const renderBlocks = () => blocks.forEach((block) => block.render(context));
  return {
    element,
    render(snapshot, view = context.view) {
      context = { ...context, snapshot, view: view || {} };
      renderBlocks();
    },
    renderHistory(history, snapshot = context.snapshot, view = context.view) {
      context = { history, snapshot, view: view || {} };
      renderBlocks();
    },
    resize: renderBlocks,
  };
}
