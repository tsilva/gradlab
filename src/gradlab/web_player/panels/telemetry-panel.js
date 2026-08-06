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

const VALUE_COMPARISON_FOOT =
  "V(s) is expected discounted future policy reward; G(s) is this trajectory’s realized discounted future reward—not its success flag or cumulative episode return.";

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

function finiteHorizonFoot(snapshot) {
  const condition = snapshot?.session?.termination_conditions?.find(
    (item) => item?.id === "limit:max_episode_steps" && item.enabled,
  );
  const steps = Number(condition?.value);
  if (!Number.isInteger(steps) || steps <= 0) return "";
  return `Finite horizon: ${steps.toLocaleString()} policy steps. If remaining time is absent from the policy observation, visually similar early and late states can share one V(s).`;
}

export function lineBlockFootPresentation(block, unavailable, snapshot = null) {
  const visibleUnavailable = unavailable?.status === "protocol-error"
    ? null
    : unavailable;
  const valueComparison = Array.isArray(block.metrics)
    && block.metrics.includes("policy/value")
    && block.metrics.includes("policy/realized-return");
  const horizon = valueComparison
    ? finiteHorizonFoot(snapshot)
    : "";
  const text = [
    block.foot || (valueComparison ? VALUE_COMPARISON_FOOT : ""),
    horizon,
  ].filter(Boolean).join(" ");
  return {
    text: visibleUnavailable?.message || text,
    warning: Boolean(visibleUnavailable),
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

function makeLineBlock(block) {
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

  return {
    element: section,
    render({ snapshot, history, view }) {
      currentContext = { snapshot, history, view };
      const availabilities = descriptors.map((descriptor) => (
        descriptorAvailability(descriptor, { snapshot })
      ));
      const unavailable = availabilities.find(
        (availability) => availability.status !== "available"
          && availability.status !== "not-yet-observed",
      );
      if (foot) {
        const presentation = lineBlockFootPresentation(block, unavailable, snapshot);
        foot.textContent = presentation.text;
        foot.hidden = !foot.textContent;
        foot.classList.toggle("warning", presentation.warning);
      }
      section.dataset.telemetryStatus = unavailable?.status || "available";
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

function discreteActionOffset(value, start, count) {
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
  const start = Number(snapshot?.session?.action_contract?.policy?.space?.start || 0);
  const names = discreteActionLabels(snapshot, count);
  const stepProbabilities = decision.probabilities.map(probabilityValue);
  const invalidStepValues = stepProbabilities.filter((value) => value === null).length;
  const executedActions = (history || [])
    .map((point) => point?.executed_action)
    .filter((value) => value !== null && value !== undefined);
  const counts = Array.from({ length: count }, () => 0);
  let unmappable = 0;
  executedActions.forEach((value) => {
    const offset = discreteActionOffset(value, start, count);
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

function actionComparisonBar(name, series, value) {
  const item = document.createElement("div");
  item.className = `action-comparison-bar ${series}`;
  const track = document.createElement("div");
  track.className = "action-comparison-track";
  track.setAttribute("role", "progressbar");
  track.setAttribute("aria-label", `${name} ${
    series === "episode" ? "episode executed frequency" : "selected-step probability"
  }`);
  track.setAttribute("aria-valuemin", "0");
  track.setAttribute("aria-valuemax", "100");
  const fill = document.createElement("div");
  fill.className = "action-comparison-fill";
  fill.style.width = `${value === null ? 0 : value * 100}%`;
  track.append(fill);
  const amount = document.createElement("span");
  amount.className = "action-comparison-amount";
  amount.textContent = formatProbability(value);
  if (value === null) track.setAttribute("aria-valuetext", "Unavailable");
  else track.setAttribute("aria-valuenow", String(value * 100));
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
    actionComparisonBar(row.name, "episode", row.episodeProbability),
    actionComparisonBar(row.name, "step", row.stepProbability),
  );
  item.append(label, bars);
  return item;
}

function setActionComparisonCaption(target, presentation, decision, snapshot) {
  const count = presentation.history.sampleCount;
  const facts = [
    `${count.toLocaleString()} executed action${count === 1 ? "" : "s"} in the retained episode`,
  ];
  if (decision.selected_action !== null && decision.selected_action !== undefined) {
    facts.push(`policy selected ${formatActionValue(decision.selected_action, snapshot)}`);
  }
  const executed = snapshot?.transition?.executed_action;
  if (executed !== null && executed !== undefined) {
    facts.push(`executed ${formatActionValue(executed, snapshot)}`);
  }
  target.textContent = `${facts.join(" · ")}.`;
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
    <span class="episode">Episode executed frequency</span>
    <span class="step">Selected-step policy probability</span>
  `;
  const caption = document.createElement("p");
  caption.className = "panel-foot";
  section.append(legend, target, caption);
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
      caption.hidden = true;
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
      caption.hidden = false;
      setActionComparisonCaption(caption, presentation, decision, snapshot);
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
  table.className = "table-scroll";
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
  const heading = document.createElement("span");
  heading.className = "chart-heading";
  heading.textContent = block.title || "Reward ledger";
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
  toolbar.append(heading, scopeLabel);

  const state = document.createElement("div");
  state.className = "reward-analysis-state empty-state";
  const content = document.createElement("div");
  content.className = "reward-analysis-content";
  const transform = document.createElement("div");
  transform.className = "reward-transform-strip";
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
  content.append(transform, scroll, summary);
  const foot = appendFoot(
    section,
    block.foot || "Signed contribution uses |final reward|; activity share uses absolute per-step impacts, so penalties remain negative and cancellation stays visible.",
  );
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
    foot.classList.toggle(
      "warning",
      ["protocol-error", "partial-history"].includes(presentation.status),
    );
    if (!available) {
      state.className = `reward-analysis-state empty-state ${presentation.status}`;
      state.textContent = presentation.message;
      return;
    }
    const clip = presentation.contract.clipBounds;
    transform.replaceChildren();
    const parts = [
      `${presentation.scope === "episode" ? "Σ raw" : "Raw"} ${rewardNumber(presentation.raw, { signed: true })}`,
      `× ${rewardNumber(presentation.contract.rewardScale)}`,
      `= ${rewardNumber(presentation.preclip, { signed: true })} pre-clip`,
      clip
        ? `clip each step to [${rewardNumber(clip[0])}, ${rewardNumber(clip[1])}]`
        : "no clipping",
      `final ${rewardNumber(presentation.final, { signed: true })}`,
    ];
    parts.forEach((value, index) => {
      const item = document.createElement("span");
      item.textContent = value;
      if (index === parts.length - 1) item.className = "final";
      transform.append(item);
    });
    const maxMagnitude = Math.max(
      0,
      ...presentation.rows.map((row) => Math.abs(row.impact)),
    );
    body.replaceChildren(...presentation.rows.map(
      (row) => rewardLedgerRow(row, maxMagnitude),
    ));
    const cards = [
      ["Bonuses", presentation.positive, "positive"],
      ["Penalties", presentation.negative, "negative"],
      [presentation.scope === "episode" ? "Episode return" : "Final reward", presentation.final, "final"],
    ].map(([label, value, kind]) => {
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
  if (block.kind === "line") return makeLineBlock(block);
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
  const target = document.createElement("div");
  target.className = "telemetry-blocks";
  element.append(target);
  const blocks = definition.config.blocks.map(
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
