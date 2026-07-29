import {
  createPanel,
  drawHistogram,
  drawLines,
  setStats,
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
  const index = cursorIndex(history, view);
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
    item.style.setProperty("--legend-color", descriptor.color || "#53d4e8");
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

export function lineBlockFootPresentation(block, unavailable) {
  const visibleUnavailable = unavailable?.status === "protocol-error"
    ? null
    : unavailable;
  return {
    text: visibleUnavailable?.message || block.foot || "",
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
  return {
    element: section,
    render({ snapshot, history, view }) {
      const availabilities = descriptors.map((descriptor) => (
        descriptorAvailability(descriptor, { snapshot })
      ));
      const unavailable = availabilities.find(
        (availability) => availability.status !== "available"
          && availability.status !== "not-yet-observed",
      );
      if (foot) {
        const presentation = lineBlockFootPresentation(block, unavailable);
        foot.textContent = presentation.text;
        foot.hidden = !foot.textContent;
        foot.classList.toggle("warning", presentation.warning);
      }
      section.dataset.telemetryStatus = unavailable?.status || "available";
      lineLegendPresentation(descriptors, history, view).forEach(({ key, value }) => {
        const target = legendValues.get(key);
        if (target) target.textContent = value;
      });
      drawLines(
        canvas,
        descriptors.map((descriptor) => ({
          values: seriesForMetric(descriptor.key, history),
          color: descriptor.color || "#53d4e8",
        })),
        { cursorIndex: cursorIndex(history, view) },
      );
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
  section.append(target);
  const foot = appendFoot(section, block.foot, { force: true });
  return {
    element: section,
    render({ snapshot }) {
      const availability = descriptorAvailability(descriptor, { snapshot });
      const semantics = snapshot?.session?.action_contract?.policy?.semantics;
      if (semantics?.status === "unavailable") {
        foot.textContent = `Action semantics unavailable: ${
          semantics.reason || "the provider did not declare them"
        }.`;
      } else {
        foot.textContent = block.foot || "";
      }
      foot.hidden = !foot.textContent;
      section.hidden = !distributionBlockVisible(availability.status);
      if (
        availability.status !== "available"
        && availability.status !== "not-yet-observed"
      ) {
        target.className = `action-probabilities empty-state ${availability.status}`;
        target.textContent = availability.message;
        section.dataset.telemetryStatus = availability.status;
        return;
      }
      const decision = descriptorValue(descriptor, { snapshot });
      if (!decision) {
        target.className = "action-probabilities empty-state";
        target.textContent = availability.message || "N/A";
        section.dataset.telemetryStatus = "not-yet-observed";
        return;
      }
      if (Array.isArray(decision.q_values)) {
        const values = decision.q_values.map(Number);
        const names = discreteActionLabels(snapshot, values.length);
        const minimum = Math.min(...values);
        const maximum = Math.max(...values);
        const span = Math.max(maximum - minimum, Number.EPSILON);
        target.className = "action-probabilities action-values";
        target.replaceChildren(...values.map((value, index) => {
          const row = document.createElement("div");
          row.className = `action-row ${
            index === decision.selected_action ? "selected" : ""
          }`;
          const label = document.createElement("span");
          label.textContent = names[index] || formatActionValue(index, snapshot);
          const track = document.createElement("div");
          track.className = "probability-track";
          const fill = document.createElement("div");
          fill.className = "probability-fill";
          fill.style.width = `${Math.max(2, ((value - minimum) / span) * 100)}%`;
          track.append(fill);
          const amount = document.createElement("span");
          amount.textContent = Number.isFinite(value) ? value.toFixed(4) : "—";
          row.append(label, track, amount);
          return row;
        }));
        section.dataset.telemetryStatus = "available";
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
        return;
      }
      const names = discreteActionLabels(snapshot, decision.probabilities.length);
      target.className = "action-probabilities";
      target.replaceChildren(...decision.probabilities.map((probability, index) => {
        const row = document.createElement("div");
        row.className = `action-row ${
          index === decision.selected_action ? "selected" : ""
        }`;
        const label = document.createElement("span");
        label.textContent = names[index] || formatActionValue(index, snapshot);
        const track = document.createElement("div");
        track.className = "probability-track";
        const fill = document.createElement("div");
        fill.className = "probability-fill";
        fill.style.width = `${
          Math.max(0, Math.min(100, Number(probability) * 100))
        }%`;
        track.append(fill);
        const amount = document.createElement("span");
        amount.textContent = `${(Number(probability) * 100).toFixed(1)}%`;
        row.append(label, track, amount);
        return row;
      }));
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
      color: descriptor?.color || "#f0c36a",
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

function makeBlock(block, definition, services) {
  if (block.kind === "stats") return makeStatsBlock(block);
  if (block.kind === "line") return makeLineBlock(block);
  if (block.kind === "histogram") return makeHistogramBlock(block);
  if (block.kind === "distribution") return makeDistributionBlock(block);
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
