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

function selectedPoint(history, snapshot, view) {
  const sequence = view?.selectedSequence ?? snapshot?.transition?.sequence;
  return history.find((point) => Number(point.sequence) === Number(sequence))
    || history.at(-1)
    || null;
}

export function cursorIndex(history, view) {
  const index = history.findIndex(
    (point) => Number(point.sequence) === Number(view.selectedSequence),
  );
  return index < 0 ? null : index;
}

function actionLabel(value, snapshot) {
  if (!Number.isInteger(value)) return String(value);
  return snapshot?.session?.action_names?.[value] || `action ${value}`;
}

function renderedValue(value, descriptor, snapshot) {
  if (descriptor?.type === "categorical" && value !== null && value !== undefined) {
    return actionLabel(value, snapshot);
  }
  return formatTelemetryValue(value, descriptor);
}

function setLegend(target, descriptors) {
  target.replaceChildren(...descriptors.map((descriptor) => {
    const item = document.createElement("span");
    item.textContent = descriptor.shortLabel;
    item.style.setProperty("--legend-color", descriptor.color || "#53d4e8");
    return item;
  }));
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
      const unsupported = rows
        .filter(({ availability }) => availability.status === "unsupported")
        .map(({ descriptor }) => descriptor?.shortLabel)
        .filter(Boolean);
      foot.textContent = unsupported.length
        ? `Unsupported here: ${unsupported.join(", ")}.`
        : (block.foot || "");
      foot.hidden = !foot.textContent;
    },
  };
}

function makeLineBlock(block) {
  const section = document.createElement("section");
  section.className = "telemetry-block telemetry-plot";
  const descriptors = block.metrics.map(descriptorFor).filter(Boolean);
  appendHeading(
    section,
    block.title || descriptors.map((item) => item.shortLabel).join(" · ") || "History",
  );
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
  setLegend(legend, descriptors);
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
        foot.textContent = unavailable?.message || block.foot;
        foot.classList.toggle("warning", Boolean(unavailable));
      }
      section.dataset.telemetryStatus = unavailable?.status || "available";
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
      const numeric = values.every((value) => Number.isInteger(value) && value >= 0);
      const names = numeric
        ? [...(snapshot?.session?.action_names || [])]
        : [...new Set(values.map(String))].sort();
      if (numeric) {
        const maximum = Math.max(-1, ...values);
        while (names.length <= maximum) names.push(`action ${names.length}`);
      }
      if (!names.length) names.push("—");
      const counts = Array.from({ length: names.length }, () => 0);
      values.forEach((value) => {
        const index = numeric ? Number(value) : names.indexOf(String(value));
        if (index >= 0) counts[index] = (counts[index] || 0) + 1;
      });
      const point = view?.inspection
        ? selectedPoint(history, snapshot, view)
        : null;
      const selected = point && descriptor?.history ? descriptor.history(point) : null;
      const highlightIndex = selected === null || selected === undefined
        ? null
        : (numeric ? Number(selected) : names.indexOf(String(selected)));
      drawHistogram(canvas, counts, names, {
        highlightIndex: highlightIndex >= 0 ? highlightIndex : null,
      });
      caption.textContent = values.length
        ? `${values.length} values in the retained episode${
          highlightIndex >= 0 ? ` · selected ${names[highlightIndex]}.` : "."
        }`
        : `No ${descriptor?.shortLabel.toLowerCase() || "metric"} values observed.`;
    },
  };
}

function makeDistributionBlock(block) {
  const section = document.createElement("section");
  section.className = "telemetry-block telemetry-distribution";
  const descriptor = descriptorFor(block.metric);
  appendHeading(section, block.title || descriptor?.label || "Distribution");
  const target = document.createElement("div");
  target.className = "action-probabilities empty-state";
  section.append(target);
  appendFoot(
    section,
    block.foot || "Decision computed from the pre-action policy input.",
  );
  return {
    element: section,
    render({ snapshot }) {
      const availability = descriptorAvailability(descriptor, { snapshot });
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
        target.textContent = availability.message || "Not yet observed.";
        section.dataset.telemetryStatus = "not-yet-observed";
        return;
      }
      if (Array.isArray(decision.q_values)) {
        const names = snapshot?.session?.action_names || [];
        const values = decision.q_values.map(Number);
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
          label.textContent = names[index] || `action ${index}`;
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
          `Executed ${JSON.stringify(snapshot?.transition?.executed_action)}`,
          `mean ${JSON.stringify(decision.mean)}`,
          `std ${JSON.stringify(decision.stddev)}`,
        ].join(" · ");
        section.dataset.telemetryStatus = "available";
        return;
      }
      const names = snapshot?.session?.action_names || [];
      target.className = "action-probabilities";
      target.replaceChildren(...decision.probabilities.map((probability, index) => {
        const row = document.createElement("div");
        row.className = `action-row ${
          index === decision.selected_action ? "selected" : ""
        }`;
        const label = document.createElement("span");
        label.textContent = names[index] || `action ${index}`;
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
