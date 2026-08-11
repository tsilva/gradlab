const ICONS = "/assets/tabler-icons.svg";
const CANVAS_FONT_UI = '12px "Inter", system-ui, sans-serif';
const CANVAS_FONT_UI_SEMIBOLD = '600 12px "Inter", system-ui, sans-serif';
const CANVAS_FONT_MONO = '12px "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace';

const THEME_COLOR_PROPERTIES = Object.freeze({
  chartSurface: "--color-chart-surface",
  chartGrid: "--color-chart-grid",
  chartAxis: "--color-chart-axis",
  chartBar: "--color-chart-bar",
  chartHighlight: "--color-chart-highlight",
  seriesViolet: "--color-series-violet",
  seriesTeal: "--color-series-teal",
  seriesAmber: "--color-series-amber",
  seriesCoral: "--color-series-coral",
  seriesLavender: "--color-series-lavender",
  seriesAqua: "--color-series-aqua",
  seriesDeepViolet: "--color-series-deep-violet",
  seriesBurntOrange: "--color-series-burnt-orange",
});

export function themeColor(name) {
  const property = THEME_COLOR_PROPERTIES[name];
  if (!property) throw new Error(`Unknown player theme color: ${name}`);
  const value = getComputedStyle(document.documentElement).getPropertyValue(property).trim();
  if (!value) throw new Error(`Missing player theme color: ${property}`);
  return value;
}

export function text(value, fallback = "—") {
  return value === null || value === undefined || value === "" ? fallback : String(value);
}

export function displayedStep(snapshot) {
  return snapshot?.transition?.step ?? snapshot?.session?.step;
}

export function displayedEpisode(snapshot) {
  return snapshot?.transition?.episode ?? snapshot?.session?.episode;
}

export function timelineLabel(snapshot) {
  return `EPISODE ${text(displayedEpisode(snapshot))} · STEP ${text(displayedStep(snapshot))}`;
}

export function setSvgUseHref(element, href) {
  if (element.getAttribute("href") === href) return false;
  element.setAttribute("href", href);
  return true;
}

export function number(value, digits = 3) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—";
}

export function createPanel({
  id,
  label,
  body = "",
  className = "",
  tag = "section",
  headerClass = "",
}) {
  const element = document.createElement(tag);
  element.className = `panel ${className}`.trim();
  element.dataset.panel = id;
  const heading = `${id}-panel-heading`;
  element.setAttribute("aria-labelledby", heading);
  element.innerHTML = `
    <header class="panel-header ${headerClass}">
      <button data-drag-handle class="icon-button icon-only panel-drag" type="button"><svg class="icon" aria-hidden="true"><use href="${ICONS}#ti-grip-vertical"></use></svg></button>
      <div class="panel-title"><h2 id="${heading}"></h2></div>
      <button data-panel-menu="${id}" class="icon-button icon-only" type="button"><svg class="icon" aria-hidden="true"><use href="${ICONS}#ti-dots-vertical"></use></svg></button>
    </header>
    ${body}
  `;
  const renderedLabel = text(label, "Panel");
  const drag = element.querySelector("[data-drag-handle]");
  const menu = element.querySelector("[data-panel-menu]");
  element.querySelector("h2").textContent = renderedLabel;
  drag.setAttribute("aria-label", `Move ${renderedLabel} panel`);
  drag.title = drag.getAttribute("aria-label");
  menu.setAttribute("aria-label", `${renderedLabel} panel options`);
  menu.title = menu.getAttribute("aria-label");
  return element;
}

export function setStats(target, values) {
  target.replaceChildren(...values.map(([label, value]) => {
    const box = document.createElement("div");
    box.className = "stat";
    const key = document.createElement("span");
    key.className = "stat-label";
    key.textContent = label;
    const rendered = document.createElement("span");
    rendered.className = "stat-value";
    rendered.textContent = text(value);
    box.append(key, rendered);
    return box;
  }));
}

export function renderJson(target, value, fallback) {
  if (value === null || value === undefined) {
    target.textContent = fallback;
    return;
  }
  const source = JSON.stringify(value, null, 2);
  const tokens = /"(?:\\.|[^"\\])*"(?=\s*:)|"(?:\\.|[^"\\])*"|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\b(?:true|false|null)\b/g;
  const fragment = document.createDocumentFragment();
  let cursor = 0;
  for (const match of source.matchAll(tokens)) {
    fragment.append(document.createTextNode(source.slice(cursor, match.index)));
    const token = document.createElement("span");
    const raw = match[0];
    if (raw.startsWith('"')) {
      token.className = source.slice(match.index + raw.length).match(/^\s*:/)
        ? "json-key"
        : "json-string";
    } else if (raw === "true" || raw === "false") token.className = "json-boolean";
    else if (raw === "null") token.className = "json-null";
    else token.className = "json-number";
    token.textContent = raw;
    fragment.append(token);
    cursor = match.index + raw.length;
  }
  fragment.append(document.createTextNode(source.slice(cursor)));
  target.replaceChildren(fragment);
}

function resizeCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(240, canvas.clientWidth);
  const height = Math.max(120, canvas.clientHeight);
  if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
  }
  return { context: canvas.getContext("2d"), ratio, width, height };
}

function niceTickStep(span, intervals = 4) {
  const rough = span / intervals;
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const normalized = rough / magnitude;
  const factor = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
  return factor * magnitude;
}

function lineChartScale(values) {
  let dataMin = Math.min(...values);
  let dataMax = Math.max(...values);
  if (dataMin === dataMax) {
    const margin = dataMin === 0 ? 1 : Math.abs(dataMin) * 0.1;
    dataMin -= margin;
    dataMax += margin;
  }
  const step = niceTickStep(dataMax - dataMin);
  const min = Math.floor(dataMin / step) * step;
  const max = Math.ceil(dataMax / step) * step;
  const intervals = Math.max(1, Math.round((max - min) / step));
  return {
    min,
    max,
    step,
    ticks: Array.from({ length: intervals + 1 }, (_, index) => min + index * step),
  };
}

function formatAxisValue(value, step) {
  const absolute = Math.abs(value);
  if (absolute >= 1e6 || (absolute > 0 && absolute < 1e-4)) {
    return value.toExponential(1).replace(".0e", "e").replace("e+", "e");
  }
  const decimals = Math.min(6, Math.max(0, -Math.floor(Math.log10(Math.abs(step)))));
  const rendered = value.toFixed(decimals);
  return rendered === "-0" ? "0" : rendered;
}

export function lineCursorX(plot, cursorIndex, pointCount) {
  const rawX = plot.left
    + (cursorIndex / Math.max(1, pointCount - 1)) * (plot.right - plot.left);
  return Math.max(plot.left + 1, Math.min(plot.right - 1, rawX));
}

export function lineCursorIndex(plot, x, pointCount) {
  if (!plot || !Number.isFinite(x) || !Number.isInteger(pointCount) || pointCount <= 0) {
    return null;
  }
  const span = plot.right - plot.left;
  if (!(span > 0)) return null;
  const fraction = Math.max(0, Math.min(1, (x - plot.left) / span));
  return Math.round(fraction * Math.max(0, pointCount - 1));
}

export function drawLines(canvas, series, { cursorIndex = null } = {}) {
  const { context, ratio, width, height } = resizeCanvas(canvas);
  const chartSurface = themeColor("chartSurface");
  const chartGrid = themeColor("chartGrid");
  const chartAxis = themeColor("chartAxis");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  context.fillStyle = chartSurface;
  context.fillRect(0, 0, width, height);
  const values = series.flatMap((item) => item.values.filter(Number.isFinite));
  if (!values.length) {
    context.fillStyle = chartAxis;
    context.font = CANVAS_FONT_UI;
    context.textAlign = "left";
    context.textBaseline = "alphabetic";
    context.fillText("No history yet", 12, 22);
    return null;
  }
  const scale = lineChartScale(values);
  const labels = scale.ticks.map((value) => formatAxisValue(value, scale.step));
  context.font = CANVAS_FONT_MONO;
  const labelWidth = Math.max(...labels.map((label) => context.measureText(label).width));
  const plot = {
    left: Math.ceil(labelWidth) + 16,
    right: width - 12,
    top: 10,
    bottom: height - 10,
  };
  context.strokeStyle = chartGrid;
  context.lineWidth = 1;
  context.fillStyle = chartAxis;
  context.textAlign = "right";
  context.textBaseline = "middle";
  scale.ticks.forEach((tick, index) => {
    const y = plot.bottom
      - ((tick - scale.min) / (scale.max - scale.min)) * (plot.bottom - plot.top);
    context.beginPath();
    context.moveTo(plot.left, y);
    context.lineTo(plot.right, y);
    context.stroke();
    context.fillText(labels[index], plot.left - 6, y);
  });
  series.forEach(({ values: points, color }) => {
    context.strokeStyle = color;
    context.lineWidth = 1.5;
    context.beginPath();
    points.forEach((value, index) => {
      if (!Number.isFinite(value)) return;
      const x = plot.left
        + (index / Math.max(1, points.length - 1)) * (plot.right - plot.left);
      const y = plot.bottom
        - ((value - scale.min) / (scale.max - scale.min)) * (plot.bottom - plot.top);
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.stroke();
  });
  const pointCount = Math.max(0, ...series.map((item) => item.values.length));
  if (
    Number.isInteger(cursorIndex)
    && cursorIndex >= 0
    && cursorIndex < pointCount
  ) {
    // Keep the dashed stroke inside the bitmap at both endpoints. A cursor at
    // the final sample otherwise sits on the right clipping edge and vanishes.
    const x = lineCursorX(plot, cursorIndex, pointCount);
    context.save();
    context.strokeStyle = themeColor("chartHighlight");
    context.lineWidth = 1.5;
    context.setLineDash([4, 3]);
    context.beginPath();
    context.moveTo(x, plot.top);
    context.lineTo(x, plot.bottom);
    context.stroke();
    context.restore();
  }
  return { plot, pointCount };
}

function fitCanvasLabel(context, value, maxWidth) {
  const label = String(value);
  if (context.measureText(label).width <= maxWidth) return label;
  let end = label.length;
  while (end > 0 && context.measureText(`${label.slice(0, end)}…`).width > maxWidth) end -= 1;
  return end > 0 ? `${label.slice(0, end)}…` : "…";
}

export function drawHistogram(
  canvas,
  counts,
  names,
  { highlightIndex = null } = {},
) {
  const { context, ratio, width, height } = resizeCanvas(canvas);
  const chartSurface = themeColor("chartSurface");
  const chartGrid = themeColor("chartGrid");
  const chartAxis = themeColor("chartAxis");
  const chartBar = themeColor("chartBar");
  const chartHighlight = themeColor("chartHighlight");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  context.fillStyle = chartSurface;
  context.fillRect(0, 0, width, height);
  const max = Math.max(1, ...counts);
  const scale = lineChartScale([0, max]);
  const labels = scale.ticks.map((value) => formatAxisValue(value, scale.step));
  context.font = CANVAS_FONT_MONO;
  const labelWidth = Math.max(...labels.map((label) => context.measureText(label).width));
  const plot = {
    left: Math.ceil(labelWidth) + 16,
    right: width - 12,
    top: 10,
    bottom: height - 30,
  };
  context.strokeStyle = chartGrid;
  context.lineWidth = 1;
  context.fillStyle = chartAxis;
  context.textAlign = "right";
  context.textBaseline = "middle";
  scale.ticks.forEach((tick, index) => {
    const y = plot.bottom
      - ((tick - scale.min) / (scale.max - scale.min)) * (plot.bottom - plot.top);
    context.beginPath();
    context.moveTo(plot.left, y);
    context.lineTo(plot.right, y);
    context.stroke();
    context.fillText(labels[index], plot.left - 6, y);
  });
  const gap = 4;
  const barWidth = Math.max(
    4,
    (plot.right - plot.left) / Math.max(1, counts.length) - gap,
  );
  counts.forEach((count, index) => {
    const barHeight = (count / scale.max) * Math.max(0, plot.bottom - plot.top);
    const x = plot.left + index * (barWidth + gap);
    context.fillStyle = index === highlightIndex ? chartHighlight : chartBar;
    context.fillRect(x, plot.bottom - barHeight, barWidth, barHeight);
    context.fillStyle = chartAxis;
    context.font = CANVAS_FONT_UI_SEMIBOLD;
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText(
      fitCanvasLabel(context, names[index] || String(index), barWidth + gap - 4),
      x + barWidth / 2,
      height - 14,
    );
  });
}
