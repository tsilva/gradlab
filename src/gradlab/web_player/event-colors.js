export const EVENT_COLOR_PALETTE = Object.freeze([
  "var(--color-series-violet)",
  "var(--color-series-teal)",
  "var(--color-series-amber)",
  "var(--color-series-coral)",
  "var(--color-series-lavender)",
  "var(--color-series-aqua)",
  "var(--color-series-deep-violet)",
  "var(--color-series-burnt-orange)",
]);

const EPISODE_BOUNDARY_LABEL = "episode boundary";

export function eventLabels(point = {}) {
  const labels = Array.isArray(point.events)
    ? point.events.map((label) => String(label).trim()).filter(Boolean)
    : [];
  return labels.length ? labels : [EPISODE_BOUNDARY_LABEL];
}

export function eventColor(label) {
  let checksum = 2166136261;
  for (const character of String(label).trim()) {
    checksum ^= character.codePointAt(0);
    checksum = Math.imul(checksum, 16777619) >>> 0;
  }
  return EVENT_COLOR_PALETTE[checksum % EVENT_COLOR_PALETTE.length];
}

export function eventColorFill(labels) {
  const colors = labels.map(eventColor);
  if (colors.length <= 1) return colors[0] || EVENT_COLOR_PALETTE[0];
  return `linear-gradient(to bottom, ${colors.map((color, index) => {
    const start = (index / colors.length) * 100;
    const end = ((index + 1) / colors.length) * 100;
    return `${color} ${start}% ${end}%`;
  }).join(", ")})`;
}
