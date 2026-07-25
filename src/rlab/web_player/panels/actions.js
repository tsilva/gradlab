import { createPanel, drawHistogram } from "./shared.js";

export function mount({ definition }) {
  const element = createPanel({
    id: definition.id,
    label: "Action histogram",
    body: `
      <canvas data-chart class="chart" aria-label="Action histogram"></canvas>
      <p data-caption class="panel-foot">No policy actions observed.</p>
    `,
  });
  const canvas = element.querySelector("[data-chart]");
  const caption = element.querySelector("[data-caption]");
  let history = [];
  let snapshot = null;
  let view = {};

  const render = () => {
    const names = snapshot?.session?.action_names || [];
    const counts = Array.from({ length: names.length || 1 }, () => 0);
    history.forEach((point) => {
      if (Number.isInteger(point.action) && point.action >= 0) {
        counts[point.action] = (counts[point.action] || 0) + 1;
      }
    });
    const selectedPoint = view.inspection
      ? history.find(
        (point) => Number(point.sequence) === Number(view.selectedSequence),
      )
      : null;
    const highlightIndex = Number.isInteger(selectedPoint?.action)
      ? selectedPoint.action
      : null;
    drawHistogram(canvas, counts, names, { highlightIndex });
    const total = counts.reduce((sum, value) => sum + value, 0);
    caption.textContent = total
      ? `${total} actions in the retained episode${highlightIndex === null
        ? "."
        : ` · selected ${names[highlightIndex] || highlightIndex}.`}`
      : "No policy actions observed.";
  };

  return {
    element,
    render(next, nextView = view) {
      snapshot = next;
      view = nextView || {};
      render();
    },
    renderHistory(next, current, nextView = view) {
      history = next;
      snapshot = current;
      view = nextView || {};
      render();
    },
    resize: render,
  };
}
