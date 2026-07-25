import { createPanel, drawLines, number, setStats } from "./shared.js";

function legend(target, items) {
  target.replaceChildren(...items.map(([label, color]) => {
    const item = document.createElement("span");
    item.textContent = label;
    item.style.setProperty("--legend-color", color);
    return item;
  }));
}

export function mount({ definition }) {
  const element = createPanel({
    id: definition.id,
    label: definition.label,
    body: `
      <div data-stats class="stat-grid"></div>
      <div class="reward-plots">
        <section class="reward-plot" aria-labelledby="value-plot-title">
          <div class="chart-heading">
            <span id="value-plot-title">Value estimate vs realized return-to-go</span>
            <span data-value-status class="chart-status"></span>
          </div>
          <canvas data-value-chart class="chart reward-chart" aria-label="Policy value estimate and realized discounted return-to-go history"></canvas>
          <div data-value-legend class="legend"></div>
          <p class="panel-foot">Selected-step error is V(s) − G(s): positive overestimates, negative underestimates. G(s) is available after the episode ends.</p>
        </section>
        <section class="reward-plot" aria-labelledby="reward-plot-title">
          <div class="chart-heading"><span id="reward-plot-title">After-action step reward</span></div>
          <canvas data-reward-chart class="chart reward-chart" aria-label="Provider and shaped step reward history"></canvas>
          <div data-reward-legend class="legend"></div>
        </section>
        <section class="reward-plot" aria-labelledby="return-plot-title">
          <div class="chart-heading"><span id="return-plot-title">Episode return</span></div>
          <canvas data-return-chart class="chart reward-chart" aria-label="Episode return history"></canvas>
          <div data-return-legend class="legend"></div>
        </section>
      </div>
    `,
  });
  const rewardChart = element.querySelector("[data-reward-chart]");
  const returnChart = element.querySelector("[data-return-chart]");
  const valueChart = element.querySelector("[data-value-chart]");
  const valueStatus = element.querySelector("[data-value-status]");
  let history = [];
  let view = {};
  let snapshot = null;

  const selectedPoint = () => {
    const sequence = view.selectedSequence ?? snapshot?.transition?.sequence;
    return history.find((point) => Number(point.sequence) === Number(sequence))
      || history.at(-1)
      || null;
  };

  const renderStats = () => {
    const transition = snapshot?.transition;
    const reward = transition?.reward || {};
    const point = selectedPoint();
    setStats(element.querySelector("[data-stats]"), [
      ["Provider r", number(reward.provider, 3)],
      ["Shaped r", number(reward.shaped, 3)],
      ["Return", number(reward.return, 2)],
      ["V(s)", number(point?.value ?? transition?.decision?.value, 3)],
      ["Realized G(s)", number(point?.realized_return, 3)],
      ["V − G", number(point?.value_error, 3)],
      ["Outcome", transition?.outcome || "continuing"],
    ]);
  };

  const renderHistory = (next, nextSnapshot = snapshot, nextView = view) => {
    history = next;
    snapshot = nextSnapshot;
    view = nextView || {};
    const points = history;
    const cursorIndex = view.inspection
      ? points.findIndex((point) => Number(point.sequence) === Number(view.selectedSequence))
      : null;
    drawLines(rewardChart, [
      { values: points.map((point) => Number(point.reward_provider)), color: "#76a9ff" },
      { values: points.map((point) => Number(point.reward_shaped)), color: "#d794ff" },
    ], { cursorIndex });
    drawLines(returnChart, [
      { values: points.map((point) => Number(point.return)), color: "#60d394" },
    ], { cursorIndex });
    drawLines(valueChart, [
      {
        values: points.map((point) => (
          point.value === null || point.value === undefined
            ? Number.NaN
            : Number(point.value)
        )),
        color: "#76a9ff",
      },
      {
        values: points.map((point) => (
          point.realized_return === null || point.realized_return === undefined
            ? Number.NaN
            : Number(point.realized_return)
        )),
        color: "#f0c36a",
      },
    ], { cursorIndex });
    legend(element.querySelector("[data-reward-legend]"), [
      ["Provider reward", "#76a9ff"], ["Shaped reward", "#d794ff"],
    ]);
    legend(element.querySelector("[data-return-legend]"), [["Return", "#60d394"]]);
    legend(element.querySelector("[data-value-legend]"), [
      ["V(s)", "#76a9ff"], ["Realized G(s)", "#f0c36a"],
    ]);
    const discount = snapshot?.session?.value_discount;
    const hasDiscount = discount !== null
      && discount !== undefined
      && Number.isFinite(Number(discount));
    valueStatus.textContent = hasDiscount
      ? `γ ${Number(discount).toFixed(3)}`
      : "value target unavailable";
    renderStats();
  };

  return {
    element,
    render(nextSnapshot) {
      snapshot = nextSnapshot;
      renderStats();
    },
    renderHistory,
    resize() { renderHistory(history, snapshot, view); },
  };
}
