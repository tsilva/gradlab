import { rewardContribution } from "./reward-discount.js";
import { themeColor } from "./shared.js";

// Keep the table bounded even when the episode chart contains thousands of samples.
// Anchor rows to playback selection, never pointer hover.
// Include the exact selected transition when chart downsampling omitted it.
export function rewardInspectionRows(history, selected, gamma, limit = 5, referenceStep = selected?.step) {
  const points = new Map(history.map((point) => [point.step, point]));
  if (selected && history.length && selected.step >= history[0].step
      && selected.step <= history.at(-1).step) points.set(selected.step, { ...points.get(selected.step), ...selected });
  const firstStep = Math.max(selected?.step ?? -Infinity, referenceStep ?? -Infinity);
  const candidates = [...points.values()].filter((point) => (
    point.step >= firstStep && (point.step === selected?.step
    || [point.reward_provider, point.reward_shaped].some((value) => Number.isFinite(value) && value !== 0))
  )).sort((a, b) => a.step - b.step);
  if (!candidates.length) return [];
  const focus = selected?.step;
  let nearest = 0;
  candidates.forEach((point, index) => {
    if (Math.abs(point.step - focus) < Math.abs(candidates[nearest].step - focus)) nearest = index;
  });
  const start = Math.max(0, Math.min(nearest - Math.floor(limit / 2), candidates.length - limit));
  return candidates.slice(start, start + limit).map((point) => {
    const result = rewardContribution(point, referenceStep, gamma);
    const delay = Number.isInteger(referenceStep) ? point.step - referenceStep : null;
    const weight = delay !== null && delay >= 0 && Number.isFinite(gamma) && gamma >= 0 && gamma <= 1
      ? gamma ** delay : null;
    return { ...point, delay, weight, contribution: result?.contribution ?? null,
      past: delay !== null && delay < 0, inspected: point.step === focus };
  });
}

export function formatRewardCell(value, digits = 3) {
  if (!Number.isFinite(value)) return "—";
  if (value !== 0 && Math.abs(value) < 10 ** -digits) return value.toExponential(2);
  return value.toFixed(digits);
}

export function createRewardInspector(legend, services) {
  legend.classList.add("reward-history-legend");
  legend.replaceChildren(...[
    ["Native reward", "seriesViolet"],
    ["Shaped reward", "seriesTeal"],
    ["Discounted contribution", "seriesAmber"],
  ].map(([label, color], index) => {
    const item = document.createElement("span");
    item.textContent = label;
    item.style.setProperty("--legend-color", themeColor(color));
    if (index === 2) item.className = "discounted-series";
    return item;
  }));
  const scroll = document.createElement("div");
  scroll.className = "reward-history-scroll";
  const table = document.createElement("table");
  table.className = "reward-history-table";
  table.setAttribute("aria-label", "Recorded rewards near the selected playback step");
  const head = table.createTHead().insertRow();
  ["Step", "Native", "Shaped", "Delay", "Weight", "Contribution", "Return G", "Value V"].forEach((label) => {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = label;
    head.append(cell);
  });
  const body = table.createTBody();
  scroll.append(table);
  let signature = "";
  return {
    element: scroll,
    render(history, selected, gamma, referenceStep = selected?.step) {
      const rows = rewardInspectionRows(history, selected, gamma, 5, referenceStep);
      const next = JSON.stringify([rows, selected?.step, gamma, referenceStep]);
      if (next === signature) return;
      signature = next;
      const focusedStep = table.contains(document.activeElement)
        ? document.activeElement.dataset.rewardStep : null;
      table.setAttribute("aria-description", `Up to five current and future recorded reward samples. Discounted shaped rewards from step ${referenceStep ?? "unavailable"}, gamma ${gamma ?? "unavailable"}. Past rewards are excluded. Discounting does not imply causation.`);
      body.replaceChildren(...rows.map((point) => {
        const row = document.createElement("tr");
        row.classList.toggle("is-inspected", point.inspected);
        const step = document.createElement("th");
        step.scope = "row";
        const button = document.createElement("button");
        button.type = "button";
        button.className = "reward-step-button";
        button.textContent = String(point.step);
        button.dataset.rewardStep = String(point.step);
        button.setAttribute("aria-label", `Inspect step ${point.step}`);
        if (point.inspected) button.setAttribute("aria-current", "step");
        button.addEventListener("click", () => {
          if (services.inspectStep) services.inspectStep(point.step);
          else services.inspectSequence?.(point.sequence);
        });
        step.append(button);
        row.append(step);
        const values = [formatRewardCell(point.reward_provider), formatRewardCell(point.reward_shaped),
          point.delay ?? "—", formatRewardCell(point.weight, 5), formatRewardCell(point.contribution, 5),
          point.value_comparison_reasons?.length ? "Incomparable"
            : Number.isFinite(point.realized_return)
              ? `${formatRewardCell(point.realized_return)}${point.realized_return_bootstrapped ? " (bootstrapped)" : ""}`
              : "Pending",
          formatRewardCell(point.value)];
        values.forEach((value, index) => {
          const cell = document.createElement("td");
          cell.textContent = String(value);
          if (value === "—") cell.title = point.past && index >= 3
            ? "Past reward: excluded from future contribution"
            : "Unavailable: required recorded data is missing";
          if (index === 5) cell.title = point.value_comparison_reasons?.join("; ")
            || "Discounted return from this row’s step; pending until comparable episode evidence is available. Bootstrapped returns include the final state value.";
          if (index === 6) cell.title = "Recorded pre-action critic prediction V(s) for this row’s state";
          row.append(cell);
        });
        return row;
      }));
      if (!rows.length) {
        const cell = body.insertRow().insertCell();
        cell.colSpan = 8;
        cell.textContent = "No recorded rewards in this window";
      }
      if (focusedStep !== null) {
        const buttons = [...body.querySelectorAll("button")];
        (buttons.find((button) => button.dataset.rewardStep === focusedStep)
          || buttons.find((button) => button.getAttribute("aria-current") === "step"))?.focus({ preventScroll: true });
      }
    },
  };
}
