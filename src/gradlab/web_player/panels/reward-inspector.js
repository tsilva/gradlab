import { rewardContribution } from "./reward-discount.js";

// Keep the table bounded even when the episode chart contains thousands of samples.
// Anchor row membership to the return reference; seeking only changes highlights.
// Restore the exact reference transition when chart downsampling omitted it.
export function rewardInspectionRows(history, selected, gamma, limit = 5, referenceStep = selected?.step) {
  const points = new Map(history.map((point) => [point.step, point]));
  if (selected && selected.step === referenceStep && history.length && selected.step >= history[0].step
      && selected.step <= history.at(-1).step) points.set(selected.step, { ...points.get(selected.step), ...selected });
  const firstStep = referenceStep ?? -Infinity;
  const candidates = [...points.values()].filter((point) => (
    point.step >= firstStep && (point.step === referenceStep
    || [point.reward_provider, point.reward_shaped].some((value) => Number.isFinite(value) && value !== 0))
  )).sort((a, b) => a.step - b.step);
  if (!candidates.length) return [];
  const focus = selected?.step;
  return candidates.slice(0, limit).map((point) => {
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

export const REWARD_LABELS = Object.freeze({
  "reward/provider": "Native reward",
  "reward/shaped": "Shaped reward",
});

