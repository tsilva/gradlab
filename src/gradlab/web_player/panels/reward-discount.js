// Contributions are relative to the selected pre-action state, never episode end.
export function rewardContribution(point, selectedStep, gamma) {
  if (!Number.isInteger(selectedStep) || !Number.isInteger(point?.step)
      || typeof gamma !== "number" || !Number.isFinite(gamma) || gamma < 0 || gamma > 1) return null;
  const delay = point.step - selectedStep;
  if (delay < 0) return { past: true, delay };
  if (typeof point.reward_shaped !== "number" || !Number.isFinite(point.reward_shaped)) return null;
  const weight = gamma ** delay;
  return { past: false, delay, reward: point.reward_shaped, weight, contribution: point.reward_shaped * weight };
}

export function contributionLabel(point, selectedStep, gamma) {
  const result = rewardContribution(point, selectedStep, gamma);
  if (!result) return "Discounted contribution unavailable: recorded reward or discount missing.";
  if (result.past) return `Step ${point.step}: past—excluded from future return.`;
  const format = (value) => Number(value.toPrecision(4)).toString();
  return `Step ${point.step}: ${format(result.reward)} reward · ${result.delay} steps later · weight ${format(result.weight)} · contributes ${format(result.contribution)}`;
}
