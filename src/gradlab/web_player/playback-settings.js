import { text } from "./panels/shared.js";

function selectionLabel(mode) {
  return ({
    stochastic: "Stochastic",
    deterministic: "Deterministic",
    epsilon_greedy: "Epsilon-greedy",
    greedy: "Greedy",
    program: "Program",
    route: "Route",
  })[mode] || String(mode || "").replaceAll("_", " ");
}

const TERMINATION_OUTCOME_ORDER = new Map([
  ["success", 0],
  ["failure", 1],
  ["timeout", 2],
]);

export function orderedTerminationConditions(conditions) {
  return (Array.isArray(conditions) ? conditions : [])
    .map((condition, index) => ({ condition, index }))
    .sort((left, right) => (
      (TERMINATION_OUTCOME_ORDER.get(String(left.condition?.outcome || "").toLowerCase()) ?? 3)
      - (TERMINATION_OUTCOME_ORDER.get(String(right.condition?.outcome || "").toLowerCase()) ?? 3)
      || left.index - right.index
    ))
    .map(({ condition }) => condition);
}

export function terminationOutcomeClass(outcome) {
  const normalized = String(outcome || "").toLowerCase();
  return TERMINATION_OUTCOME_ORDER.has(normalized) ? `outcome-${normalized}` : "";
}

export function frameSkipPresentation(playbackContract) {
  const values = playbackContract?.frame_skip;
  if (!values || typeof values !== "object") return null;
  const training = Number(values.training);
  const playback = Number(values.playback);
  if (
    !Number.isInteger(training)
    || training < 1
    || !Number.isInteger(playback)
    || playback < 1
  ) return null;
  return {
    training,
    playback,
    differs: training !== playback,
    label: `Frame skip · training ${training} · playback ${playback}`,
  };
}
