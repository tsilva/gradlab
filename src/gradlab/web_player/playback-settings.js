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

const COMPARISON_OPERATORS = ["==", "!=", "<", "<=", ">", ">="];

export function stopConditionSuggestions(source, cursor, symbols = []) {
  const text = String(source || "");
  const end = Math.max(0, Math.min(text.length, Number(cursor) || 0));
  const before = text.slice(0, end);
  const fragment = before.match(/[A-Za-z_][A-Za-z0-9_.-]*$/)?.[0] || "";
  const from = end - fragment.length;
  const suffix = text.slice(end).match(/^[A-Za-z0-9_.-]*/)?.[0] || "";
  const prefix = before.slice(0, from).trimEnd();
  const comparison = /(?:^|\(|\band\b|\bor\b)\s*[A-Za-z_][A-Za-z0-9_.-]*\s*(?:==|!=|<=|>=|<|>)\s*[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?\s*$/;
  let candidates;
  if (comparison.test(before)) {
    candidates = ["and", "or"].map((value) => ({ value, kind: "keyword" }));
  } else if (/(?:^|\(|\band\b|\bor\b)\s*[A-Za-z_][A-Za-z0-9_.-]*\s*$/.test(before)
    && !/(?:^|\(|\band\b|\bor\b)\s*$/.test(prefix)) {
    candidates = COMPARISON_OPERATORS.map((value) => ({ value, kind: "operator" }));
  } else {
    candidates = (Array.isArray(symbols) ? symbols : [])
      .filter((symbol) => symbol && typeof symbol.name === "string")
      .map((symbol) => ({
        value: symbol.name,
        kind: symbol.kind || "symbol",
        detail: symbol.available === false ? "unavailable" : symbol.value,
      }));
  }
  const normalized = fragment.toLowerCase();
  const items = candidates.filter((item) => (
    !normalized || item.value.toLowerCase().startsWith(normalized)
  ));
  return { from, to: end + suffix.length, items };
}

export function applyStopConditionSuggestion(source, cursor, from, to, value) {
  const text = String(source || "");
  const start = Math.max(0, Math.min(text.length, Number(from) || 0));
  const requestedEnd = Number.isFinite(Number(to)) ? Number(to) : Number(cursor) || 0;
  const end = Math.max(start, Math.min(text.length, requestedEnd));
  const insertion = String(value || "");
  return {
    source: `${text.slice(0, start)}${insertion}${text.slice(end)}`,
    cursor: start + insertion.length,
  };
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
