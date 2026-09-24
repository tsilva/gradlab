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
const MATCH_COMPARISON_OPERATORS = ["==", "!=", "<=", ">=", "<", ">"];
const IDENTIFIER = /[A-Za-z_][A-Za-z0-9_-]*(?:\.[A-Za-z0-9_-]+)*/y;
const NUMBER = /[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?/y;

export function stopConditionSuggestions(source, cursor, symbols = []) {
  const text = String(source || "");
  const end = Math.max(0, Math.min(text.length, Number(cursor) || 0));
  const before = text.slice(0, end);
  const fragment = before.match(/[A-Za-z_][A-Za-z0-9_.-]*$/)?.[0] || "";
  const from = end - fragment.length;
  const suffix = text.slice(end).match(/^[A-Za-z0-9_.-]*/)?.[0] || "";
  const prefix = before.slice(0, from).trimEnd();
  const comparison = /(?:^|\(|\band\b|\bor\b)\s*[A-Za-z_][A-Za-z0-9_.-]*\s*(?:==|!=|<=|>=|<|>)\s*[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?\s*$/;
  const awaitingNumber = /(?:^|\(|\band\b|\bor\b)\s*[A-Za-z_][A-Za-z0-9_.-]*\s*(?:==|!=|<=|>=|<|>)\s*[^\s()]*\s*$/;
  let candidates;
  if (comparison.test(before)) {
    candidates = ["and", "or"].map((value) => ({ value, kind: "keyword" }));
  } else if (awaitingNumber.test(before)) {
    candidates = [];
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

function tokenAt(source, offset, symbolKinds) {
  const whitespace = source.slice(offset).match(/^\s+/)?.[0];
  if (whitespace) return { text: whitespace, kind: "plain" };
  const operator = MATCH_COMPARISON_OPERATORS.find((value) => source.startsWith(value, offset));
  if (operator) return { text: operator, kind: "operator" };
  if (source[offset] === "(" || source[offset] === ")") {
    return { text: source[offset], kind: "parenthesis" };
  }
  IDENTIFIER.lastIndex = offset;
  const identifier = IDENTIFIER.exec(source);
  if (identifier) {
    const text = identifier[0];
    if (text === "and" || text === "or") return { text, kind: "keyword" };
    return { text, kind: `symbol-${symbolKinds.get(text) || "unknown"}` };
  }
  NUMBER.lastIndex = offset;
  const number = NUMBER.exec(source);
  if (number) return { text: number[0], kind: "number" };
  return { text: source[offset], kind: "invalid" };
}

export function stopConditionHighlightSegments(source, error = null, symbols = []) {
  const text = String(source || "");
  const symbolKinds = new Map(
    (Array.isArray(symbols) ? symbols : [])
      .filter((symbol) => symbol && typeof symbol.name === "string")
      .map((symbol) => [symbol.name, symbol.kind || "unknown"]),
  );
  const raw = [];
  let offset = 0;
  while (offset < text.length) {
    const token = tokenAt(text, offset, symbolKinds);
    raw.push({ ...token, from: offset, to: offset + token.text.length });
    offset += token.text.length;
  }

  const errorStart = error
    ? Math.max(0, Math.min(text.length, Number(error.offset) || 0))
    : null;
  const errorLength = error
    ? Math.max(0, Math.min(text.length - errorStart, Number(error.length) || 0))
    : 0;
  const errorEnd = errorStart == null ? null : errorStart + errorLength;
  const segments = [];
  let markerInserted = false;
  for (const token of raw) {
    const boundaries = [token.from, token.to];
    if (errorLength > 0) {
      if (errorStart > token.from && errorStart < token.to) boundaries.push(errorStart);
      if (errorEnd > token.from && errorEnd < token.to) boundaries.push(errorEnd);
    } else if (
      errorStart != null
      && errorStart > token.from
      && errorStart < token.to
    ) {
      boundaries.push(errorStart);
    }
    boundaries.sort((left, right) => left - right);
    for (let index = 0; index < boundaries.length - 1; index += 1) {
      const from = boundaries[index];
      const to = boundaries[index + 1];
      if (!markerInserted && errorLength === 0 && errorStart === from) {
        segments.push({ text: "", kind: "error-marker", error: true, from, to: from });
        markerInserted = true;
      }
      segments.push({
        text: text.slice(from, to),
        kind: token.kind,
        error: errorLength > 0 && from < errorEnd && to > errorStart,
        from,
        to,
      });
    }
  }
  if (!markerInserted && errorLength === 0 && errorStart != null) {
    segments.push({
      text: "",
      kind: "error-marker",
      error: true,
      from: errorStart,
      to: errorStart,
    });
  }
  return segments;
}

export function stopConditionPopoverPlacement(rect, viewport = {}) {
  const viewportWidth = Math.max(1, Number(viewport.width) || 1);
  const gutter = 8;
  const gap = 6;
  const preferredHeight = 240;
  const above = Math.max(0, rect.top - gap - gutter);
  const maxHeight = Math.max(64, Math.min(preferredHeight, above));
  const width = Math.max(
    1,
    Math.min(Number(rect.width) || 1, viewportWidth - gutter * 2),
  );
  const left = Math.min(
    Math.max(gutter, Number(rect.left) || 0),
    viewportWidth - gutter - width,
  );
  const top = Number(rect.top) - gap;
  return { left, top, width, maxHeight, placement: "above" };
}
