const HUNK_HEADER = /^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?:.*)$/;

function documentLines(value) {
  if (!value) return [];
  const lines = String(value).split("\n");
  if (lines.at(-1) === "") lines.pop();
  return lines.map((line) => line.endsWith("\r") ? line.slice(0, -1) : line);
}

function hunkStart(value) {
  const parsed = Number(value);
  return parsed === 0 ? 0 : parsed - 1;
}

function hunkCount(value) {
  return value === undefined ? 1 : Number(value);
}

function comparisonError(message) {
  return new Error(`Contract diff is inconsistent: ${message}`);
}

function parseHunks(value) {
  const lines = documentLines(value);
  const hunks = [];
  let cursor = 0;
  while (cursor < lines.length) {
    const match = lines[cursor].match(HUNK_HEADER);
    if (!match) {
      if (
        lines[cursor]
        && !lines[cursor].startsWith("--- ")
        && !lines[cursor].startsWith("+++ ")
      ) {
        throw comparisonError(`unsupported diff line ${JSON.stringify(lines[cursor])}`);
      }
      cursor += 1;
      continue;
    }
    const hunk = {
      baseStart: hunkStart(match[1]),
      baseCount: hunkCount(match[2]),
      resolvedStart: hunkStart(match[3]),
      resolvedCount: hunkCount(match[4]),
      lines: [],
    };
    cursor += 1;
    while (cursor < lines.length && !HUNK_HEADER.test(lines[cursor])) {
      const line = lines[cursor];
      if (line === "\\ No newline at end of file") {
        cursor += 1;
        continue;
      }
      const marker = line[0];
      if (marker !== " " && marker !== "+" && marker !== "-") {
        throw comparisonError(`unsupported hunk line ${JSON.stringify(line)}`);
      }
      hunk.lines.push({ marker, text: line.slice(1) });
      cursor += 1;
    }
    hunks.push(hunk);
  }
  return hunks;
}

function inlineRanges(before, after) {
  if (before === after) return [[], []];
  let prefix = 0;
  const sharedLength = Math.min(before.length, after.length);
  while (prefix < sharedLength && before[prefix] === after[prefix]) prefix += 1;

  let suffix = 0;
  while (
    suffix < sharedLength - prefix
    && before[before.length - suffix - 1] === after[after.length - suffix - 1]
  ) {
    suffix += 1;
  }
  const beforeEnd = before.length - suffix;
  const afterEnd = after.length - suffix;
  return [
    beforeEnd > prefix ? [{ start: prefix, end: beforeEnd }] : [],
    afterEnd > prefix ? [{ start: prefix, end: afterEnd }] : [],
  ];
}

function line(number, text, change, emphasis = []) {
  return { number, text, change, emphasis };
}

function contextRow(baseNumber, resolvedNumber, text) {
  return {
    kind: "context",
    base: line(baseNumber, text, "context"),
    resolved: line(resolvedNumber, text, "context"),
  };
}

function changedRows(baseItems, resolvedItems) {
  const rows = [];
  const count = Math.max(baseItems.length, resolvedItems.length);
  for (let index = 0; index < count; index += 1) {
    const before = baseItems[index] || null;
    const after = resolvedItems[index] || null;
    const [beforeRanges, afterRanges] = before && after
      ? inlineRanges(before.text, after.text)
      : [
          before?.text ? [{ start: 0, end: before.text.length }] : [],
          after?.text ? [{ start: 0, end: after.text.length }] : [],
        ];
    rows.push({
      kind: "change",
      base: before
        ? line(before.number, before.text, "removed", beforeRanges)
        : null,
      resolved: after
        ? line(after.number, after.text, "added", afterRanges)
        : null,
    });
  }
  return rows;
}

function assertLine(lines, index, expected, label) {
  if (index >= lines.length) {
    throw comparisonError(`${label} line ${index + 1} is missing`);
  }
  if (lines[index] !== expected) {
    throw comparisonError(`${label} line ${index + 1} does not match its hunk`);
  }
}

function appendUnchanged(
  rows,
  baseLines,
  resolvedLines,
  baseCursor,
  resolvedCursor,
  baseEnd,
  resolvedEnd,
) {
  if (
    baseEnd < baseCursor
    || resolvedEnd < resolvedCursor
    || baseEnd > baseLines.length
    || resolvedEnd > resolvedLines.length
  ) {
    throw comparisonError("an unchanged range is outside its document");
  }
  const baseCount = baseEnd - baseCursor;
  const resolvedCount = resolvedEnd - resolvedCursor;
  if (baseCount !== resolvedCount) {
    throw comparisonError("unchanged ranges have different lengths");
  }
  for (let offset = 0; offset < baseCount; offset += 1) {
    const baseIndex = baseCursor + offset;
    const resolvedIndex = resolvedCursor + offset;
    if (baseLines[baseIndex] !== resolvedLines[resolvedIndex]) {
      throw comparisonError("a change is missing from the unified diff");
    }
    rows.push(contextRow(baseIndex + 1, resolvedIndex + 1, baseLines[baseIndex]));
  }
}

export function buildSideBySideRows({ baseText, resolvedText, unifiedDiff }) {
  const baseLines = documentLines(baseText);
  const resolvedLines = documentLines(resolvedText);
  const hunks = parseHunks(unifiedDiff);
  if (!hunks.length && (unifiedDiff || baseText !== resolvedText)) {
    throw comparisonError("comparison has no diff hunks");
  }

  const rows = [];
  let baseCursor = 0;
  let resolvedCursor = 0;
  hunks.forEach((hunk) => {
    if (hunk.baseStart < baseCursor || hunk.resolvedStart < resolvedCursor) {
      throw comparisonError("diff hunks overlap or are out of order");
    }
    appendUnchanged(
      rows,
      baseLines,
      resolvedLines,
      baseCursor,
      resolvedCursor,
      hunk.baseStart,
      hunk.resolvedStart,
    );
    baseCursor = hunk.baseStart;
    resolvedCursor = hunk.resolvedStart;
    const baseHunkStart = baseCursor;
    const resolvedHunkStart = resolvedCursor;

    let hunkCursor = 0;
    while (hunkCursor < hunk.lines.length) {
      const item = hunk.lines[hunkCursor];
      if (item.marker === " ") {
        assertLine(baseLines, baseCursor, item.text, "base");
        assertLine(resolvedLines, resolvedCursor, item.text, "resolved");
        rows.push(contextRow(baseCursor + 1, resolvedCursor + 1, item.text));
        baseCursor += 1;
        resolvedCursor += 1;
        hunkCursor += 1;
        continue;
      }

      const removed = [];
      while (hunk.lines[hunkCursor]?.marker === "-") {
        const removedLine = hunk.lines[hunkCursor].text;
        assertLine(baseLines, baseCursor, removedLine, "base");
        removed.push({ number: baseCursor + 1, text: removedLine });
        baseCursor += 1;
        hunkCursor += 1;
      }
      const added = [];
      while (hunk.lines[hunkCursor]?.marker === "+") {
        const addedLine = hunk.lines[hunkCursor].text;
        assertLine(resolvedLines, resolvedCursor, addedLine, "resolved");
        added.push({ number: resolvedCursor + 1, text: addedLine });
        resolvedCursor += 1;
        hunkCursor += 1;
      }
      if (!removed.length && !added.length) {
        throw comparisonError("change block has no additions or removals");
      }
      rows.push(...changedRows(removed, added));
    }

    if (baseCursor - baseHunkStart !== hunk.baseCount) {
      throw comparisonError("base hunk length does not match its header");
    }
    if (resolvedCursor - resolvedHunkStart !== hunk.resolvedCount) {
      throw comparisonError("resolved hunk length does not match its header");
    }
  });

  appendUnchanged(
    rows,
    baseLines,
    resolvedLines,
    baseCursor,
    resolvedCursor,
    baseLines.length,
    resolvedLines.length,
  );
  return rows;
}

export function sideBySideSearchCounts(rows, query) {
  const counts = { base: 0, resolved: 0 };
  if (!query) return counts;
  const normalizedQuery = query.toLocaleLowerCase();
  rows.forEach((row) => {
    ["base", "resolved"].forEach((side) => {
      const value = row[side]?.text.toLocaleLowerCase() || "";
      let cursor = 0;
      while (cursor < value.length) {
        const start = value.indexOf(normalizedQuery, cursor);
        if (start < 0) break;
        counts[side] += 1;
        cursor = start + query.length;
      }
    });
  });
  return counts;
}
