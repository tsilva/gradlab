function appendToken(tokens, text, className = "") {
  if (!text) return;
  const previous = tokens.at(-1);
  if (previous?.className === className) {
    previous.text += text;
    return;
  }
  tokens.push({ text, className });
}

function quotedEnd(line, start, quote) {
  let cursor = start + 1;
  while (cursor < line.length) {
    if (quote === "'" && line[cursor] === "'" && line[cursor + 1] === "'") {
      cursor += 2;
      continue;
    }
    if (quote === '"' && line[cursor] === "\\") {
      cursor += 2;
      continue;
    }
    if (line[cursor] === quote) return cursor + 1;
    cursor += 1;
  }
  return line.length;
}

function mappingColon(line, start) {
  let quote = "";
  let depth = 0;
  for (let cursor = start; cursor < line.length; cursor += 1) {
    const character = line[cursor];
    if (quote) {
      if (quote === "'" && character === "'" && line[cursor + 1] === "'") {
        cursor += 1;
      } else if (quote === '"' && character === "\\") {
        cursor += 1;
      } else if (character === quote) {
        quote = "";
      }
      continue;
    }
    if (character === "'" || character === '"') {
      quote = character;
      continue;
    }
    if (character === "[" || character === "{") {
      depth += 1;
      continue;
    }
    if (character === "]" || character === "}") {
      depth = Math.max(0, depth - 1);
      continue;
    }
    if (character === "#" && (cursor === 0 || /\s/.test(line[cursor - 1]))) {
      return -1;
    }
    if (
      character === ":"
      && depth === 0
      && (cursor + 1 === line.length || /\s/.test(line[cursor + 1]))
    ) {
      return cursor;
    }
  }
  return -1;
}

function tokenizeScalar(line, start, tokens) {
  let cursor = start;
  while (cursor < line.length) {
    const rest = line.slice(cursor);
    const whitespace = rest.match(/^\s+/);
    if (whitespace) {
      appendToken(tokens, whitespace[0]);
      cursor += whitespace[0].length;
      continue;
    }
    if (line[cursor] === "#" && (cursor === 0 || /\s/.test(line[cursor - 1]))) {
      appendToken(tokens, line.slice(cursor), "syntax-comment");
      return;
    }
    if (line[cursor] === "'" || line[cursor] === '"') {
      const end = quotedEnd(line, cursor, line[cursor]);
      appendToken(tokens, line.slice(cursor, end), "syntax-string");
      cursor = end;
      continue;
    }
    const reference = rest.match(/^[&*!][A-Za-z0-9_.:/-]+/);
    if (reference) {
      appendToken(tokens, reference[0], "syntax-reference");
      cursor += reference[0].length;
      continue;
    }
    const keyword = rest.match(/^(?:true|false|null|~)(?=$|[\s,\[\]{}#])/i);
    if (keyword) {
      const className = /^(?:null|~)$/i.test(keyword[0])
        ? "syntax-null"
        : "syntax-boolean";
      appendToken(tokens, keyword[0], className);
      cursor += keyword[0].length;
      continue;
    }
    const number = rest.match(
      /^[+-]?(?:(?:0x[0-9a-f_]+)|(?:0o[0-7_]+)|(?:(?:\d[\d_]*)?\.\d[\d_]*(?:e[+-]?\d+)?|\d[\d_]*(?:e[+-]?\d+)?))(?=$|[\s,\[\]{}#])/i,
    );
    if (number) {
      appendToken(tokens, number[0], "syntax-number");
      cursor += number[0].length;
      continue;
    }
    if ("[]{},".includes(line[cursor])) {
      appendToken(tokens, line[cursor], "syntax-punctuation");
      cursor += 1;
      continue;
    }
    if (
      (line[cursor] === "|" || line[cursor] === ">")
      && (cursor + 1 === line.length || /[+\-\d]/.test(line[cursor + 1]))
    ) {
      appendToken(tokens, line.slice(cursor), "syntax-directive");
      return;
    }
    const plain = rest.match(/^[^\s#[\]{},'"]+/);
    if (plain) {
      appendToken(tokens, plain[0], "syntax-string");
      cursor += plain[0].length;
      continue;
    }
    appendToken(tokens, line[cursor]);
    cursor += 1;
  }
}

function tokenizeYamlLine(line, tokens) {
  if (/^\s*(?:---|\.\.\.|%YAML(?:\s|$)|%TAG(?:\s|$))/.test(line)) {
    appendToken(tokens, line, "syntax-directive");
    return;
  }

  const indentation = line.match(/^\s*/)?.[0] || "";
  appendToken(tokens, indentation);
  let cursor = indentation.length;
  if (line[cursor] === "-" && (cursor + 1 === line.length || /\s/.test(line[cursor + 1]))) {
    appendToken(tokens, "-", "syntax-punctuation");
    cursor += 1;
    const whitespace = line.slice(cursor).match(/^\s+/)?.[0] || "";
    appendToken(tokens, whitespace);
    cursor += whitespace.length;
  }

  const colon = mappingColon(line, cursor);
  if (colon >= cursor) {
    const key = line.slice(cursor, colon);
    const trailingWhitespace = key.match(/\s*$/)?.[0] || "";
    appendToken(tokens, key.slice(0, key.length - trailingWhitespace.length), "syntax-key");
    appendToken(tokens, trailingWhitespace);
    appendToken(tokens, ":", "syntax-punctuation");
    tokenizeScalar(line, colon + 1, tokens);
    return;
  }
  tokenizeScalar(line, cursor, tokens);
}

function tokenizeYaml(value) {
  const tokens = [];
  const lines = value.match(/[^\n]*(?:\n|$)/g) || [];
  lines.forEach((lineWithEnding) => {
    if (!lineWithEnding) return;
    const hasNewline = lineWithEnding.endsWith("\n");
    const line = hasNewline ? lineWithEnding.slice(0, -1) : lineWithEnding;
    tokenizeYamlLine(line, tokens);
    if (hasNewline) appendToken(tokens, "\n");
  });
  return tokens;
}

function tokenizeDiff(value) {
  const tokens = [];
  const lines = value.match(/[^\n]*(?:\n|$)/g) || [];
  lines.forEach((lineWithEnding) => {
    if (!lineWithEnding) return;
    const hasNewline = lineWithEnding.endsWith("\n");
    const line = hasNewline ? lineWithEnding.slice(0, -1) : lineWithEnding;
    let className = "";
    if (/^(?:---|\+\+\+) /.test(line)) className = "syntax-diff-metadata";
    else if (line.startsWith("@@")) className = "syntax-diff-hunk";
    else if (line.startsWith("+")) className = "syntax-diff-added";
    else if (line.startsWith("-")) className = "syntax-diff-removed";
    appendToken(tokens, line, className);
    if (hasNewline) appendToken(tokens, "\n", className);
  });
  return tokens;
}

export function contractSyntaxTokens(value, view) {
  return view === "changes" ? tokenizeDiff(value) : tokenizeYaml(value);
}

export function contractSearchRanges(value, query) {
  if (!value || !query) return [];
  const ranges = [];
  const normalizedValue = value.toLocaleLowerCase();
  const normalizedQuery = query.toLocaleLowerCase();
  let cursor = 0;
  while (cursor < value.length) {
    const start = normalizedValue.indexOf(normalizedQuery, cursor);
    if (start < 0) break;
    ranges.push({ start, end: start + query.length });
    cursor = start + query.length;
  }
  return ranges;
}
