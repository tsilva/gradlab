import assert from "node:assert/strict";
import test from "node:test";

import {
  buildSideBySideRows,
  sideBySideSearchCounts,
} from "../../src/gradlab/web_player/documents/diff.js";

test("full-file diff aligns replacements and preserves all context", () => {
  const rows = buildSideBySideRows({
    baseText: "alpha\nmax_steps: 1050\nkeep: true\nremove: yes\nomega\n",
    resolvedText: "alpha\nmax_steps: 525\nkeep: true\nadd: yes\nomega\n",
    unifiedDiff: [
      "--- goal-base.yaml",
      "+++ goal-resolved.yaml",
      "@@ -1,5 +1,5 @@",
      " alpha",
      "-max_steps: 1050",
      "+max_steps: 525",
      " keep: true",
      "-remove: yes",
      "+add: yes",
      " omega",
      "",
    ].join("\n"),
  });

  assert.equal(rows.length, 5);
  assert.deepEqual(rows.map((row) => row.kind), [
    "context", "change", "context", "change", "context",
  ]);
  assert.deepEqual(rows[1], {
    kind: "change",
    base: {
      number: 2,
      text: "max_steps: 1050",
      change: "removed",
      emphasis: [{ start: 11, end: 15 }],
    },
    resolved: {
      number: 2,
      text: "max_steps: 525",
      change: "added",
      emphasis: [{ start: 11, end: 14 }],
    },
  });
  assert.equal(rows.at(-1).base.number, 5);
  assert.equal(rows.at(-1).resolved.number, 5);
});

test("unequal replacement blocks create aligned spacer rows", () => {
  const rows = buildSideBySideRows({
    baseText: "a\nb\nc\nd\n",
    resolvedText: "a\nx\ny\nz\nd\n",
    unifiedDiff: [
      "--- recipe-base.yaml",
      "+++ recipe-resolved.yaml",
      "@@ -1,4 +1,5 @@",
      " a",
      "-b",
      "-c",
      "+x",
      "+y",
      "+z",
      " d",
      "",
    ].join("\n"),
  });

  assert.equal(rows.length, 5);
  assert.deepEqual(rows[3], {
    kind: "change",
    base: null,
    resolved: {
      number: 4,
      text: "z",
      change: "added",
      emphasis: [{ start: 0, end: 1 }],
    },
  });
  assert.equal(rows[4].base.number, 4);
  assert.equal(rows[4].resolved.number, 5);
});

test("multiple hunks retain complete unchanged ranges between and around them", () => {
  const base = [
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
  ].join("\n") + "\n";
  const resolved = base.replace("two", "TWO").replace("nine", "NINE");
  const rows = buildSideBySideRows({
    baseText: base,
    resolvedText: resolved,
    unifiedDiff: [
      "--- goal-base.yaml",
      "+++ goal-resolved.yaml",
      "@@ -1,4 +1,4 @@",
      " one",
      "-two",
      "+TWO",
      " three",
      " four",
      "@@ -7,4 +7,4 @@",
      " seven",
      " eight",
      "-nine",
      "+NINE",
      " ten",
      "",
    ].join("\n"),
  });

  assert.equal(rows.length, 10);
  assert.deepEqual(rows.map((row) => row.base?.text), base.trimEnd().split("\n"));
  assert.deepEqual(rows.map((row) => row.resolved?.text), resolved.trimEnd().split("\n"));
  assert.equal(rows[4].kind, "context");
  assert.equal(rows[5].kind, "context");
});

test("insertions at the start and missing final newlines retain line numbers", () => {
  const rows = buildSideBySideRows({
    baseText: "existing",
    resolvedText: "new\nexisting",
    unifiedDiff: [
      "--- goal-base.yaml",
      "+++ goal-resolved.yaml",
      "@@ -0,0 +1 @@",
      "+new",
      "\\ No newline at end of file",
      "",
    ].join("\n"),
  });

  assert.equal(rows[0].base, null);
  assert.equal(rows[0].resolved.number, 1);
  assert.equal(rows[1].base.number, 1);
  assert.equal(rows[1].resolved.number, 2);
});

test("pure deletions keep resolved lines aligned with a spacer", () => {
  const rows = buildSideBySideRows({
    baseText: "before\nremove\nafter\n",
    resolvedText: "before\nafter\n",
    unifiedDiff: [
      "--- goal-base.yaml",
      "+++ goal-resolved.yaml",
      "@@ -1,3 +1,2 @@",
      " before",
      "-remove",
      " after",
      "",
    ].join("\n"),
  });

  assert.equal(rows[1].base.text, "remove");
  assert.equal(rows[1].resolved, null);
  assert.equal(rows[2].base.number, 3);
  assert.equal(rows[2].resolved.number, 2);
});

test("malformed or incomplete diffs fail instead of rendering misleading rows", () => {
  assert.throws(
    () => buildSideBySideRows({
      baseText: "before\n",
      resolvedText: "after\n",
      unifiedDiff: "",
    }),
    /comparison has no diff hunks/,
  );
  assert.throws(
    () => buildSideBySideRows({
      baseText: "before\n",
      resolvedText: "after\n",
      unifiedDiff: "@@ -1 +1 @@\n-wrong\n+after\n",
    }),
    /base line 1 does not match its hunk/,
  );
  assert.throws(
    () => buildSideBySideRows({
      baseText: "same\n",
      resolvedText: "same\n",
      unifiedDiff: "not a unified diff\n",
    }),
    /unsupported diff line/,
  );
  assert.throws(
    () => buildSideBySideRows({
      baseText: "same\n",
      resolvedText: "same\n",
      unifiedDiff: "--- goal-base.yaml\n+++ goal-resolved.yaml\n@@ -5,0 +5,0 @@\n",
    }),
    /unchanged range is outside its document/,
  );
});

test("search counts matches independently in both complete panes", () => {
  const rows = [
    {
      kind: "context",
      base: { text: "Frame frame" },
      resolved: { text: "frame" },
    },
    {
      kind: "change",
      base: null,
      resolved: { text: "FRAME" },
    },
  ];

  assert.deepEqual(sideBySideSearchCounts(rows, "frame"), { base: 2, resolved: 2 });
  assert.deepEqual(sideBySideSearchCounts(rows, ""), { base: 0, resolved: 0 });
});
