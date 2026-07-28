import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const panel = readFileSync(
  new URL("../../src/gradlab/web_player/panels/events.js", import.meta.url),
  "utf8",
);
const app = readFileSync(
  new URL("../../src/gradlab/web_player/app.js", import.meta.url),
  "utf8",
);
const styles = readFileSync(
  new URL("../../src/gradlab/web_player/styles.css", import.meta.url),
  "utf8",
);

test("event rows inspect their retained transition", () => {
  assert.match(panel, /jump\.type = "button"/);
  assert.match(
    panel,
    /jump\.addEventListener\("click", \(\) => services\.inspectSequence\(point\.sequence\)\)/,
  );
  assert.match(panel, /jump\.setAttribute\("aria-current", "step"\)/);
  assert.match(app, /services: \{[\s\S]*?\n    inspectSequence,/);
});

test("inspection pauses active playback before selecting the event sequence", () => {
  const cursorStart = app.indexOf("function setInspectionCursor(");
  const cursorEnd = app.indexOf("\nfunction inspectSequence(", cursorStart);
  const cursor = app.slice(cursorStart, cursorEnd);

  assert.ok(cursor.indexOf("maybePauseForInspection()") >= 0);
  assert.ok(
    cursor.indexOf("maybePauseForInspection()")
      < cursor.indexOf("state.inspectionSequence = numericSequence"),
  );
});

test("the events panel owns the only scrollbar", () => {
  const listRule = styles.match(/\.event-list \{([^}]*)\}/)?.[1] || "";
  assert.doesNotMatch(listRule, /overflow|max-height/);
});
