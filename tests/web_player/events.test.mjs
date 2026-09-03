import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  EVENT_COLOR_PALETTE,
  eventColor,
  eventColorFill,
  eventLabels,
} from "../../src/gradlab/web_player/event-colors.js";

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

test("event labels select stable colors from a finite palette", () => {
  for (const label of ["life_loss", "coin", "level_change", "猫"]) {
    assert.equal(eventColor(label), eventColor(label));
    assert.ok(EVENT_COLOR_PALETTE.includes(eventColor(label)));
  }
  assert.equal(eventColor("life_loss"), "var(--color-series-burnt-orange)");
  assert.equal(eventColor("coin"), "var(--color-series-lavender)");
  assert.deepEqual(eventLabels({ events: [" life_loss ", "coin"] }), ["life_loss", "coin"]);
  assert.deepEqual(eventLabels({ boundary: true, events: [] }), ["episode boundary"]);
});

test("event rows and timeline markers share the same label-derived fill", () => {
  const labels = eventLabels({ events: ["life_loss", "coin"] });
  const fill = eventColorFill(labels);
  assert.match(fill, /^linear-gradient\(to bottom,/);
  assert.ok(fill.includes(eventColor("life_loss")));
  assert.ok(fill.includes(eventColor("coin")));
  assert.match(panel, /eventColorFill\(labels\)/);
  assert.match(app, /eventColorFill\(eventLabels\(point\)\)/);
  assert.match(styles, /\.event-item::before \{[^}]*background: var\(--event-colors/);
  assert.match(styles, /\.timeline-marker \{[^}]*background: var\(--event-colors/);
});
