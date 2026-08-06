import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const app = readFileSync(
  new URL("../../src/gradlab/web_player/app.js", import.meta.url),
  "utf8",
);
const runtime = readFileSync(
  new URL("../../src/gradlab/web_player/panels/runtime.js", import.meta.url),
  "utf8",
);
const styles = readFileSync(
  new URL("../../src/gradlab/web_player/styles.css", import.meta.url),
  "utf8",
);

test("stats panels expose one standardized persisted processing switch", () => {
  assert.match(app, /input\.dataset\.panelEnabled = name/);
  assert.match(app, /instance\.enabled = input\.checked/);
  assert.match(app, /processing: processing\(\)/);
  assert.match(styles, /Disabled — data processing is off/);
});

test("diagnostic processing switches also control their captures", () => {
  assert.match(app, /name === "attribution"/);
  assert.match(app, /function syncAttributionToPanel/);
  assert.match(app, /command\("set_attribution", payload\)/);
  assert.match(app, /"CNN features"/);
  assert.match(app, /function syncCnnCaptureToPanel/);
  assert.match(app, /command\("set_cnn_inspection", \{ enabled: desired \}\)/);
});

test("disabled panels receive no snapshot, history, frame, or resize processing", () => {
  assert.match(runtime, /if \(instance\.definition\.enabled\) this\.safeCall\(id, "render"/);
  assert.match(runtime, /if \(!instance\.definition\.enabled\) return;/);
  assert.match(runtime, /instance\.definition\.enabled && instance\.definition\.frameKinds/);
  assert.match(runtime, /if \(instance\.definition\.enabled\) this\.safeCall\(id, "resize"/);
});
