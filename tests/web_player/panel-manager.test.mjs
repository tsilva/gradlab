import assert from "node:assert/strict";
import test from "node:test";

import {
  defaultBlockForKind,
  editorFieldsForBlock,
} from "../../src/gradlab/web_player/panels/manager.js";

test("reward breakdown editor exposes scope without a metric selector", () => {
  const block = defaultBlockForKind("reward-breakdown");

  assert.deepEqual(block, { kind: "reward-breakdown", scope: "step" });
  assert.deepEqual(editorFieldsForBlock(block), {
    metric: false,
    namespace: false,
    scope: true,
  });
});

test("ordinary telemetry blocks retain their metric editor", () => {
  const block = defaultBlockForKind("line");

  assert.deepEqual(block, { kind: "line", metrics: ["reward/shaped"] });
  assert.equal(editorFieldsForBlock(block).metric, true);
  assert.equal(editorFieldsForBlock(block).scope, false);
});

