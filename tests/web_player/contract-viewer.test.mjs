import assert from "node:assert/strict";
import test from "node:test";
import { ContractViewer } from "../../src/gradlab/web_player/documents/viewer.js";
import { renderComponent } from "./helpers/svelte-render.mjs";

async function contractHtml(view, query = "", changes = "@@ -1 +1 @@\n-value: 1\n+value: 2\n") {
  const controller = new ContractViewer({ addEventListener() {} }, {});
  controller.payload = { documents: { goal: {
    title: "Goal YAML", availability: "frozen", views: {
      base: "value: 1\n", resolved: "value: 2\n", changes: { unified_diff: changes },
    },
  } } };
  controller.view = view;
  controller.query = query;
  let presentation;
  controller.presentation = { render(value) { presentation = value; } };
  controller.render();
  return renderComponent(new URL("../../frontend/components/ContractViewer.svelte", import.meta.url), {
    controller, presentation,
  });
}

test("contract changes render both YAML sides, inline changes, and search counts", async () => {
  const html = await contractHtml("changes", "value");
  assert.match(html, /Base 1 · Resolved 1/);
  assert.match(html, /contract-diff-removed/);
  assert.match(html, /contract-diff-added/);
  assert.match(html, /contract-diff-inline/);
  assert.match(html, /<mark>/);
  assert.match(html, /aria-label="Contract view"/);
});

test("contract changes expose invalid diffs instead of displaying an inaccurate comparison", async () => {
  const html = await contractHtml("changes", "", "@@ -1 +1 @@\n-wrong: 1\n+value: 2\n");
  assert.match(html, /diff does not match the supplied Base and Resolved YAML/);
});
