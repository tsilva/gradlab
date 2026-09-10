import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import test from "node:test";
import { PanelRuntime } from "../../src/gradlab/web_player/panels/runtime.js";

const source = readFileSync(new URL("../../src/gradlab/web_player/app.js", import.meta.url), "utf8");

test("fullscreen suspends only panels outside the game in this document", () => {
  const document = { fullscreenElement: null };
  const context = vm.createContext({ document });
  vm.runInContext(source.slice(source.indexOf("function panelSuspended("), source.indexOf("function processingPanels(")), context);
  assert.equal(context.panelSuspended("policy"), false);
  document.fullscreenElement = {
    matches: () => true,
    closest: () => ({ dataset: { panel: "game" } }),
  };
  assert.equal(context.panelSuspended("policy"), true);
  assert.equal(context.panelSuspended("game"), false);
  document.fullscreenElement = { matches: () => false };
  assert.equal(context.panelSuspended("policy"), false);
  document.fullscreenElement = null;
  assert.equal(context.panelSuspended("policy"), false);
});

test("suspended instances keep their state, skip work and resume with fresh data", async () => {
  let fullscreen = true;
  const calls = [];
  const runtime = new PanelRuntime({ isSuspended: (id) => fullscreen && id !== "game" });
  const panel = {
    definition: { enabled: true, frameKinds: [1] },
    selection: "episode",
    render: (snapshot) => calls.push(snapshot.sequence),
    renderHistory: () => calls.push("history"),
    prepareFrame: () => calls.push("prepare"),
    renderFrame: () => calls.push("frame"),
    resize: () => calls.push("resize"),
    resetFrames: () => calls.push("reset"),
  };
  runtime.instances.set("policy", panel);
  runtime.renderSnapshot({ sequence: 1 });
  runtime.renderHistory([]);
  await runtime.prepareFrame(1, null);
  await runtime.renderFrame(1, null);
  runtime.resize();
  assert.deepEqual(calls, []);
  runtime.resetFrames();
  assert.deepEqual(calls, ["reset"]);
  fullscreen = false;
  runtime.renderSnapshot({ sequence: 2 });
  runtime.renderHistory([]);
  await runtime.renderFrame(1, null);
  runtime.resize();
  assert.deepEqual(calls, ["reset", 2, "history", "frame", "resize"]);
  assert.equal(runtime.instances.get("policy"), panel);
  assert.equal(panel.selection, "episode");
  assert.equal(panel.definition.enabled, true);
});
