import test from "node:test";
import assert from "node:assert/strict";
import { mountTrajectoryControls } from "../../src/gradlab/web_player/trajectory-controls.js";

test("episode downloads require confirmation, show preparation, and recover from errors", async () => {
  const elements = new Map();
  const element = (id) => {
    if (!elements.has(id)) elements.set(id, {
      handlers: {}, hidden: false, disabled: false, textContent: "",
      addEventListener(type, handler) { this.handlers[type] = handler; },
      setAttribute() {}, removeAttribute() {}, after() {}, prepend() {},
      showModal() { this.open = true; }, close() { this.open = false; },
      click() { this.clicked = true; }, remove() {},
    });
    return elements.get(id);
  };
  const oldDocument = globalThis.document;
  const anchor = element("anchor");
  globalThis.document = {
    querySelector: element, createElement: () => anchor, body: { append() {} },
  };
  let resolveRequest;
  let rejectRequest;
  let requests = 0;
  const trajectory = { available: true, enabled: true, transitions: 42 };
  try {
    const controls = mountTrajectoryControls({
      command() {}, toast() {},
      getState: () => ({ hasControl: true, snapshot: { trajectory } }),
      request: () => {
        requests++;
        return new Promise((resolve, reject) => { resolveRequest = resolve; rejectRequest = reject; });
      },
    });
    controls.render();
    assert.equal(element("#trajectory-status").hidden, true);
    element("#trajectory-download").handlers.click();
    assert.equal(requests, 0);
    assert.equal(element("#trajectory-download-dialog").open, true);
    element("#trajectory-download-cancel").handlers.click();
    assert.equal(element("#trajectory-download-dialog").open, false);
    element("#trajectory-download").handlers.click();
    const pending = element("#trajectory-download-confirm").handlers.click();
    assert.equal(requests, 1);
    assert.equal(element("#trajectory-download-progress").hidden, false);
    assert.equal(element("#trajectory-download-confirm").disabled, true);
    let prevented = false;
    element("#trajectory-download-dialog").handlers.cancel({ preventDefault() { prevented = true; } });
    assert.equal(prevented, true);
    resolveRequest({ url: "/download/test", filename: "Game-v0-checkpoint-abc-123.trj" });
    await pending;
    assert.equal(anchor.clicked, true);
    assert.equal(anchor.download, "Game-v0-checkpoint-abc-123.trj");
    assert.equal(element("#trajectory-download-dialog").open, false);
    element("#trajectory-download").handlers.click();
    const failed = element("#trajectory-download-confirm").handlers.click();
    rejectRequest(new Error("Preparation failed"));
    await failed;
    assert.equal(element("#trajectory-download-error").textContent, "Preparation failed");
    assert.equal(element("#trajectory-download-dialog").open, true);
    assert.equal(element("#trajectory-download-confirm").disabled, false);
    assert.equal(element("#trajectory-download-progress").hidden, true);
    trajectory.error = "Disk full";
    controls.render();
    assert.equal(element("#trajectory-status").hidden, false);
  } finally {
    globalThis.document = oldDocument;
  }
});
