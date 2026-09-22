import assert from "node:assert/strict";
import test from "node:test";
import { createObservationSurface } from "../../frontend/observation-surface.js";

test("exact overlay restores after a missing base without repeatedly decoding unchanged frames", async (t) => {
  let decodes = 0,
    overlayDraws = 0;
  const previous = globalThis.createImageBitmap;
  globalThis.createImageBitmap = async () => {
    decodes++;
    return { width: 4, height: 4, close() {} };
  };
  t.after(() => {
    if (previous) globalThis.createImageBitmap = previous;
    else delete globalThis.createImageBitmap;
  });
  const canvas = (overlay) => ({
    width: 4,
    height: 4,
    hidden: false,
    style: {},
    getContext: () => ({
      drawImage() {
        if (overlay) overlayDraws++;
      },
      clearRect() {},
    }),
  });
  const base = canvas(false),
    overlay = canvas(true),
    stage = { hidden: false };
  const nodes = {
    "[data-observation-canvas]": base,
    "[data-diagnostic-canvas]": overlay,
    ".observation-stage": stage,
  };
  const surface = createObservationSurface(
    { querySelector: (name) => nodes[name] },
    () => {},
  );
  const snapshot = (sequence) => ({
    sequence,
    policy: { attribution: { supported_modes: ["gradcam"] } },
    session: { attribution: { mode: "gradcam" } },
    transition: {
      sequence,
      attribution: { status: "available", generation: 1 },
    },
  });
  const exact = async () => {
    surface.render(snapshot(1));
    await surface.renderFrame(2, {}, { sequence: 1 });
    await surface.renderFrame(3, {}, { sequence: 1, generation: 1 });
  };
  await exact();
  assert.equal(overlay.hidden, false);
  assert.equal(overlayDraws, 1);
  await exact();
  assert.equal(decodes, 2);
  assert.equal(overlayDraws, 1);
  surface.render(snapshot(2));
  await surface.renderFrame(2, null, { sequence: 2 });
  assert.equal(overlay.hidden, true);
  await exact();
  assert.equal(overlay.hidden, false);
  assert.equal(overlayDraws, 2);
  surface.destroy();
});
