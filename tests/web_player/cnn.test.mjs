import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  atlasTileRect,
  cnnFrameIdentity,
  cnnPresentation,
  peakRegionLabel,
} from "../../src/gradlab/web_player/panels/cnn.js";

const source = readFileSync(
  new URL("../../src/gradlab/web_player/panels/cnn.js", import.meta.url),
  "utf8",
);
const appSource = readFileSync(
  new URL("../../src/gradlab/web_player/app.js", import.meta.url),
  "utf8",
);
const catalogSource = readFileSync(
  new URL("../../src/gradlab/web_player/panels/catalog.js", import.meta.url),
  "utf8",
);

function snapshot({
  layers = [{ id: "cnn.0", label: "Conv 1" }],
  session = { enabled: true, status: "active", layer_id: "cnn.0", interval: 1, top_k: 12 },
  cnn = {
    status: "available",
    layer_id: "cnn.0",
    generation: 5,
    inspection: { generation: 5, filters: [] },
  },
} = {}) {
  return {
    sequence: 14,
    policy: { cnn: { layers, unavailable_reason: layers.length ? null : "no CNN" } },
    session: { cnn: session },
    transition: { sequence: 14, cnn },
  };
}

test("CNN frames require exact transition and generation identity", () => {
  assert.deepEqual(cnnFrameIdentity(snapshot()), { sequence: 14, generation: 5 });
  assert.equal(cnnFrameIdentity(snapshot({
    cnn: { status: "available", generation: 0, inspection: {} },
  })), null);
  assert.equal(cnnFrameIdentity(snapshot({
    cnn: { status: "not_computed", generation: 5, inspection: null },
  })), null);
});

test("CNN status distinguishes exact, computing, cadence, off, and unavailable", () => {
  assert.equal(cnnPresentation(snapshot(), true).label, "Exact");
  assert.equal(cnnPresentation(snapshot(), false).label, "Computing");
  assert.equal(cnnPresentation(snapshot({
    cnn: { status: "not_computed", generation: 0, reason: "cadence", inspection: null },
  })).label, "Cadence skipped");
  assert.equal(cnnPresentation(snapshot({
    session: { enabled: false, status: "off" },
  })).label, "Off");
  assert.equal(cnnPresentation(snapshot({ layers: [] })).label, "Unavailable");
});

test("atlas tiles and peak regions map deterministically", () => {
  assert.deepEqual(atlasTileRect(
    { columns: 3, tile_width: 84, tile_height: 84 },
    5,
  ), { x: 168, y: 84, width: 84, height: 84 });
  assert.equal(peakRegionLabel({ x0: 1.2, y0: 3.1, x1: 8.4, y1: 9.8 }), "x 1–9, y 3–10");
});

test("CNN panel is modular, opt-in, and generation-tagged", () => {
  assert.match(source, /services\.command\("set_cnn_inspection"/);
  assert.match(source, /Kernel tiles show every learned input-channel plane/);
  assert.match(source, /not why the policy selected its action/);
  assert.match(catalogSource, /module: "\.\/cnn\.js"/);
  assert.match(catalogSource, /subscriptions: \["observation", "cnn-inspection"\]/);
  assert.match(catalogSource, /frameKinds: \[FRAME_OBSERVATION, FRAME_CNN_INSPECTION\]/);
  assert.match(appSource, /cnnInspectionGeneration/);
  assert.match(appSource, /FRAME_CNN_INSPECTION/);
});
