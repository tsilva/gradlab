import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  attributionFrameIdentity,
  attributionPresentation,
  observationFrameIdentity,
} from "../../src/gradlab/web_player/panels/observation.js";
import {
  OVERLAY_ATTRIBUTION,
  OVERLAY_CNN,
  OVERLAY_NONE,
  cnnWinnerLegend,
  diagnosticActivity,
  drawCnnWinnerOverlay,
  reconcileOverlaySelection,
  sameFrameIdentity,
} from "../../src/gradlab/web_player/panels/diagnostic-overlays.js";
import { PANEL_TYPES } from "../../src/gradlab/web_player/panels/catalog.js";

const source = readFileSync(
  new URL("../../src/gradlab/web_player/panels/observation.js", import.meta.url),
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
  supported = ["gradcam", "occlusion"],
  session = { mode: "gradcam", status: "active", interval: 1 },
  transition = { status: "available", mode: "gradcam", generation: 4, reason: null },
} = {}) {
  return {
    sequence: 9,
    policy: {
      attribution: {
        supported_modes: supported,
        unavailable_reason: supported.length ? null : "no convolutional actor encoder",
      },
    },
    session: { attribution: session },
    transition: transition === null ? null : { sequence: 9, attribution: transition },
  };
}

test("attribution frames require exact transition and generation identity", () => {
  assert.deepEqual(observationFrameIdentity(snapshot()), { sequence: 9 });
  assert.deepEqual(attributionFrameIdentity(snapshot()), { sequence: 9, generation: 4 });
  assert.equal(attributionFrameIdentity(snapshot({
    transition: { status: "available", mode: "gradcam", generation: 0 },
  })), null);
  assert.equal(attributionFrameIdentity(snapshot({
    transition: { status: "not_computed", mode: "gradcam", generation: 4, reason: "cadence" },
  })), null);
  assert.equal(sameFrameIdentity({ sequence: 9 }, { sequence: 9 }), true);
  assert.equal(sameFrameIdentity({ sequence: 9 }, { sequence: 10 }), false);
  assert.equal(
    sameFrameIdentity({ sequence: 9, generation: 4 }, { sequence: 9, generation: 5 }),
    false,
  );
  assert.equal(
    sameFrameIdentity({ sequence: 9 }, { sequence: 9, generation: 4 }),
    false,
  );
});

test("attribution UI distinguishes computing, cadence, no decision, error, and unavailable", () => {
  assert.equal(attributionPresentation(snapshot(), false).label, "Computing");
  assert.equal(attributionPresentation(snapshot(), true).label, "Available");
  assert.equal(attributionPresentation(snapshot({
    transition: { status: "not_computed", mode: "occlusion", generation: 0, reason: "cadence" },
  })).label, "Cadence skipped");
  assert.equal(attributionPresentation(snapshot({
    transition: { status: "not_computed", mode: "gradcam", generation: 0, reason: "no_policy_decision" },
  })).label, "No policy decision");
  assert.equal(attributionPresentation(snapshot({
    session: { mode: "gradcam", status: "error", interval: 1, error: "boom" },
  })).label, "Error");
  assert.equal(attributionPresentation(snapshot({ supported: [] })).label, "Unavailable");
});

test("method and cadence are shared commands while selection and opacity stay local", () => {
  assert.match(source, /services\.command\("set_attribution"/);
  const localControls = source.slice(
    source.indexOf('overlay.addEventListener("change"'),
    source.indexOf("return {", source.indexOf('overlay.addEventListener("change"')),
  );
  assert.doesNotMatch(localControls, /services\.command/);
  assert.match(source, /METHOD_DEFAULT_INTERVAL.*gradcam: 1, occlusion: 8/);
  assert.doesNotMatch(source, /data-attribution-visible/);
});

test("overlay selection is exclusive, sticky, and follows newly active tools", () => {
  const none = { attribution: false, cnn: false };
  const both = { attribution: true, cnn: true };
  assert.equal(reconcileOverlaySelection({ activity: both }), OVERLAY_ATTRIBUTION);
  assert.equal(reconcileOverlaySelection({
    selection: OVERLAY_NONE,
    previousActivity: both,
    activity: both,
    initialized: true,
  }), OVERLAY_NONE);
  assert.equal(reconcileOverlaySelection({
    selection: OVERLAY_ATTRIBUTION,
    previousActivity: { attribution: true, cnn: false },
    activity: both,
    initialized: true,
  }), OVERLAY_CNN);
  assert.equal(reconcileOverlaySelection({
    selection: OVERLAY_CNN,
    previousActivity: both,
    activity: { attribution: true, cnn: false },
    initialized: true,
  }), OVERLAY_ATTRIBUTION);
  assert.equal(reconcileOverlaySelection({
    selection: OVERLAY_ATTRIBUTION,
    previousActivity: { attribution: true, cnn: false },
    activity: none,
    initialized: true,
  }), OVERLAY_NONE);
});

test("diagnostic activity follows supported shared computation state", () => {
  assert.deepEqual(diagnosticActivity({
    policy: {
      attribution: { supported_modes: ["gradcam"] },
      cnn: { layers: [{ id: "cnn.0" }] },
    },
    session: {
      attribution: { mode: "gradcam", status: "active" },
      cnn: { enabled: true, status: "active" },
    },
  }), { attribution: true, cnn: true });
  assert.deepEqual(diagnosticActivity({
    policy: {
      attribution: { supported_modes: [] },
      cnn: { layers: [{ id: "cnn.0" }] },
    },
    session: {
      attribution: { mode: "gradcam", status: "active" },
      cnn: { enabled: false, status: "off" },
    },
  }), { attribution: false, cnn: false });
});

test("CNN winner renderer repeats the exact atlas tile across observation frames", () => {
  const calls = [];
  const context = {
    imageSmoothingEnabled: false,
    drawImage(...args) { calls.push(args); },
  };
  const rendered = drawCnnWinnerOverlay(context, { id: "atlas" }, {
    transition: {
      before: { observation_frames: 4 },
      cnn: {
        inspection: {
          atlas: { columns: 3, tile_width: 84, tile_height: 84, winner_tile: 5 },
          filters: [
            { filter_index: 7, color: "#ffcc00" },
            { filter_index: 2, color: "#00ccff" },
          ],
        },
      },
    },
  }, 336, 84);
  assert.equal(rendered, true);
  assert.equal(context.imageSmoothingEnabled, true);
  assert.equal(calls.length, 4);
  assert.deepEqual(calls[0].slice(1), [168, 84, 84, 84, 0, 0, 84, 84]);
  assert.deepEqual(calls[3].slice(1), [168, 84, 84, 84, 252, 0, 84, 84]);
  assert.deepEqual(cnnWinnerLegend({
    transition: { cnn: { inspection: { filters: [
      { filter_index: 7, color: "#ffcc00" },
      { filter_index: "bad", color: "#fff" },
    ] } } },
  }), [{ index: 7, color: "#ffcc00" }]);
});

test("observation receives CNN frames without demanding CNN processing", () => {
  assert.deepEqual(
    PANEL_TYPES.observation.subscriptions,
    ["observation", "attribution", "cnn-inspection"],
  );
  assert.deepEqual(PANEL_TYPES.observation.processing, ["observation", "attribution"]);
  assert.deepEqual(PANEL_TYPES.observation.frameKinds, [2, 3, 4]);
  assert.match(source, /baseIdentity/);
  assert.match(source, /sameFrameIdentity\(baseIdentity, expectedBaseIdentity\(\)\)/);
  assert.ok(
    source.indexOf("if (!sameFrameIdentity(incoming, expectedBaseIdentity())) return true;")
      < source.indexOf("const request = ++baseBitmapRequest;"),
    "stale base frames are rejected before they can cancel an exact-frame decode",
  );
  assert.ok(
    source.indexOf("if (!sameFrameIdentity(incoming, expectedAttributionIdentity())) return true;")
      < source.indexOf("const request = ++attributionBitmapRequest;"),
    "stale attribution frames are rejected before they can cancel an exact-frame decode",
  );
  assert.ok(
    source.indexOf("if (!sameFrameIdentity(incoming, expectedCnnIdentity())) return true;")
      < source.indexOf("const request = ++cnnBitmapRequest;"),
    "stale CNN frames are rejected before they can cancel an exact-frame decode",
  );
  assert.match(appSource, /magic !== "RLP3"/);
  assert.match(appSource, /getBigUint64\(24\)/);
  assert.match(appSource, /function frameGeneration/);
  assert.match(catalogSource, /cnn-inspection/);
});
