import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  observationFrameIdentity,
} from "../../src/gradlab/web_player/panels/observation.js";
import {
  OVERLAY_ATTRIBUTION,
  OVERLAY_CNN,
  OVERLAY_NONE,
  attributionFrameIdentity,
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

test("observation owns no attribution controls or processing", () => {
  assert.doesNotMatch(source, /set_attribution/);
  assert.doesNotMatch(source, /data-attribution-method/);
  assert.doesNotMatch(source, /data-attribution-interval/);
  assert.doesNotMatch(source, /data-diagnostic-overlay/);
  assert.doesNotMatch(source, /data-overlay-opacity/);
  assert.match(source, /reconcileOverlaySelection/);
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
  assert.deepEqual(PANEL_TYPES.observation.processing, ["observation"]);
  assert.deepEqual(PANEL_TYPES.observation.frameKinds, [2, 3, 4]);
  assert.match(source, /baseIdentity/);
  assert.match(source, /sameFrameIdentity\(baseIdentity, expectedBaseIdentity\(\)\)/);
  for (const [identity, request] of [
    ["targetBaseIdentity", "baseBitmapRequest"],
    ["targetAttributionIdentity", "attributionBitmapRequest"],
    ["targetCnnIdentity", "cnnBitmapRequest"],
  ]) {
    const guard = source.indexOf(
      `if (!sameFrameIdentity(incoming, ${identity}())) return true;`,
    );
    const decode = source.indexOf(`const request = ++${request};`, guard);
    assert.ok(guard >= 0 && decode > guard, `${identity} rejects unrelated frames before decoding`);
  }
  assert.match(appSource, /magic !== "RLP3"/);
  assert.match(appSource, /getBigUint64\(24\)/);
  assert.match(appSource, /function frameGeneration/);
  assert.match(catalogSource, /cnn-inspection/);
});

test("observation commits an exact decoded frame and its metadata without blanking between frames", () => {
  const renderStart = source.indexOf("render(nextSnapshot)");
  const frameStart = source.indexOf("async renderFrame", renderStart);
  const renderSource = source.slice(renderStart, frameStart);
  assert.match(renderSource, /targetSnapshot = nextSnapshot/);
  assert.doesNotMatch(renderSource, /closeBase\(\)/);
  assert.match(source, /request < baseBitmapCommittedRequest/);
  assert.match(source, /commitSnapshot\(frameSnapshot\)/);
  assert.match(source, /baseCanvas\.hidden = true/);
  assert.doesNotMatch(source, /baseCanvas\.width = 1/);
});

test("observation omits the frame stage when no exact frame exists", () => {
  assert.match(source, /<div class="observation-stage" hidden>/);
  assert.match(source, /stage\.hidden = !exactBase/);
  assert.doesNotMatch(source, /data-empty/);
  assert.doesNotMatch(source, /No exact pre-action observation frame/);
});
