import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  attributionFrameIdentity,
  attributionPresentation,
} from "../../src/gradlab/web_player/panels/attribution.js";
import { PANEL_TYPES } from "../../src/gradlab/web_player/panels/catalog.js";

const source = readFileSync(
  new URL("../../src/gradlab/web_player/panels/attribution.js", import.meta.url),
  "utf8",
);
const appSource = readFileSync(
  new URL("../../src/gradlab/web_player/app.js", import.meta.url),
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
  assert.deepEqual(attributionFrameIdentity(snapshot()), { sequence: 9, generation: 4 });
  assert.equal(attributionFrameIdentity(snapshot({
    transition: { status: "available", mode: "gradcam", generation: 0 },
  })), null);
  assert.equal(attributionFrameIdentity(snapshot({
    transition: { status: "not_computed", mode: "gradcam", generation: 4, reason: "cadence" },
  })), null);
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

test("attribution is an independent opt-in control panel", () => {
  assert.match(source, /services\.command\("set_attribution"/);
  assert.match(source, /METHOD_DEFAULT_INTERVAL.*gradcam: 1, occlusion: 8/);
  assert.doesNotMatch(source, /FRAME_OBSERVATION/);
  assert.doesNotMatch(source, /data-observation-canvas/);
  assert.equal(PANEL_TYPES.attribution.module, "./attribution.js");
  assert.deepEqual(PANEL_TYPES.attribution.subscriptions, []);
  assert.deepEqual(PANEL_TYPES.attribution.processing, ["attribution"]);
  assert.deepEqual(PANEL_TYPES.attribution.frameKinds, []);
  assert.match(appSource, /function syncAttributionToPanel/);
  assert.match(appSource, /command\("set_attribution", payload\)/);
});
