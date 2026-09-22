import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const app = readFileSync(
  new URL("../../src/gradlab/web_player/app.js", import.meta.url),
  "utf8",
);
const styles = readFileSync(
  new URL("../../src/gradlab/web_player/styles.css", import.meta.url),
  "utf8",
);

test("stats panels expose one standardized persisted processing switch", () => {
  assert.match(app, /processing: processing\(\)/);
  assert.doesNotMatch(app, /label\.textContent = definition\.enabled \? "Enabled" : "Disabled"/);
  assert.match(
    styles,
    /\.panel-processing-toggle \{[^}]*grid-column: -3 \/ -2;[^}]*justify-self: end;/,
  );
  assert.match(
    styles,
    /\.panel-processing-toggle input \{[^}]*appearance: none;[^}]*border-radius: 999px;/,
  );
  assert.match(styles, /\.panel-processing-toggle input:checked \{/);
  assert.match(styles, /Disabled — data processing is off/);
});

test("diagnostic processing switches also control their captures", () => {
  assert.match(app, /function syncAttributionToPanel/);
  assert.match(app, /command\("set_attribution", payload\)/);
  assert.match(app, /function syncCnnCaptureToPanel/);
  assert.match(app, /command\("set_cnn_inspection", \{ enabled: desired \}\)/);
});


test("disabled delivery gates every optional path and preserves disposal/reset", async () => {
  const { PanelDelivery } = await import('../../frontend/panel-delivery.js');
  const calls = [];
  const delivery = new PanelDelivery();
  const definition = {enabled:false,frameKinds:[2]};
  const methods = Object.fromEntries(['render','renderHistory','prepareFrame','renderFrame','resize','resetFrames'].map(name=>[name,()=>calls.push(name)]));
  delivery.instances.set('input',{definition,...methods});
  delivery.renderSnapshot({sequence:1});delivery.renderHistory([]);
  await delivery.prepareFrame(2,new Blob());await delivery.renderFrame(2,new Blob());
  delivery.resize();delivery.invoke('input','resize');
  assert.deepEqual(calls,[]);
  delivery.resetFrames();assert.deepEqual(calls,['resetFrames']);
  definition.enabled=true;delivery.renderSnapshot({sequence:2});
  assert.deepEqual(calls,['resetFrames','render']);
});
