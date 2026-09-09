import assert from "node:assert/strict";
import test from "node:test";

import {
  unavailableDiagnosticRows,
} from "../../src/gradlab/web_player/panels/diagnostic-availability.js";
import { attributionPresentation, cnnPresentation } from "../../src/gradlab/web_player/panels/diagnostic-overlays.js";

test("imported diagnostics distinguish omitted capture from unsupported Policy features", () => {
  const snapshot = {
    mode: "trajectory",
    policy: { attribution: { supported_modes: ["gradcam"] }, cnn: { layers: ["conv"] } },
    session: { attribution: { status: "not-recorded" }, cnn: { status: "not-recorded" } },
  };
  assert.equal(attributionPresentation(snapshot).label, "Not recorded");
  assert.equal(cnnPresentation(snapshot).label, "Not recorded");
  assert.deepEqual(unavailableDiagnosticRows([
    { panel: "Attribution", statuses: ["not-recorded"] },
    { panel: "Action values", statuses: ["unsupported"] },
  ]), [
    { panel: "Attribution", label: "Not recorded", tone: "not-recorded" },
    { panel: "Action values", label: "Unsupported", tone: "unsupported" },
  ]);
});

test("initial observation does not become an unavailable-diagnostics banner", () => {
  const initialObservation = [
    { panel: "Action decision", statuses: ["not-yet-observed"] },
    { panel: "Reward analysis", statuses: ["not-yet-observed"] },
  ];
  const firstDecision = [
    { panel: "Action decision", statuses: ["available"] },
    { panel: "Reward analysis", statuses: ["available"] },
  ];

  assert.deepEqual(unavailableDiagnosticRows(initialObservation), []);
  assert.deepEqual(unavailableDiagnosticRows(firstDecision), []);
  assert.deepEqual(unavailableDiagnosticRows(initialObservation), []);
});

test("unsupported, incomparable, and erroneous diagnostics remain visible", () => {
  assert.deepEqual(
    unavailableDiagnosticRows([
      { panel: "Action decision", statuses: ["unsupported"] },
      { panel: "Value estimate", statuses: ["contract-incomparable"] },
      { panel: "Reward analysis", statuses: ["protocol-error"] },
    ]),
    [
      { panel: "Action decision", label: "Unsupported", tone: "unsupported" },
      { panel: "Value estimate", label: "Incomparable", tone: "incomparable" },
      { panel: "Reward analysis", label: "Protocol error", tone: "error" },
    ],
  );
});
