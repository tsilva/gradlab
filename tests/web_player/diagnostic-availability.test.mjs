import assert from "node:assert/strict";
import test from "node:test";

import {
  unavailableDiagnosticRows,
} from "../../src/gradlab/web_player/panels/diagnostic-availability.js";

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
