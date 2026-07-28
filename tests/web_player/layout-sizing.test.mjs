import assert from "node:assert/strict";
import test from "node:test";

import {
  DEFAULT_GRID_CELL_HEIGHT,
  viewportGridCellHeight,
} from "../../src/gradlab/web_player/panels/layout-sizing.js";

test("compact workspaces expand to use the available viewport height", () => {
  assert.equal(viewportGridCellHeight({
    viewportHeight: 1024,
    dashboardTop: 60,
    timelineHeight: 84,
    rows: 15,
  }), 57);
});

test("multi-row workspaces retain the normal scrollable grid sizing", () => {
  assert.equal(viewportGridCellHeight({
    viewportHeight: 1024,
    dashboardTop: 60,
    timelineHeight: 84,
    rows: 24,
  }), DEFAULT_GRID_CELL_HEIGHT);
});

test("small viewports never shrink panels below the default row height", () => {
  assert.equal(viewportGridCellHeight({
    viewportHeight: 400,
    dashboardTop: 60,
    timelineHeight: 84,
    rows: 15,
  }), DEFAULT_GRID_CELL_HEIGHT);
});
