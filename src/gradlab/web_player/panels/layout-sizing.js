export const DEFAULT_GRID_CELL_HEIGHT = 32;
export const VIEWPORT_FILL_MAX_ROWS = 15;
const VIEWPORT_BOTTOM_GAP = 16;
const GRID_MIN_ROWS = 8;

export function viewportGridCellHeight({
  viewportHeight,
  dashboardTop,
  timelineHeight,
  rows,
}) {
  const visibleRows = Math.max(0, Number(rows) || 0);
  if (!visibleRows || visibleRows > VIEWPORT_FILL_MAX_ROWS) {
    return DEFAULT_GRID_CELL_HEIGHT;
  }
  const availableHeight = (
    Math.max(0, Number(viewportHeight) || 0)
    - Math.max(0, Number(dashboardTop) || 0)
    - Math.max(0, Number(timelineHeight) || 0)
    - VIEWPORT_BOTTOM_GAP
  );
  const fittedHeight = Math.floor(
    availableHeight / Math.max(GRID_MIN_ROWS, visibleRows),
  );
  return Math.max(DEFAULT_GRID_CELL_HEIGHT, fittedHeight);
}
