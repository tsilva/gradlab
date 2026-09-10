// A shared episode-step viewport; pointer gestures never move the playback cursor.
export function selectedRange(plot, startX, endX, first, last) {
  if (!plot || !Number.isFinite(first) || !Number.isFinite(last) || last <= first || Math.abs(endX - startX) < 5) return null;
  const step = (x) => Math.round(first + Math.max(0, Math.min(1, (x - plot.left) / (plot.right - plot.left))) * (last - first));
  const a = step(startX), b = step(endX);
  return a === b ? null : { first: Math.min(a, b), last: Math.max(a, b) };
}

export function bindChartRange(canvas, geometry, context, services) {
  let drag = null;
  let suppressClick = false;
  const selection = document.createElement("div");
  selection.className = "chart-drag-selection";
  selection.hidden = true;
  canvas.parentElement.style.position = "relative";
  canvas.parentElement.append(selection);
  const x = (event) => (event.clientX - canvas.getBoundingClientRect().left) * canvas.clientWidth / canvas.getBoundingClientRect().width;
  canvas.style.touchAction = "none";
  canvas.addEventListener("pointerdown", (event) => {
    const plot = geometry()?.plot;
    if (event.button !== 0 || !plot || x(event) < plot.left || x(event) > plot.right) return;
    const history = context().view?.chartHistory || context().history;
    drag = { x: x(event), id: event.pointerId, plot, first: history[0]?.step, last: history.at(-1)?.step };
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!drag) return;
    const plot = geometry()?.plot;
    if (!plot) return;
    const left = Math.max(plot.left, Math.min(drag.x, x(event)));
    const right = Math.min(plot.right, Math.max(drag.x, x(event)));
    selection.hidden = false;
    selection.style.cssText = `left:${canvas.offsetLeft + left}px;top:${canvas.offsetTop + plot.top}px;width:${right - left}px;height:${plot.bottom - plot.top}px`;

  });
  canvas.addEventListener("pointerup", (event) => {
    if (!drag) return;
    const range = selectedRange(drag.plot, drag.x, x(event), drag.first, drag.last);
    selection.hidden = true;
    suppressClick = Math.abs(drag.x - x(event)) >= 5;
    drag = null;
    canvas.releasePointerCapture(event.pointerId);
    if (range) services.setChartRange?.(range);
  });
  canvas.addEventListener("pointercancel", () => { drag = null; selection.hidden = true; });
  canvas.addEventListener("click", (event) => {
    if (suppressClick) { event.stopImmediatePropagation(); suppressClick = false; }
  }, true);
  canvas.addEventListener("dblclick", (event) => {
    event.preventDefault();
    services.setChartRange?.(null);
  });
}
