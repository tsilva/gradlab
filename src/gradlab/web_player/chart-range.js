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

export function resizeChartRange(range, episode, edge, step) {
  if (!range || !episode || episode.last <= episode.first) return range;
  return edge === "first"
    ? { first: Math.max(episode.first, Math.min(range.last - 1, Math.round(step))), last: range.last }
    : { first: range.first, last: Math.min(episode.last, Math.max(range.first + 1, Math.round(step))) };
}

export function bindTimelineRange(band, getEpisode, getRange, setRange) {
  for (const handle of band.querySelectorAll(".timeline-range-handle")) {
    const edge = handle.dataset.edge;
    let drag = null;
    const move = (event) => {
      if (!drag || event.pointerId !== drag.id) return;
      const step = drag.range[edge] + (event.clientX - drag.x) / drag.width * (drag.episode.last - drag.episode.first);
      const next = resizeChartRange(drag.range, drag.episode, edge, step);
      const current = getRange();
      if (current && (next.first !== current.first || next.last !== current.last)) setRange(next);
    };
    handle.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 || !getRange() || !getEpisode()) return;
      event.preventDefault();
      event.stopPropagation();
      handle.focus();
      drag = { id: event.pointerId, x: event.clientX, width: band.parentElement.getBoundingClientRect().width,
        range: { ...getRange() }, episode: { ...getEpisode() } };
      handle.setPointerCapture(event.pointerId);
    });
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", (event) => {
      if (!drag || event.pointerId !== drag.id) return;
      move(event);
      drag = null;
      handle.releasePointerCapture(event.pointerId);
    });
    for (const name of ["pointercancel", "lostpointercapture"]) handle.addEventListener(name, () => { drag = null; });
    handle.addEventListener("click", (event) => event.stopPropagation());
    handle.addEventListener("keydown", (event) => {
      const range = getRange(), episode = getEpisode();
      if (!range || !episode) return;
      const delta = event.shiftKey ? 10 : 1;
      const step = { ArrowLeft: range[edge] - delta, ArrowDown: range[edge] - delta,
        ArrowRight: range[edge] + delta, ArrowUp: range[edge] + delta,
        Home: episode.first, End: episode.last }[event.key];
      if (step === undefined) return;
      event.preventDefault();
      event.stopPropagation();
      setRange(resizeChartRange(range, episode, edge, step));
    });
  }
}
