// Browser-local diagnostics. Count changed canvases at most once per refresh,
// independently of incoming snapshots and the requested playback rate.
export function mountRenderSpeed(element) {
  let started = performance.now();
  let frames = 0;
  let draws = 0;
  let decodeTotal = 0;
  let drawTotal = 0;
  let pendingFrame = null;
  let available = false;

  const clearWindow = () => {
    started = performance.now();
    frames = draws = decodeTotal = drawTotal = 0;
  };
  const reset = () => {
    if (pendingFrame !== null) cancelAnimationFrame(pendingFrame);
    pendingFrame = null;
    available = false;
    clearWindow();
    element.textContent = "Render — FPS";
  };
  reset();
  element.title = "Changed game frames per browser refresh, measured over each elapsed second. Decode is average image decoding time; draw is average canvas update time. These timings exclude server work, transport, other panels, and GPU presentation.";

  const timer = setInterval(() => {
    if (document.hidden) {
      reset();
      return;
    }
    if (!available) return;
    const elapsed = performance.now() - started;
    if (elapsed < 1000) return;
    element.textContent = `Render ${(frames * 1000 / elapsed).toFixed(1)} FPS`
      + (draws ? ` · Decode ${(decodeTotal / draws).toFixed(1)} ms · Draw ${(drawTotal / draws).toFixed(1)} ms` : "");
    clearWindow();
  }, 1000);
  document.addEventListener("visibilitychange", reset);

  return {
    record(decodeMs, drawMs) {
      if (document.hidden) return;
      if (!available) clearWindow();
      available = true;
      draws += 1;
      decodeTotal += decodeMs;
      drawTotal += drawMs;
      if (pendingFrame !== null) return;
      pendingFrame = requestAnimationFrame(() => {
        pendingFrame = null;
        frames += 1;
      });
    },
    reset,
    destroy() {
      reset();
      clearInterval(timer);
      document.removeEventListener("visibilitychange", reset);
    },
  };
}
