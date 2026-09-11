// Browser-local diagnostics. Count changed canvases at most once per refresh,
// independently of incoming snapshots and the requested playback rate.
export function mountRenderSpeed(element) {
  element.innerHTML = '<div data-fps-value>— FPS</div><svg viewBox="0 0 120 32" preserveAspectRatio="none" aria-hidden="true"><path data-fps-area></path></svg>';
  const value = element.querySelector("[data-fps-value]");
  const area = element.querySelector("[data-fps-area]");
  const history = [];
  const renderDescription = "Changed game frames per browser refresh, sampled about once per second. Chart and range show the last 60 samples. Decode and draw timings exclude server work, transport, other panels, and GPU presentation.";
  const playbackDescription = "Playback steps per second while RGB is hidden, measured from snapshot progress. Chart and range show the last 60 samples. No RGB frames are received or drawn.";
  let mode = "render";
  let previousProgress = null;
  const description = () => mode === "playback" ? playbackDescription : renderDescription;
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
    previousProgress = null;
    clearWindow();
    history.length = 0;
    value.textContent = "— FPS";
    area.setAttribute("d", "");
    element.title = description();
  };
  reset();


  const timer = setInterval(() => {
    if (document.hidden) {
      reset();
      return;
    }
    if (!available) return;
    const elapsed = performance.now() - started;
    if (elapsed < 1000) return;
    const fps = frames * 1000 / elapsed;
    history.push(fps);
    if (history.length > 60) history.shift();
    const low = Math.min(...history);
    const high = Math.max(...history);
    value.textContent = `${fps.toFixed(1)} FPS (${low.toFixed(0)}–${high.toFixed(0)})`;
    const ceiling = Math.max(60, high * 1.1);
    const left = 120 - history.length * 2;
    const points = history.map((sample, index) => {
      const x = left + index * 2;
      const y = (32 - sample / ceiling * 32).toFixed(2);
      return `L${x},${y} L${x + 2},${y}`;
    }).join(" ");
    area.setAttribute("d", `M${left},32 ${points} L120,32 Z`);
    element.title = description()
      + (draws ? ` Decode ${(decodeTotal / draws).toFixed(1)} ms · Draw ${(drawTotal / draws).toFixed(1)} ms.` : "");
    clearWindow();
  }, 1000);
  document.addEventListener("visibilitychange", reset);

  return {
    setPlaybackMode(enabled) {
      const nextMode = enabled ? "playback" : "render";
      if (mode === nextMode) return;
      mode = nextMode;
      reset();
    },
    recordProgress(snapshot, advancing) {
      if (document.hidden || mode !== "playback") return;
      const step = snapshot?.transition?.step ?? snapshot?.session?.step;
      if (!Number.isFinite(step)) return;
      const identity = JSON.stringify([
        snapshot.session_epoch,
        snapshot.trajectory?.episode_id,
        snapshot.transition?.episode ?? snapshot.session?.episode,
      ]);
      if (!available) clearWindow();
      available = true;
      if (previousProgress?.identity === identity && step >= previousProgress.step) {
        if (advancing || previousProgress.advancing) frames += step - previousProgress.step;
      } else if (previousProgress) {
        reset();
        available = true;
      }
      previousProgress = { identity, step, advancing };
    },
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
