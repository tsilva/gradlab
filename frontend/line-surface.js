import {
  drawLines,
  themeColor,
} from "../src/gradlab/web_player/panels/shared.js";

// Cache the expensive series raster independently of transient cursor/hover.
// The caller supplies stable series and step identities from Svelte derived state.
export function createLineSurface(canvas) {
  const buffer = document.createElement("canvas");
  let cached = null;
  /** @param {any} series @param {number[]} steps @param {{cursorStep?: number, referenceStep?: number|null, dimBeforeStep?: number|null}} options */
  return (
    series,
    steps,
    { cursorStep, referenceStep = null, dimBeforeStep = null } = {},
  ) => {
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(240, canvas.clientWidth),
      height = Math.max(120, canvas.clientHeight);
    if (
      !cached ||
      cached.series !== series ||
      cached.steps !== steps ||
      cached.width !== width ||
      cached.height !== height ||
      cached.ratio !== ratio ||
      cached.referenceStep !== referenceStep ||
      cached.dimBeforeStep !== dimBeforeStep
    ) {
      // drawLines sizes by CSS dimensions; detached buffers have no layout box.
      Object.defineProperties(buffer, {
        clientWidth: { configurable: true, value: width },
        clientHeight: { configurable: true, value: height },
      });
      const geometry = drawLines(buffer, series, {
        steps,
        referenceStep,
        dimBeforeStep,
      });
      cached = {
        series,
        steps,
        width,
        height,
        ratio,
        referenceStep,
        dimBeforeStep,
        geometry,
      };
    }
    const pixelWidth = Math.round(width * ratio),
      pixelHeight = Math.round(height * ratio);
    if (canvas.width !== pixelWidth) canvas.width = pixelWidth;
    if (canvas.height !== pixelHeight) canvas.height = pixelHeight;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(buffer, 0, 0);
    const plot = cached.geometry?.plot;
    if (
      plot &&
      Number.isFinite(cursorStep) &&
      steps.length &&
      cursorStep >= steps[0] &&
      cursorStep <= steps.at(-1)
    ) {
      const x = Math.max(
        plot.left + 1,
        Math.min(
          plot.right - 1,
          plot.left +
            ((cursorStep - steps[0]) / Math.max(1, steps.at(-1) - steps[0])) *
              (plot.right - plot.left),
        ),
      );
      ctx.save();
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      ctx.strokeStyle = themeColor("chartHighlight");
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(x, plot.top);
      ctx.lineTo(x, plot.bottom);
      ctx.stroke();
      ctx.restore();
    }
    return cached.geometry;
  };
}
