import {
  atlasTileRect,
  cnnFrameIdentity,
  cnnPresentation,
  sameFrameIdentity,
} from "./diagnostic-overlays.js";

const FRAME_CNN_INSPECTION = 4;

export { atlasTileRect, cnnFrameIdentity, cnnPresentation };

export function peakRegionLabel(region) {
  if (![region?.x0, region?.y0, region?.x1, region?.y1].every(Number.isFinite)) return "Region unavailable";
  return `x ${Math.floor(region.x0)}–${Math.ceil(region.x1)}, y ${Math.floor(region.y0)}–${Math.ceil(region.y1)}`;
}

function response(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "—";
  if (numeric === 0) return "0";
  if (Math.abs(numeric) >= 1000 || Math.abs(numeric) < 0.001) return numeric.toExponential(2);
  return numeric.toFixed(3);
}

