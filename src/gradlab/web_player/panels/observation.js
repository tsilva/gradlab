import {
  OVERLAY_ATTRIBUTION,
  OVERLAY_CNN,
  OVERLAY_NONE,
  attributionFrameIdentity,
  attributionPresentation,
  cnnFrameIdentity,
  cnnPresentation,
  cnnWinnerLegend,
  diagnosticActivity,
  drawAttributionOverlay,
  drawCnnWinnerOverlay,
  observationFrameIdentity,
  reconcileOverlaySelection,
  sameFrameIdentity,
} from "./diagnostic-overlays.js";
import { createPanel } from "./shared.js";

const FRAME_OBSERVATION = 2;
const FRAME_ATTRIBUTION = 3;
const FRAME_CNN_INSPECTION = 4;
const OVERLAY_DEFAULT_OPACITY = Object.freeze({
  [OVERLAY_ATTRIBUTION]: 0.45,
  [OVERLAY_CNN]: 0.55,
});

export { observationFrameIdentity };

export function mount({ definition }) {
  const element = createPanel({
    id: definition.id,
    label: definition.label,
    className: "observation-panel",
    body: `
      <div class="diagnostic-state" data-diagnostic-state hidden>
        <strong data-diagnostic-label></strong>
        <span data-diagnostic-detail></span>
      </div>
      <div class="observation-stage" hidden>
        <canvas data-observation-canvas></canvas>
        <canvas data-diagnostic-canvas class="diagnostic-overlay-canvas"></canvas>
      </div>
      <div class="diagnostic-context" data-diagnostic-context hidden>
        <span data-diagnostic-explanation></span>
        <div class="cnn-winner-legend" data-cnn-winner-legend aria-label="CNN winner-map filters"></div>
      </div>
      <pre data-input class="compact-pre">No policy input yet.</pre>
    `,
  });
  const baseCanvas = element.querySelector("[data-observation-canvas]");
  const overlayCanvas = element.querySelector("[data-diagnostic-canvas]");
  const stage = element.querySelector(".observation-stage");
  const status = element.querySelector("[data-diagnostic-state]");
  const statusLabel = element.querySelector("[data-diagnostic-label]");
  const statusDetail = element.querySelector("[data-diagnostic-detail]");
  const context = element.querySelector("[data-diagnostic-context]");
  const explanation = element.querySelector("[data-diagnostic-explanation]");
  const legend = element.querySelector("[data-cnn-winner-legend]");
  const input = element.querySelector("[data-input]");
  let snapshot = null;
  let targetSnapshot = null;
  let activity = { attribution: false, cnn: false };
  let activityInitialized = false;
  let selectedOverlay = OVERLAY_NONE;
  let baseBitmap = null;
  let baseIdentity = null;
  let preparedBaseBitmap = null;
  let preparedBaseIdentity = null;
  let attributionBitmap = null;
  let attributionIdentity = null;
  let cnnBitmap = null;
  let cnnIdentity = null;
  let baseBitmapRequest = 0;
  let attributionBitmapRequest = 0;
  let cnnBitmapRequest = 0;
  let mounted = true;

  const expectedBaseIdentity = () => observationFrameIdentity(snapshot);
  const expectedAttributionIdentity = () => attributionFrameIdentity(snapshot);
  const expectedCnnIdentity = () => cnnFrameIdentity(snapshot);
  const targetBaseIdentity = () => observationFrameIdentity(targetSnapshot);
  const targetAttributionIdentity = () => attributionFrameIdentity(targetSnapshot);
  const targetCnnIdentity = () => cnnFrameIdentity(targetSnapshot);
  const baseIsExact = () => sameFrameIdentity(baseIdentity, expectedBaseIdentity());
  const attributionIsExact = () => (
    baseIsExact() && sameFrameIdentity(attributionIdentity, expectedAttributionIdentity())
  );
  const cnnIsExact = () => (
    baseIsExact() && sameFrameIdentity(cnnIdentity, expectedCnnIdentity())
  );
  const closeBase = () => {
    baseBitmap?.close();
    baseBitmap = null;
    baseIdentity = null;
  };
  const closePreparedBase = () => {
    preparedBaseBitmap?.close();
    preparedBaseBitmap = null;
    preparedBaseIdentity = null;
  };
  const closeAttribution = () => {
    attributionBitmap?.close();
    attributionBitmap = null;
    attributionIdentity = null;
  };
  const closeCnn = () => {
    cnnBitmap?.close();
    cnnBitmap = null;
    cnnIdentity = null;
  };
  const selectedOpacity = () => OVERLAY_DEFAULT_OPACITY[selectedOverlay]
    ?? OVERLAY_DEFAULT_OPACITY[OVERLAY_ATTRIBUTION];
  const renderLegend = () => {
    const entries = selectedOverlay === OVERLAY_CNN && cnnIsExact()
      ? cnnWinnerLegend(snapshot)
      : [];
    legend.replaceChildren(...entries.map((item) => {
      const entry = document.createElement("span");
      entry.className = "cnn-winner-legend-item";
      const swatch = document.createElement("span");
      swatch.className = "cnn-filter-swatch";
      swatch.style.setProperty("--filter-color", item.color);
      swatch.setAttribute("aria-hidden", "true");
      const label = document.createElement("span");
      label.textContent = `Filter ${item.index}`;
      entry.append(swatch, label);
      return entry;
    }));
  };
  const updateContext = () => {
    if (selectedOverlay === OVERLAY_ATTRIBUTION) {
      context.hidden = false;
      explanation.textContent = "Attribution highlights policy-input regions associated with the selected policy action.";
    } else if (selectedOverlay === OVERLAY_CNN) {
      context.hidden = false;
      explanation.textContent = "The winner map identifies the displayed filter with the largest raw positive response at each location. It describes activation, not why the policy selected its action.";
    } else {
      context.hidden = true;
      explanation.textContent = "";
    }
    renderLegend();
  };
  const draw = () => {
    const exactBase = Boolean(baseBitmap && baseIsExact());
    stage.hidden = !exactBase;
    if (!exactBase) {
      baseCanvas.hidden = true;
      overlayCanvas.hidden = true;
      renderLegend();
      return;
    }
    baseCanvas.hidden = false;
    for (const canvas of [baseCanvas, overlayCanvas]) {
      canvas.width = baseBitmap.width;
      canvas.height = baseBitmap.height;
    }
    const baseContext = baseCanvas.getContext("2d", { alpha: false });
    baseContext.imageSmoothingEnabled = false;
    baseContext.drawImage(baseBitmap, 0, 0);
    const overlayContext = overlayCanvas.getContext("2d");
    overlayContext.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    let shown = false;
    if (selectedOverlay === OVERLAY_ATTRIBUTION && attributionBitmap && attributionIsExact()) {
      shown = drawAttributionOverlay(
        overlayContext,
        attributionBitmap,
        overlayCanvas.width,
        overlayCanvas.height,
      );
    } else if (selectedOverlay === OVERLAY_CNN && cnnBitmap && cnnIsExact()) {
      shown = drawCnnWinnerOverlay(
        overlayContext,
        cnnBitmap,
        snapshot,
        overlayCanvas.width,
        overlayCanvas.height,
      );
    }
    overlayCanvas.hidden = !shown;
    overlayCanvas.style.opacity = String(selectedOpacity());
    renderLegend();
  };
  const updateStatus = () => {
    status.hidden = selectedOverlay === OVERLAY_NONE;
    if (status.hidden) return;
    let presentation;
    if (selectedOverlay === OVERLAY_ATTRIBUTION) {
      presentation = attributionPresentation(snapshot, attributionIsExact());
    } else presentation = cnnPresentation(snapshot, cnnIsExact());
    status.dataset.kind = presentation.kind;
    statusLabel.textContent = presentation.label;
    statusDetail.textContent = presentation.detail;
  };
  const commitSnapshot = (nextSnapshot) => {
    snapshot = nextSnapshot;
    const nextActivity = diagnosticActivity(snapshot);
    selectedOverlay = reconcileOverlaySelection({
      selection: selectedOverlay,
      previousActivity: activity,
      activity: nextActivity,
      initialized: activityInitialized,
    });
    activity = nextActivity;
    activityInitialized = true;
    input.textContent = snapshot?.transition?.before?.model_input?.join("\n")
      || "No policy input yet.";
    updateStatus();
    updateContext();
    draw();
  };
  const promotePreparedBase = () => {
    if (!sameFrameIdentity(preparedBaseIdentity, targetBaseIdentity())) return false;
    closeBase();
    baseBitmap = preparedBaseBitmap;
    baseIdentity = preparedBaseIdentity;
    preparedBaseBitmap = null;
    preparedBaseIdentity = null;
    return true;
  };
  const prepareFrame = async (kind, blob, metadata = {}) => {
    if (kind !== FRAME_OBSERVATION || !blob) return false;
    const incoming = { sequence: Number(metadata.sequence) };
    if (
      sameFrameIdentity(incoming, baseIdentity)
      || sameFrameIdentity(incoming, preparedBaseIdentity)
    ) return true;
    const request = ++baseBitmapRequest;
    const bitmap = await createImageBitmap(blob);
    if (!mounted || request !== baseBitmapRequest) {
      bitmap.close();
      return true;
    }
    closePreparedBase();
    preparedBaseBitmap = bitmap;
    preparedBaseIdentity = incoming;
    return true;
  };

  return {
    element,
    render(nextSnapshot) {
      targetSnapshot = nextSnapshot;
      promotePreparedBase();
      if (sameFrameIdentity(baseIdentity, targetBaseIdentity())) {
        commitSnapshot(nextSnapshot);
      }
    },
    prepareFrame,
    async renderFrame(kind, blob, metadata = {}) {
      if (kind === FRAME_OBSERVATION) {
        const incoming = { sequence: Number(metadata.sequence) };
        if (!sameFrameIdentity(incoming, targetBaseIdentity())) return true;
        const frameSnapshot = targetSnapshot;
        if (!blob) {
          baseBitmapRequest += 1;
          closeBase();
          closePreparedBase();
          commitSnapshot(frameSnapshot);
          return true;
        }
        if (sameFrameIdentity(incoming, baseIdentity)) {
          commitSnapshot(frameSnapshot);
          return true;
        }
        await prepareFrame(kind, blob, metadata);
        if (
          sameFrameIdentity(incoming, targetBaseIdentity())
          && promotePreparedBase()
        ) commitSnapshot(frameSnapshot);
        return true;
      }
      if (kind === FRAME_ATTRIBUTION) {
        const incoming = {
          sequence: Number(metadata.sequence),
          generation: Number(metadata.generation),
        };
        if (!sameFrameIdentity(incoming, targetAttributionIdentity())) return true;
        const request = ++attributionBitmapRequest;
        if (!blob) {
          closeAttribution();
          updateStatus();
          draw();
          return true;
        }
        const bitmap = await createImageBitmap(blob);
        if (
          !mounted
          || request !== attributionBitmapRequest
          || !sameFrameIdentity(incoming, targetAttributionIdentity())
        ) {
          bitmap.close();
          return true;
        }
        closeAttribution();
        attributionBitmap = bitmap;
        attributionIdentity = incoming;
        updateStatus();
        draw();
        return true;
      }
      if (kind !== FRAME_CNN_INSPECTION) return false;
      const incoming = {
        sequence: Number(metadata.sequence),
        generation: Number(metadata.generation),
      };
      if (!sameFrameIdentity(incoming, targetCnnIdentity())) return true;
      const request = ++cnnBitmapRequest;
      if (!blob) {
        closeCnn();
        updateStatus();
        draw();
        return true;
      }
      const bitmap = await createImageBitmap(blob);
      if (
        !mounted
        || request !== cnnBitmapRequest
        || !sameFrameIdentity(incoming, targetCnnIdentity())
      ) {
        bitmap.close();
        return true;
      }
      closeCnn();
      cnnBitmap = bitmap;
      cnnIdentity = incoming;
      updateStatus();
      updateContext();
      draw();
      return true;
    },
    resetFrames() {
      baseBitmapRequest += 1;
      attributionBitmapRequest += 1;
      cnnBitmapRequest += 1;
      snapshot = null;
      targetSnapshot = null;
      closeBase();
      closePreparedBase();
      closeAttribution();
      closeCnn();
    },
    destroy() {
      mounted = false;
      baseBitmapRequest += 1;
      attributionBitmapRequest += 1;
      cnnBitmapRequest += 1;
      closeBase();
      closePreparedBase();
      closeAttribution();
      closeCnn();
    },
  };
}
