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
const METHOD_DEFAULT_INTERVAL = Object.freeze({ gradcam: 1, occlusion: 8 });
const OVERLAY_DEFAULT_OPACITY = Object.freeze({
  [OVERLAY_ATTRIBUTION]: 0.45,
  [OVERLAY_CNN]: 0.55,
});

export { attributionFrameIdentity, attributionPresentation, observationFrameIdentity };

function hiddenOverlayPresentation(activity) {
  if (activity.attribution || activity.cnn) {
    return {
      kind: "off",
      label: "Overlay hidden",
      detail: "Choose an active diagnostic to display over the observation.",
    };
  }
  return {
    kind: "off",
    label: "No active overlay",
    detail: "Enable attribution here or CNN capture in CNN feature explorer.",
  };
}

export function mount({ definition, services }) {
  const element = createPanel({
    id: definition.id,
    label: definition.label,
    className: "observation-panel",
    body: `
      <div class="observation-diagnostic-controls">
        <label>Attribution method
          <select data-attribution-method>
            <option value="none">Off</option>
            <option value="gradcam">Grad-CAM</option>
            <option value="occlusion">Occlusion</option>
          </select>
        </label>
        <label>Every N steps
          <input data-attribution-interval type="number" min="1" step="1" value="1" inputmode="numeric">
        </label>
        <label>Displayed overlay
          <select data-diagnostic-overlay>
            <option value="none">None</option>
            <option value="attribution">Attribution</option>
            <option value="cnn">CNN winner map</option>
          </select>
        </label>
        <label>Opacity <output data-overlay-opacity-output>45%</output>
          <input data-overlay-opacity type="range" min="0" max="1" step="0.05" value="0.45">
        </label>
      </div>
      <div class="diagnostic-state" data-diagnostic-state>
        <strong data-diagnostic-label>No active overlay</strong>
        <span data-diagnostic-detail>Enable attribution here or CNN capture in CNN feature explorer.</span>
      </div>
      <div class="observation-stage">
        <canvas data-observation-canvas></canvas>
        <canvas data-diagnostic-canvas class="diagnostic-overlay-canvas"></canvas>
        <div data-empty class="empty-state">No image stack is available.</div>
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
  const empty = element.querySelector("[data-empty]");
  const method = element.querySelector("[data-attribution-method]");
  const interval = element.querySelector("[data-attribution-interval]");
  const overlay = element.querySelector("[data-diagnostic-overlay]");
  const opacity = element.querySelector("[data-overlay-opacity]");
  const opacityOutput = element.querySelector("[data-overlay-opacity-output]");
  const status = element.querySelector("[data-diagnostic-state]");
  const statusLabel = element.querySelector("[data-diagnostic-label]");
  const statusDetail = element.querySelector("[data-diagnostic-detail]");
  const context = element.querySelector("[data-diagnostic-context]");
  const explanation = element.querySelector("[data-diagnostic-explanation]");
  const legend = element.querySelector("[data-cnn-winner-legend]");
  const input = element.querySelector("[data-input]");
  let snapshot = null;
  let activity = { attribution: false, cnn: false };
  let activityInitialized = false;
  let selectedOverlay = OVERLAY_NONE;
  const overlayOpacity = {
    [OVERLAY_ATTRIBUTION]: OVERLAY_DEFAULT_OPACITY[OVERLAY_ATTRIBUTION],
    [OVERLAY_CNN]: OVERLAY_DEFAULT_OPACITY[OVERLAY_CNN],
  };
  let baseBitmap = null;
  let baseIdentity = null;
  let attributionBitmap = null;
  let attributionIdentity = null;
  let cnnBitmap = null;
  let cnnIdentity = null;
  let baseBitmapRequest = 0;
  let attributionBitmapRequest = 0;
  let cnnBitmapRequest = 0;

  const expectedBaseIdentity = () => observationFrameIdentity(snapshot);
  const expectedAttributionIdentity = () => attributionFrameIdentity(snapshot);
  const expectedCnnIdentity = () => cnnFrameIdentity(snapshot);
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
  const selectedOpacity = () => overlayOpacity[selectedOverlay]
    ?? OVERLAY_DEFAULT_OPACITY[OVERLAY_ATTRIBUTION];
  const syncOpacity = () => {
    const enabled = selectedOverlay !== OVERLAY_NONE;
    const value = selectedOpacity();
    opacity.disabled = !enabled;
    opacity.value = String(value);
    opacityOutput.textContent = `${Math.round(value * 100)}%`;
  };
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
    if (!exactBase) {
      baseCanvas.width = 1;
      baseCanvas.height = 1;
      overlayCanvas.width = 1;
      overlayCanvas.height = 1;
      overlayCanvas.hidden = true;
      empty.hidden = false;
      renderLegend();
      return;
    }
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
    empty.hidden = true;
    renderLegend();
  };
  const updateStatus = () => {
    let presentation = hiddenOverlayPresentation(activity);
    if (selectedOverlay === OVERLAY_ATTRIBUTION) {
      presentation = attributionPresentation(snapshot, attributionIsExact());
    } else if (selectedOverlay === OVERLAY_CNN) {
      presentation = cnnPresentation(snapshot, cnnIsExact());
    }
    status.dataset.kind = presentation.kind;
    statusLabel.textContent = presentation.label;
    statusDetail.textContent = presentation.detail;
  };
  const applyOverlaySelection = (selection) => {
    selectedOverlay = selection;
    overlay.value = selection;
    syncOpacity();
    updateStatus();
    updateContext();
    draw();
  };
  const sendConfiguration = () => {
    if (method.value === "none") services.command("set_attribution", { mode: "none" });
    else services.command("set_attribution", {
      mode: method.value,
      interval: Number(interval.value),
    });
  };

  method.addEventListener("change", () => {
    if (method.value !== "none") {
      interval.value = String(METHOD_DEFAULT_INTERVAL[method.value] || 1);
    }
    sendConfiguration();
  });
  interval.addEventListener("change", () => {
    if (method.value === "none" || !interval.validity.valid) return;
    sendConfiguration();
  });
  overlay.addEventListener("change", () => applyOverlaySelection(overlay.value));
  opacity.addEventListener("input", () => {
    if (selectedOverlay === OVERLAY_NONE) return;
    overlayOpacity[selectedOverlay] = Number(opacity.value);
    opacityOutput.textContent = `${Math.round(Number(opacity.value) * 100)}%`;
    draw();
  });

  return {
    element,
    render(nextSnapshot) {
      snapshot = nextSnapshot;
      const capability = snapshot?.policy?.attribution || {};
      const supported = new Set(capability.supported_modes || []);
      [...method.options].forEach((option) => {
        option.disabled = option.value !== "none" && !supported.has(option.value);
      });
      const shared = snapshot?.session?.attribution || {};
      if (document.activeElement !== method) method.value = shared.mode || "none";
      if (document.activeElement !== interval) interval.value = String(shared.interval || 1);
      const hasControl = Boolean(services.getState().hasControl);
      method.disabled = !hasControl || supported.size === 0;
      interval.disabled = !hasControl || method.value === "none" || supported.size === 0;

      const nextActivity = diagnosticActivity(snapshot);
      selectedOverlay = reconcileOverlaySelection({
        selection: selectedOverlay,
        previousActivity: activity,
        activity: nextActivity,
        initialized: activityInitialized,
      });
      activity = nextActivity;
      activityInitialized = true;
      overlay.querySelector(`option[value="${OVERLAY_ATTRIBUTION}"]`).disabled = !activity.attribution;
      overlay.querySelector(`option[value="${OVERLAY_CNN}"]`).disabled = !activity.cnn;
      overlay.value = selectedOverlay;
      syncOpacity();

      if (!sameFrameIdentity(baseIdentity, expectedBaseIdentity())) closeBase();
      if (!sameFrameIdentity(attributionIdentity, expectedAttributionIdentity())) closeAttribution();
      if (!sameFrameIdentity(cnnIdentity, expectedCnnIdentity())) closeCnn();
      input.textContent = snapshot?.transition?.before?.model_input?.join("\n")
        || "No policy input yet.";
      updateStatus();
      updateContext();
      draw();
    },
    async renderFrame(kind, blob, metadata = {}) {
      if (kind === FRAME_OBSERVATION) {
        const incoming = { sequence: Number(metadata.sequence) };
        if (!sameFrameIdentity(incoming, expectedBaseIdentity())) return true;
        const request = ++baseBitmapRequest;
        if (!blob) {
          closeBase();
          empty.textContent = "No exact pre-action observation frame was retained for this transition.";
          updateStatus();
          draw();
          return true;
        }
        const bitmap = await createImageBitmap(blob);
        if (
          request !== baseBitmapRequest
          || !sameFrameIdentity(incoming, expectedBaseIdentity())
        ) {
          bitmap.close();
          return true;
        }
        closeBase();
        baseBitmap = bitmap;
        baseIdentity = incoming;
        empty.textContent = "No image stack is available.";
        updateStatus();
        draw();
        return true;
      }
      if (kind === FRAME_ATTRIBUTION) {
        const incoming = {
          sequence: Number(metadata.sequence),
          generation: Number(metadata.generation),
        };
        if (!sameFrameIdentity(incoming, expectedAttributionIdentity())) return true;
        const request = ++attributionBitmapRequest;
        if (!blob) {
          closeAttribution();
          updateStatus();
          draw();
          return true;
        }
        const bitmap = await createImageBitmap(blob);
        if (
          request !== attributionBitmapRequest
          || !sameFrameIdentity(incoming, expectedAttributionIdentity())
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
      if (!sameFrameIdentity(incoming, expectedCnnIdentity())) return true;
      const request = ++cnnBitmapRequest;
      if (!blob) {
        closeCnn();
        updateStatus();
        draw();
        return true;
      }
      const bitmap = await createImageBitmap(blob);
      if (request !== cnnBitmapRequest || !sameFrameIdentity(incoming, expectedCnnIdentity())) {
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
    destroy() {
      baseBitmapRequest += 1;
      attributionBitmapRequest += 1;
      cnnBitmapRequest += 1;
      closeBase();
      closeAttribution();
      closeCnn();
    },
  };
}
