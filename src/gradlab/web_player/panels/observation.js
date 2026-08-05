import { createPanel } from "./shared.js";

const FRAME_OBSERVATION = 2;
const FRAME_ATTRIBUTION = 3;
const METHOD_DEFAULT_INTERVAL = Object.freeze({ gradcam: 1, occlusion: 8 });

export function attributionFrameIdentity(snapshot) {
  const attribution = snapshot?.transition?.attribution;
  if (attribution?.status !== "available") return null;
  const sequence = Number(snapshot?.transition?.sequence ?? snapshot?.sequence);
  const generation = Number(attribution.generation);
  if (!Number.isFinite(sequence) || !Number.isFinite(generation) || generation < 1) return null;
  return { sequence, generation };
}

export function attributionPresentation(snapshot, hasExactFrame = false) {
  const capability = snapshot?.policy?.attribution || {};
  const supported = Array.isArray(capability.supported_modes)
    ? capability.supported_modes
    : [];
  const session = snapshot?.session?.attribution || {};
  const transition = snapshot?.transition?.attribution;
  if (!supported.length) {
    return {
      kind: "unavailable",
      label: "Unavailable",
      detail: capability.unavailable_reason
        || session.unavailable_reason
        || "This playback source does not expose live-policy attribution.",
    };
  }
  if (session.status === "error" || transition?.status === "error") {
    return {
      kind: "error",
      label: "Error",
      detail: transition?.reason || session.error || "Attribution failed.",
    };
  }
  if (session.mode === "none" || session.status === "off") {
    return { kind: "off", label: "Off", detail: "Attribution is disabled." };
  }
  if (!transition) {
    return {
      kind: "computing",
      label: "Active",
      detail: "The next policy decision will produce attribution.",
    };
  }
  if (transition.status === "available") {
    return hasExactFrame
      ? { kind: "available", label: "Available", detail: "Exact transition overlay." }
      : { kind: "computing", label: "Computing", detail: "Waiting for the exact overlay frame." };
  }
  if (transition.status === "not_computed") {
    const reason = transition.reason;
    if (reason === "cadence") {
      return { kind: "skipped", label: "Cadence skipped", detail: "No map was computed for this step." };
    }
    if (reason === "no_policy_decision") {
      return { kind: "unobserved", label: "No policy decision", detail: "This action was not selected by the policy." };
    }
    return { kind: "unobserved", label: "Not computed", detail: "No compatible image input was available." };
  }
  return {
    kind: "off-step",
    label: "Off for this step",
    detail: "Attribution was disabled when this transition was recorded.",
  };
}

export function mount({ definition, services }) {
  const element = createPanel({
    id: definition.id,
    label: definition.label,
    className: "observation-panel",
    body: `
      <div class="attribution-controls">
        <label>Method
          <select data-attribution-method>
            <option value="none">Off</option>
            <option value="gradcam">Grad-CAM</option>
            <option value="occlusion">Occlusion</option>
          </select>
        </label>
        <label>Every N steps
          <input data-attribution-interval type="number" min="1" step="1" value="1" inputmode="numeric">
        </label>
        <label class="attribution-local-toggle"><input data-attribution-visible type="checkbox" checked> Show overlay</label>
        <label>Opacity <output data-attribution-opacity-output>45%</output>
          <input data-attribution-opacity type="range" min="0" max="1" step="0.05" value="0.45">
        </label>
      </div>
      <div class="attribution-state" data-attribution-state>
        <strong data-attribution-label>Off</strong>
        <span data-attribution-detail>Attribution is disabled.</span>
      </div>
      <div class="observation-stage">
        <canvas data-observation-canvas></canvas>
        <canvas data-attribution-canvas class="attribution-canvas"></canvas>
        <div data-empty class="empty-state">No image stack is available.</div>
      </div>
      <pre data-input class="compact-pre">No policy input yet.</pre>
    `,
  });
  const baseCanvas = element.querySelector("[data-observation-canvas]");
  const overlayCanvas = element.querySelector("[data-attribution-canvas]");
  const empty = element.querySelector("[data-empty]");
  const method = element.querySelector("[data-attribution-method]");
  const interval = element.querySelector("[data-attribution-interval]");
  const visible = element.querySelector("[data-attribution-visible]");
  const opacity = element.querySelector("[data-attribution-opacity]");
  const opacityOutput = element.querySelector("[data-attribution-opacity-output]");
  const status = element.querySelector("[data-attribution-state]");
  const statusLabel = element.querySelector("[data-attribution-label]");
  const statusDetail = element.querySelector("[data-attribution-detail]");
  let snapshot = null;
  let baseBitmap = null;
  let overlayBitmap = null;
  let overlayIdentity = null;
  let baseBitmapRequest = 0;
  let overlayBitmapRequest = 0;

  const expectedIdentity = () => attributionFrameIdentity(snapshot);
  const identityMatches = (left, right) => Boolean(
    left && right
    && Number(left.sequence) === Number(right.sequence)
    && Number(left.generation) === Number(right.generation)
  );
  const closeOverlay = () => {
    overlayBitmap?.close();
    overlayBitmap = null;
    overlayIdentity = null;
  };
  const draw = () => {
    if (!baseBitmap) {
      baseCanvas.width = 1;
      baseCanvas.height = 1;
      overlayCanvas.width = 1;
      overlayCanvas.height = 1;
      empty.hidden = false;
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
    const show = visible.checked && identityMatches(overlayIdentity, expectedIdentity());
    if (show && overlayBitmap) {
      overlayContext.imageSmoothingEnabled = false;
      overlayContext.drawImage(overlayBitmap, 0, 0);
    }
    overlayCanvas.hidden = !show;
    overlayCanvas.style.opacity = opacity.value;
    empty.hidden = true;
  };
  const updateStatus = () => {
    const presentation = attributionPresentation(
      snapshot,
      identityMatches(overlayIdentity, expectedIdentity()),
    );
    status.dataset.kind = presentation.kind;
    statusLabel.textContent = presentation.label;
    statusDetail.textContent = presentation.detail;
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
  visible.addEventListener("change", () => { draw(); updateStatus(); });
  opacity.addEventListener("input", () => {
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
      if (!identityMatches(overlayIdentity, expectedIdentity())) closeOverlay();
      element.querySelector("[data-input]").textContent = snapshot?.transition?.before?.model_input?.join("\n")
        || "No policy input yet.";
      updateStatus();
      draw();
    },
    async renderFrame(kind, blob, metadata = {}) {
      if (kind === FRAME_OBSERVATION) {
        const request = ++baseBitmapRequest;
        const bitmap = blob ? await createImageBitmap(blob) : null;
        if (request !== baseBitmapRequest) {
          bitmap?.close();
          return true;
        }
        const previous = baseBitmap;
        baseBitmap = bitmap;
        previous?.close();
        empty.textContent = blob
          ? "No image stack is available."
          : "No exact pre-action observation frame was retained for this transition.";
        draw();
        return true;
      }
      if (kind !== FRAME_ATTRIBUTION) return false;
      const incoming = {
        sequence: Number(metadata.sequence),
        generation: Number(metadata.generation),
      };
      const request = ++overlayBitmapRequest;
      if (!blob || !identityMatches(incoming, expectedIdentity())) {
        closeOverlay();
        updateStatus();
        draw();
        return true;
      }
      const bitmap = await createImageBitmap(blob);
      if (request !== overlayBitmapRequest || !identityMatches(incoming, expectedIdentity())) {
        bitmap.close();
        return true;
      }
      closeOverlay();
      overlayBitmap = bitmap;
      overlayIdentity = incoming;
      updateStatus();
      draw();
      return true;
    },
    destroy() {
      baseBitmapRequest += 1;
      overlayBitmapRequest += 1;
      baseBitmap?.close();
      closeOverlay();
    },
  };
}
