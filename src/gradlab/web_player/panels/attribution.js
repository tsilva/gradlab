import {
  attributionFrameIdentity,
  attributionPresentation,
} from "./diagnostic-overlays.js";
import { createPanel } from "./shared.js";

const METHOD_DEFAULT_INTERVAL = Object.freeze({ gradcam: 1, occlusion: 8 });

export { attributionFrameIdentity, attributionPresentation };

export function mount({ definition, services }) {
  const element = createPanel({
    id: definition.id,
    label: definition.label,
    className: "attribution-panel",
    body: `
      <div class="attribution-controls">
        <label>Method
          <select data-attribution-method>
            <option value="gradcam">Grad-CAM</option>
            <option value="occlusion">Occlusion</option>
          </select>
        </label>
        <label>Every N steps
          <input data-attribution-interval type="number" min="1" step="1" value="1" inputmode="numeric">
        </label>
      </div>
      <div class="attribution-state" data-attribution-state>
        <strong data-attribution-label>Off</strong>
        <span data-attribution-detail>Enable this panel to compute live attribution.</span>
      </div>
      <div class="attribution-explanation">
        <strong>Shared spatial view</strong>
        <span>The Input panel overlays the selected method on the exact model input while this panel is enabled.</span>
        <span>Attribution associates input regions with the selected policy action without changing the shared trajectory.</span>
      </div>
      <p class="panel-foot">Grad-CAM is fast and convolution-specific. Occlusion is slower, but directly measures the selected action score after masking input regions.</p>
    `,
  });
  const method = element.querySelector("[data-attribution-method]");
  const interval = element.querySelector("[data-attribution-interval]");
  const status = element.querySelector("[data-attribution-state]");
  const statusLabel = element.querySelector("[data-attribution-label]");
  const statusDetail = element.querySelector("[data-attribution-detail]");
  let snapshot = null;

  const sendConfiguration = () => {
    const config = { mode: method.value, interval: Number(interval.value) };
    services.setAttributionPreference?.(config);
    services.command("set_attribution", config);
  };

  method.addEventListener("change", () => {
    interval.value = String(METHOD_DEFAULT_INTERVAL[method.value] || 1);
    sendConfiguration();
  });
  interval.addEventListener("change", () => {
    if (!interval.validity.valid) return;
    sendConfiguration();
  });

  return {
    element,
    render(nextSnapshot) {
      snapshot = nextSnapshot;
      const capability = snapshot?.policy?.attribution || {};
      const supported = new Set(capability.supported_modes || []);
      [...method.options].forEach((option) => {
        option.disabled = !supported.has(option.value);
      });
      const shared = snapshot?.session?.attribution || {};
      const preference = services.getState().attributionPreference || {};
      const selected = supported.has(shared.mode)
        ? shared.mode
        : supported.has(preference.mode)
          ? preference.mode
          : [...supported][0];
      if (document.activeElement !== method && selected) method.value = selected;
      if (document.activeElement !== interval) {
        interval.value = String(
          supported.has(shared.mode)
            ? shared.interval || METHOD_DEFAULT_INTERVAL[selected] || 1
            : preference.interval || METHOD_DEFAULT_INTERVAL[selected] || 1,
        );
      }
      const hasControl = Boolean(services.getState().hasControl);
      method.disabled = !hasControl || supported.size === 0;
      interval.disabled = !hasControl || supported.size === 0;

      const presentation = attributionPresentation(
        snapshot,
        snapshot?.transition?.attribution?.status === "available",
      );
      status.dataset.kind = presentation.kind;
      statusLabel.textContent = presentation.label;
      statusDetail.textContent = presentation.detail;
    },
  };
}
