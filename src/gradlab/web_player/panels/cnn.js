import {
  atlasTileRect,
  cnnFrameIdentity,
  cnnPresentation,
  sameFrameIdentity,
} from "./diagnostic-overlays.js";
import { createPanel } from "./shared.js";

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

export function mount({ definition, services }) {
  const element = createPanel({
    id: definition.id,
    label: definition.label,
    className: "cnn-panel",
    body: `
      <div class="cnn-controls">
        <label>Layer <select data-cnn-layer></select></label>
        <label>Every N steps <input data-cnn-interval type="number" min="1" step="1" value="1" inputmode="numeric"></label>
        <label>Top filters
          <select data-cnn-top-k>
            <option value="4">4</option>
            <option value="8">8</option>
            <option value="12" selected>12</option>
            <option value="16">16</option>
            <option value="24">24</option>
            <option value="32">32</option>
          </select>
        </label>
      </div>
      <div class="cnn-state" data-cnn-state>
        <strong data-cnn-label>Off</strong>
        <span data-cnn-detail>Enable this panel to capture live CNN responses.</span>
      </div>
      <div class="cnn-explanation">
        <strong>Shared spatial view</strong>
        <span>The Input panel displays the selected layer's winner map over the exact model input.</span>
        <span>Peak regions below use the layer's exact stride and receptive field in input pixels.</span>
      </div>
      <div class="cnn-filter-grid" data-cnn-filters></div>
      <p class="panel-foot">Kernel tiles show every learned input-channel plane within the filter's convolution group: amber is positive, cyan is negative. Activation tiles are normalized within each filter; ranking and winner colors use unnormalized responses. These views explain the representation, not why the policy selected its action.</p>
    `,
  });
  const layer = element.querySelector("[data-cnn-layer]");
  const interval = element.querySelector("[data-cnn-interval]");
  const topK = element.querySelector("[data-cnn-top-k]");
  const status = element.querySelector("[data-cnn-state]");
  const statusLabel = element.querySelector("[data-cnn-label]");
  const statusDetail = element.querySelector("[data-cnn-detail]");
  const filters = element.querySelector("[data-cnn-filters]");
  let snapshot = null;
  let atlasBitmap = null;
  let atlasIdentity = null;
  let layerKey = "";
  let bitmapRequest = 0;

  const expectedIdentity = () => cnnFrameIdentity(snapshot);
  const inspection = () => snapshot?.transition?.cnn?.inspection || null;
  const hasExactAtlas = () => sameFrameIdentity(atlasIdentity, expectedIdentity());
  const closeAtlas = () => {
    atlasBitmap?.close();
    atlasBitmap = null;
    atlasIdentity = null;
  };
  const drawTile = (canvas, tileIndex, smooth) => {
    const rect = atlasTileRect(inspection()?.atlas, tileIndex);
    if (!atlasBitmap || !hasExactAtlas() || !rect) return;
    canvas.width = rect.width;
    canvas.height = rect.height;
    const context = canvas.getContext("2d");
    context.imageSmoothingEnabled = smooth;
    context.drawImage(
      atlasBitmap,
      rect.x,
      rect.y,
      rect.width,
      rect.height,
      0,
      0,
      rect.width,
      rect.height,
    );
  };
  const renderFilters = () => {
    const entries = inspection()?.filters || [];
    if (!entries.length) {
      filters.replaceChildren();
      return;
    }
    filters.replaceChildren(...entries.map((item) => {
      const card = document.createElement("article");
      card.className = "cnn-filter-card";
      card.style.setProperty("--filter-color", item.color);
      card.innerHTML = `
        <header><span class="cnn-filter-swatch"></span><strong></strong><small></small></header>
        <div class="cnn-filter-visuals">
          <figure><canvas data-kernel></canvas><figcaption>Kernel weights</figcaption></figure>
          <figure><canvas data-activation></canvas><figcaption>Activation</figcaption></figure>
        </div>
        <dl>
          <div><dt>Peak</dt><dd></dd></div>
          <div><dt>Mean +</dt><dd></dd></div>
          <div><dt>Coverage</dt><dd></dd></div>
        </dl>
        <p class="cnn-region"></p>
      `;
      card.querySelector("strong").textContent = `Filter ${item.filter_index}`;
      card.querySelector("small").textContent = `rank ${item.rank}`;
      const values = card.querySelectorAll("dd");
      values[0].textContent = response(item.peak_response);
      values[1].textContent = response(item.mean_positive_response);
      values[2].textContent = Number.isFinite(Number(item.positive_coverage))
        ? `${Math.round(Number(item.positive_coverage) * 100)}%`
        : "—";
      card.querySelector(".cnn-region").textContent = `Strongest at ${peakRegionLabel(item.peak_input_region)}`;
      drawTile(card.querySelector("[data-kernel]"), item.kernel_tile, false);
      drawTile(card.querySelector("[data-activation]"), item.activation_tile, true);
      return card;
    }));
  };
  const updateStatus = () => {
    const presentation = cnnPresentation(snapshot, hasExactAtlas());
    status.dataset.kind = presentation.kind;
    statusLabel.textContent = presentation.label;
    statusDetail.textContent = presentation.detail;
  };
  const sendConfiguration = () => {
    services.command("set_cnn_inspection", {
      enabled: true,
      layer_id: layer.value,
      interval: Number(interval.value),
      top_k: Number(topK.value),
    });
  };

  layer.addEventListener("change", sendConfiguration);
  interval.addEventListener("change", () => {
    if (interval.validity.valid) sendConfiguration();
  });
  topK.addEventListener("change", sendConfiguration);

  return {
    element,
    render(nextSnapshot) {
      snapshot = nextSnapshot;
      const capability = snapshot?.policy?.cnn || {};
      const layers = Array.isArray(capability.layers) ? capability.layers : [];
      const nextLayerKey = JSON.stringify(layers.map((item) => [item.id, item.label]));
      if (nextLayerKey !== layerKey) {
        layerKey = nextLayerKey;
        layer.replaceChildren(...layers.map((item) => {
          const option = document.createElement("option");
          option.value = item.id;
          option.textContent = item.label;
          return option;
        }));
      }
      const shared = snapshot?.session?.cnn || {};
      if (document.activeElement !== layer && shared.layer_id) layer.value = shared.layer_id;
      if (document.activeElement !== interval) interval.value = String(shared.interval || 1);
      if (document.activeElement !== topK) topK.value = String(shared.top_k || 12);
      const hasControl = Boolean(services.getState().hasControl);
      layer.disabled = !hasControl || !layers.length;
      interval.disabled = !hasControl || !layers.length;
      topK.disabled = !hasControl || !layers.length;
      if (!sameFrameIdentity(atlasIdentity, expectedIdentity())) closeAtlas();
      renderFilters();
      updateStatus();
    },
    async renderFrame(kind, blob, metadata = {}) {
      if (kind !== FRAME_CNN_INSPECTION) return false;
      const incoming = {
        sequence: Number(metadata.sequence),
        generation: Number(metadata.generation),
      };
      const request = ++bitmapRequest;
      if (!blob || !sameFrameIdentity(incoming, expectedIdentity())) {
        closeAtlas();
        renderFilters();
        updateStatus();
        return true;
      }
      const bitmap = await createImageBitmap(blob);
      if (request !== bitmapRequest || !sameFrameIdentity(incoming, expectedIdentity())) {
        bitmap.close();
        return true;
      }
      closeAtlas();
      atlasBitmap = bitmap;
      atlasIdentity = incoming;
      renderFilters();
      updateStatus();
      return true;
    },
    destroy() {
      bitmapRequest += 1;
      closeAtlas();
    },
  };
}
