import { createPanel } from "./shared.js";

const FRAME_OBSERVATION = 2;
const FRAME_CNN_INSPECTION = 4;

export function cnnFrameIdentity(snapshot) {
  const cnn = snapshot?.transition?.cnn;
  const inspection = cnn?.inspection;
  if (cnn?.status !== "available" || !inspection) return null;
  const sequence = Number(snapshot?.transition?.sequence ?? snapshot?.sequence);
  const generation = Number(cnn.generation);
  if (!Number.isFinite(sequence) || !Number.isFinite(generation) || generation < 1) return null;
  return { sequence, generation };
}

export function cnnPresentation(snapshot, hasExactFrame = false) {
  const capability = snapshot?.policy?.cnn || {};
  const layers = Array.isArray(capability.layers) ? capability.layers : [];
  const session = snapshot?.session?.cnn || {};
  const transition = snapshot?.transition?.cnn;
  if (!layers.length) {
    return {
      kind: "unavailable",
      label: "Unavailable",
      detail: capability.unavailable_reason
        || session.unavailable_reason
        || "This playback source has no inspectable actor CNN.",
    };
  }
  if (session.status === "error" || transition?.status === "error") {
    return {
      kind: "error",
      label: "Error",
      detail: transition?.reason || session.error || "CNN inspection failed.",
    };
  }
  if (!session.enabled || session.status === "off") {
    return { kind: "off", label: "Off", detail: "Enable inspection to capture live CNN responses." };
  }
  if (!transition) {
    return { kind: "computing", label: "Active", detail: "The next observation will be inspected." };
  }
  if (transition.status === "available") {
    return hasExactFrame
      ? { kind: "available", label: "Exact", detail: "Filter responses match this transition." }
      : { kind: "computing", label: "Computing", detail: "Waiting for the exact activation atlas." };
  }
  if (transition.status === "not_computed") {
    if (transition.reason === "cadence") {
      return { kind: "skipped", label: "Cadence skipped", detail: "No CNN capture was requested for this step." };
    }
    return { kind: "unobserved", label: "No image input", detail: "This transition has no compatible policy image." };
  }
  return { kind: "off-step", label: "Off for this step", detail: "Inspection was disabled for this transition." };
}

export function atlasTileRect(atlas, tileIndex) {
  const columns = Number(atlas?.columns);
  const tileWidth = Number(atlas?.tile_width);
  const tileHeight = Number(atlas?.tile_height);
  const index = Number(tileIndex);
  if (
    !Number.isInteger(columns) || columns < 1
    || !Number.isFinite(tileWidth) || tileWidth < 1
    || !Number.isFinite(tileHeight) || tileHeight < 1
    || !Number.isInteger(index) || index < 0
  ) return null;
  return {
    x: (index % columns) * tileWidth,
    y: Math.floor(index / columns) * tileHeight,
    width: tileWidth,
    height: tileHeight,
  };
}

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

function sameIdentity(left, right) {
  return Boolean(
    left && right
    && Number(left.sequence) === Number(right.sequence)
    && Number(left.generation) === Number(right.generation)
  );
}

export function mount({ definition, services }) {
  const element = createPanel({
    id: definition.id,
    label: definition.label,
    className: "cnn-panel",
    body: `
      <div class="cnn-controls">
        <label class="cnn-enable"><input data-cnn-enabled type="checkbox"> Inspect CNN</label>
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
        <label class="cnn-local-toggle"><input data-cnn-winners type="checkbox" checked> Show winner map</label>
        <label>Map opacity <output data-cnn-opacity-output>55%</output>
          <input data-cnn-opacity type="range" min="0" max="1" step="0.05" value="0.55">
        </label>
      </div>
      <div class="cnn-state" data-cnn-state>
        <strong data-cnn-label>Off</strong>
        <span data-cnn-detail>Enable inspection to capture live CNN responses.</span>
      </div>
      <div class="cnn-overview">
        <div class="cnn-observation-stage">
          <canvas data-cnn-observation></canvas>
          <canvas data-cnn-winner class="cnn-winner-canvas"></canvas>
          <div data-cnn-empty class="empty-state">No exact policy observation is available.</div>
        </div>
        <div class="cnn-explanation">
          <strong>Winner map</strong>
          <span>At each feature-map cell, color identifies the shown filter with the largest raw positive response; opacity is response strength.</span>
          <span>Peak regions below use the layer's exact stride and receptive field in policy-input pixels.</span>
        </div>
      </div>
      <div class="cnn-filter-grid" data-cnn-filters></div>
      <p class="panel-foot">Kernel tiles show every learned input-channel plane within the filter's convolution group: amber is positive, cyan is negative. Activation tiles are normalized within each filter; ranking and winner colors use unnormalized responses. These views explain the representation, not why the policy selected its action.</p>
    `,
  });
  const enabled = element.querySelector("[data-cnn-enabled]");
  const layer = element.querySelector("[data-cnn-layer]");
  const interval = element.querySelector("[data-cnn-interval]");
  const topK = element.querySelector("[data-cnn-top-k]");
  const winners = element.querySelector("[data-cnn-winners]");
  const opacity = element.querySelector("[data-cnn-opacity]");
  const opacityOutput = element.querySelector("[data-cnn-opacity-output]");
  const status = element.querySelector("[data-cnn-state]");
  const statusLabel = element.querySelector("[data-cnn-label]");
  const statusDetail = element.querySelector("[data-cnn-detail]");
  const observationCanvas = element.querySelector("[data-cnn-observation]");
  const winnerCanvas = element.querySelector("[data-cnn-winner]");
  const empty = element.querySelector("[data-cnn-empty]");
  const filters = element.querySelector("[data-cnn-filters]");
  let snapshot = null;
  let observationBitmap = null;
  let atlasBitmap = null;
  let atlasIdentity = null;
  let layerKey = "";
  let bitmapRequest = 0;
  let observationRequest = 0;

  const expectedIdentity = () => cnnFrameIdentity(snapshot);
  const inspection = () => snapshot?.transition?.cnn?.inspection || null;
  const hasExactAtlas = () => sameIdentity(atlasIdentity, expectedIdentity());
  const closeAtlas = () => {
    atlasBitmap?.close();
    atlasBitmap = null;
    atlasIdentity = null;
  };
  const drawWinner = () => {
    if (!observationBitmap) {
      observationCanvas.width = 1;
      observationCanvas.height = 1;
      winnerCanvas.width = 1;
      winnerCanvas.height = 1;
      empty.hidden = false;
      return;
    }
    for (const canvas of [observationCanvas, winnerCanvas]) {
      canvas.width = observationBitmap.width;
      canvas.height = observationBitmap.height;
    }
    const base = observationCanvas.getContext("2d", { alpha: false });
    base.imageSmoothingEnabled = false;
    base.drawImage(observationBitmap, 0, 0);
    const overlay = winnerCanvas.getContext("2d");
    overlay.clearRect(0, 0, winnerCanvas.width, winnerCanvas.height);
    const metadata = inspection();
    const rect = atlasTileRect(metadata?.atlas, metadata?.atlas?.winner_tile);
    const frameCount = Math.max(1, Number(snapshot?.transition?.before?.observation_frames || 1));
    const frameWidth = winnerCanvas.width / frameCount;
    const show = winners.checked && atlasBitmap && hasExactAtlas() && rect;
    if (show) {
      for (let frame = 0; frame < frameCount; frame += 1) {
        overlay.drawImage(
          atlasBitmap,
          rect.x,
          rect.y,
          rect.width,
          rect.height,
          frame * frameWidth,
          0,
          frameWidth,
          winnerCanvas.height,
        );
      }
    }
    winnerCanvas.hidden = !show;
    winnerCanvas.style.opacity = opacity.value;
    empty.hidden = true;
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
    if (!enabled.checked) {
      services.command("set_cnn_inspection", { enabled: false });
      return;
    }
    services.command("set_cnn_inspection", {
      enabled: true,
      layer_id: layer.value,
      interval: Number(interval.value),
      top_k: Number(topK.value),
    });
  };

  enabled.addEventListener("change", sendConfiguration);
  layer.addEventListener("change", sendConfiguration);
  interval.addEventListener("change", () => {
    if (enabled.checked && interval.validity.valid) sendConfiguration();
  });
  topK.addEventListener("change", sendConfiguration);
  winners.addEventListener("change", drawWinner);
  opacity.addEventListener("input", () => {
    opacityOutput.textContent = `${Math.round(Number(opacity.value) * 100)}%`;
    drawWinner();
  });

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
      if (document.activeElement !== enabled) enabled.checked = Boolean(shared.enabled);
      if (document.activeElement !== layer && shared.layer_id) layer.value = shared.layer_id;
      if (document.activeElement !== interval) interval.value = String(shared.interval || 1);
      if (document.activeElement !== topK) topK.value = String(shared.top_k || 12);
      const hasControl = Boolean(services.getState().hasControl);
      enabled.disabled = !hasControl || !layers.length;
      layer.disabled = !hasControl || !enabled.checked || !layers.length;
      interval.disabled = !hasControl || !enabled.checked || !layers.length;
      topK.disabled = !hasControl || !enabled.checked || !layers.length;
      if (!sameIdentity(atlasIdentity, expectedIdentity())) closeAtlas();
      renderFilters();
      drawWinner();
      updateStatus();
    },
    async renderFrame(kind, blob, metadata = {}) {
      if (kind === FRAME_OBSERVATION) {
        const request = ++observationRequest;
        const bitmap = blob ? await createImageBitmap(blob) : null;
        if (request !== observationRequest) {
          bitmap?.close();
          return true;
        }
        const previous = observationBitmap;
        observationBitmap = bitmap;
        previous?.close();
        drawWinner();
        return true;
      }
      if (kind !== FRAME_CNN_INSPECTION) return false;
      const incoming = {
        sequence: Number(metadata.sequence),
        generation: Number(metadata.generation),
      };
      const request = ++bitmapRequest;
      if (!blob || !sameIdentity(incoming, expectedIdentity())) {
        closeAtlas();
        renderFilters();
        drawWinner();
        updateStatus();
        return true;
      }
      const bitmap = await createImageBitmap(blob);
      if (request !== bitmapRequest || !sameIdentity(incoming, expectedIdentity())) {
        bitmap.close();
        return true;
      }
      closeAtlas();
      atlasBitmap = bitmap;
      atlasIdentity = incoming;
      renderFilters();
      drawWinner();
      updateStatus();
      return true;
    },
    destroy() {
      bitmapRequest += 1;
      observationRequest += 1;
      observationBitmap?.close();
      closeAtlas();
    },
  };
}
