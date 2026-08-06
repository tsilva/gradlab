export const OVERLAY_NONE = "none";
export const OVERLAY_ATTRIBUTION = "attribution";
export const OVERLAY_CNN = "cnn";

function transitionSequence(snapshot) {
  const sequence = Number(snapshot?.transition?.sequence ?? snapshot?.sequence);
  return Number.isFinite(sequence) ? sequence : null;
}

export function observationFrameIdentity(snapshot) {
  const sequence = transitionSequence(snapshot);
  return sequence === null ? null : { sequence };
}

export function attributionFrameIdentity(snapshot) {
  const attribution = snapshot?.transition?.attribution;
  if (attribution?.status !== "available") return null;
  const sequence = transitionSequence(snapshot);
  const generation = Number(attribution.generation);
  if (sequence === null || !Number.isFinite(generation) || generation < 1) return null;
  return { sequence, generation };
}

export function cnnFrameIdentity(snapshot) {
  const cnn = snapshot?.transition?.cnn;
  if (cnn?.status !== "available" || !cnn.inspection) return null;
  const sequence = transitionSequence(snapshot);
  const generation = Number(cnn.generation);
  if (sequence === null || !Number.isFinite(generation) || generation < 1) return null;
  return { sequence, generation };
}

export function sameFrameIdentity(left, right) {
  if (!left || !right || Number(left.sequence) !== Number(right.sequence)) return false;
  const leftGenerated = left.generation !== undefined;
  const rightGenerated = right.generation !== undefined;
  if (leftGenerated !== rightGenerated) return false;
  return !leftGenerated || Number(left.generation) === Number(right.generation);
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
    if (transition.reason === "cadence") {
      return { kind: "skipped", label: "Cadence skipped", detail: "No map was computed for this step." };
    }
    if (transition.reason === "no_policy_decision") {
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
    return { kind: "off", label: "Off", detail: "Enable this panel to capture live CNN responses." };
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

export function diagnosticActivity(snapshot) {
  const attribution = snapshot?.session?.attribution || {};
  const supportedModes = snapshot?.policy?.attribution?.supported_modes;
  const cnn = snapshot?.session?.cnn || {};
  const cnnLayers = snapshot?.policy?.cnn?.layers;
  return {
    attribution: Array.isArray(supportedModes)
      && supportedModes.includes(attribution.mode)
      && attribution.mode !== "none"
      && attribution.status !== "off",
    cnn: Array.isArray(cnnLayers)
      && cnnLayers.length > 0
      && Boolean(cnn.enabled)
      && cnn.status !== "off",
  };
}

function preferredOverlay(activity) {
  if (activity.attribution) return OVERLAY_ATTRIBUTION;
  if (activity.cnn) return OVERLAY_CNN;
  return OVERLAY_NONE;
}

export function reconcileOverlaySelection({
  selection = OVERLAY_NONE,
  previousActivity = { attribution: false, cnn: false },
  activity = { attribution: false, cnn: false },
  initialized = false,
} = {}) {
  if (!initialized) return preferredOverlay(activity);
  const attributionStarted = !previousActivity.attribution && activity.attribution;
  const cnnStarted = !previousActivity.cnn && activity.cnn;
  if (attributionStarted) return OVERLAY_ATTRIBUTION;
  if (cnnStarted) return OVERLAY_CNN;
  if (selection === OVERLAY_ATTRIBUTION && !activity.attribution) {
    return preferredOverlay(activity);
  }
  if (selection === OVERLAY_CNN && !activity.cnn) return preferredOverlay(activity);
  return [OVERLAY_NONE, OVERLAY_ATTRIBUTION, OVERLAY_CNN].includes(selection)
    ? selection
    : preferredOverlay(activity);
}

export function drawAttributionOverlay(context, bitmap, width, height) {
  if (!context || !bitmap || width < 1 || height < 1) return false;
  context.imageSmoothingEnabled = false;
  context.drawImage(bitmap, 0, 0, width, height);
  return true;
}

export function drawCnnWinnerOverlay(context, bitmap, snapshot, width, height) {
  const atlas = snapshot?.transition?.cnn?.inspection?.atlas;
  const rect = atlasTileRect(atlas, atlas?.winner_tile);
  if (!context || !bitmap || !rect || width < 1 || height < 1) return false;
  const requestedFrameCount = Number(snapshot?.transition?.before?.observation_frames || 1);
  const frameCount = Number.isFinite(requestedFrameCount)
    ? Math.max(1, Math.floor(requestedFrameCount))
    : 1;
  const frameWidth = width / frameCount;
  context.imageSmoothingEnabled = true;
  for (let frame = 0; frame < frameCount; frame += 1) {
    context.drawImage(
      bitmap,
      rect.x,
      rect.y,
      rect.width,
      rect.height,
      frame * frameWidth,
      0,
      frameWidth,
      height,
    );
  }
  return true;
}

export function cnnWinnerLegend(snapshot) {
  const filters = snapshot?.transition?.cnn?.inspection?.filters;
  if (!Array.isArray(filters)) return [];
  return filters.flatMap((item) => {
    const index = Number(item?.filter_index);
    const color = String(item?.color || "").trim();
    return Number.isInteger(index) && color ? [{ index, color }] : [];
  });
}
