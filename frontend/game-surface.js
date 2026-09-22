import { createRenderSpeed } from "../src/gradlab/web_player/panels/render-speed.js";
import {
  gameFramePhase,
  gameFrameBoundaryKind,
  gameFrameTerminationDetail,
  gameFrameTerminationTone,
} from "../src/gradlab/web_player/panels/game.js";
const FRAME_GAME = 1;
export function createGameSurface(
  element,
  services,
  display,
  controls,
  displaySpeed,
) {
  const rgbToggle = element.querySelector("[data-rgb-toggle]");
  rgbToggle.addEventListener("click", () =>
    services.setRgbEnabled(services.getState().rgbEnabled === false),
  );
  const stage = element.querySelector(".game-stage");
  const viewport = element.querySelector(".game-viewport");
  const frame = element.querySelector(".game-frame");
  const canvas = element.querySelector("canvas");
  const empty = element.querySelector(".game-empty");
  const renderSpeed = createRenderSpeed(displaySpeed);
  const pressed = new Set();
  let focused = false;
  let aspect = 256 / 240;
  let targetSnapshot = null;
  let targetSequence = null;
  let frameSequence = null;
  let preparedBitmap = null;
  let preparedSequence = null;
  let preparedHasFrame = false;
  let preparedDecodeMs = 0;
  let bitmapRequest = 0;
  let mounted = true;
  const mapping = new Map([
    ["ArrowUp", "up"],
    ["ArrowDown", "down"],
    ["ArrowLeft", "left"],
    ["ArrowRight", "right"],
    ["z", "b"],
    ["Z", "b"],
    ["x", "a"],
    ["X", "a"],
    ["Enter", "start"],
    ["Shift", "select"],
  ]);

  const publish = (hasFocus = focused) =>
    services.send({
      type: "input",
      pressed: [...pressed],
      focused: hasFocus,
    });
  const fit = () => {
    const width = viewport.clientWidth;
    const height = viewport.clientHeight;
    if (!width || !height) return;
    const fittedWidth = Math.min(width, height * aspect);
    const fittedHeight = fittedWidth / aspect;
    frame.style.width = `${Math.max(1, Math.floor(fittedWidth))}px`;
    frame.style.height = `${Math.max(1, Math.floor(fittedHeight))}px`;
    frame.style.aspectRatio = String(aspect);
  };
  const resizeObserver = new ResizeObserver(fit);
  resizeObserver.observe(viewport);
  const loseFocus = () => {
    if (!focused && !pressed.size) return;
    focused = false;
    pressed.clear();
    publish(false);
  };
  const visibility = () => {
    if (document.hidden) loseFocus();
  };

  element.addEventListener("mousedown", (event) => {
    if (event.button !== 0) return;
    if (
      event.target.closest(
        "button, input, select, textarea, a, [role='slider'], [contenteditable='true']",
      )
    )
      return;
    const state = services.getState();
    if (state.hasControl && state.snapshot?.driver === "human") return;
    const scrubber = document.querySelector("#timeline-scrubber");
    if (!scrubber || scrubber.disabled) return;
    event.preventDefault();
    scrubber.focus({ preventScroll: true });
  });

  canvas.addEventListener("focus", () => {
    focused = true;
    publish(true);
  });
  canvas.addEventListener("blur", loseFocus);
  canvas.addEventListener("keydown", (event) => {
    const label = mapping.get(event.key);
    if (!label) return;
    event.preventDefault();
    pressed.add(label);
    publish(true);
  });
  canvas.addEventListener("keyup", (event) => {
    const label = mapping.get(event.key);
    if (!label) return;
    event.preventDefault();
    pressed.delete(label);
    publish(true);
  });
  document.addEventListener("visibilitychange", visibility);
  element.querySelector("[data-fullscreen]").addEventListener("click", () => {
    stage
      .requestFullscreen({ navigationUI: "hide" })
      .catch((error) => services.showToast(error.message, true));
  });
  const keepalive = setInterval(() => {
    const state = services.getState();
    if (focused && state.hasControl && state.snapshot?.driver === "human")
      publish(true);
  }, 50);

  const commitSnapshot = (snapshot) => {
    const phase = gameFramePhase(snapshot);
    const boundaryKind = gameFrameBoundaryKind(snapshot);
    const detail = gameFrameTerminationDetail(snapshot);
    const tone = gameFrameTerminationTone(snapshot);
    display({ phase, boundaryKind, detail, tone });
  };
  const clearPrepared = () => {
    preparedBitmap?.close();
    preparedBitmap = null;
    preparedSequence = null;
    preparedHasFrame = false;
    preparedDecodeMs = 0;
  };
  const commitPrepared = (snapshot) => {
    if (preparedSequence !== targetSequence) return false;
    if (!preparedHasFrame) {
      renderSpeed.reset();
      frameSequence = null;
      canvas.width = 1;
      canvas.height = 1;
      empty.textContent =
        "No exact post-action game frame was retained for this transition.";
      empty.hidden = false;
      preparedSequence = null;
      commitSnapshot(snapshot);
      return true;
    }
    if (preparedBitmap) {
      const drawStarted = performance.now();
      aspect = preparedBitmap.width / Math.max(1, preparedBitmap.height);
      fit();
      if (canvas.width !== preparedBitmap.width)
        canvas.width = preparedBitmap.width;
      if (canvas.height !== preparedBitmap.height)
        canvas.height = preparedBitmap.height;
      const context = canvas.getContext("2d", { alpha: false });
      context.imageSmoothingEnabled = false;
      context.drawImage(preparedBitmap, 0, 0);
      renderSpeed.record(preparedDecodeMs, performance.now() - drawStarted);
      preparedBitmap.close();
      preparedBitmap = null;
    }
    frameSequence = preparedSequence;
    preparedSequence = null;
    preparedHasFrame = false;
    empty.textContent = "This environment has no RGB renderer.";
    empty.hidden = true;
    commitSnapshot(snapshot);
    return true;
  };

  const prepareFrame = async (kind, blob, metadata = {}) => {
    if (kind !== FRAME_GAME) return false;
    if (services.getState().rgbEnabled === false) return true;
    const incomingSequence = Number(metadata.sequence);
    if (incomingSequence === frameSequence) {
      clearPrepared();
      preparedSequence = incomingSequence;
      preparedHasFrame = true;
      return true;
    }
    const request = ++bitmapRequest;
    if (!blob) {
      clearPrepared();
      preparedSequence = incomingSequence;
      return true;
    }
    const decodeStarted = performance.now();
    const bitmap = await createImageBitmap(blob);
    if (
      !mounted ||
      request !== bitmapRequest ||
      metadata.isCurrent?.() === false
    ) {
      bitmap.close();
      return true;
    }
    clearPrepared();
    preparedDecodeMs = performance.now() - decodeStarted;
    preparedBitmap = bitmap;
    preparedSequence = incomingSequence;
    preparedHasFrame = true;
    return true;
  };

  return {
    element,
    render(nextSnapshot) {
      const state = services.getState();
      const rgbEnabled = state.rgbEnabled !== false;
      renderSpeed.setPlaybackMode(!rgbEnabled);
      controls({ rgbEnabled, hasControl: state.hasControl });
      if (!rgbEnabled) {
        bitmapRequest += 1;
        clearPrepared();
        renderSpeed.recordProgress(
          nextSnapshot,
          Boolean(state.replayingInspection) ||
            (state.inspectionSequence == null &&
              ["playing", "stepping", "continuing"].includes(
                nextSnapshot?.run_state,
              )),
        );
        empty.textContent = "RGB hidden · Maximum playback speed";
        empty.hidden = false;
        commitSnapshot(nextSnapshot);
        return;
      }
      targetSnapshot = nextSnapshot;
      targetSequence = Number(nextSnapshot?.sequence);
      if (frameSequence === targetSequence) commitSnapshot(nextSnapshot);
      else commitPrepared(nextSnapshot);
    },
    prepareFrame,
    async renderFrame(kind, blob, metadata = {}) {
      if (kind !== FRAME_GAME) return false;
      if (services.getState().rgbEnabled === false) return true;
      const incomingSequence = Number(metadata.sequence);
      if (incomingSequence !== targetSequence) return true;
      const frameSnapshot = targetSnapshot;
      if (incomingSequence === frameSequence) {
        commitSnapshot(frameSnapshot);
        return true;
      }
      await prepareFrame(kind, blob, metadata);
      if (
        metadata.isCurrent?.() !== false &&
        incomingSequence === targetSequence
      )
        commitPrepared(frameSnapshot);
      return true;
    },
    resetFrames() {
      bitmapRequest += 1;
      renderSpeed.reset();
      clearPrepared();
      targetSnapshot = null;
      targetSequence = null;
      frameSequence = null;
    },
    resize: fit,
    destroy() {
      mounted = false;
      resizeObserver.disconnect();
      renderSpeed.destroy();
      bitmapRequest += 1;
      clearPrepared();
      loseFocus();
      clearInterval(keepalive);
      document.removeEventListener("visibilitychange", visibility);
    },
  };
}
