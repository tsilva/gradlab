<script lang="ts">
  import { onMount } from "svelte";
  import { createGameSurface } from "../game-surface.js";
  let { definition, services } = $props();
  let element: HTMLElement;
  let speed = $state({ value: "— FPS", path: "", title: "" });
  let frame = $state({
    phase: "Initial observation",
    boundaryKind: "",
    detail: "",
    tone: "",
  });
  let rgbEnabled = $state(true),
    hasControl = $state(false);
  let rgbLabel = $derived(
    rgbEnabled
      ? "Hide RGB and play at maximum speed"
      : "Show RGB and restore configured speed",
  );
  let surface: ReturnType<typeof createGameSurface>;
  onMount(() => {
    surface = createGameSurface(
      element,
      services,
      (value: typeof frame) => (frame = value),
      (value: { rgbEnabled: boolean; hasControl: boolean }) => {
        rgbEnabled = value.rgbEnabled;
        hasControl = value.hasControl;
      },
      (value: typeof speed) => {
        speed = value;
      },
    );
    return () => surface.destroy();
  });
  export const render = (snapshot: unknown) => surface?.render(snapshot);
  export const prepareFrame = (
    ...args: Parameters<typeof surface.prepareFrame>
  ) => surface?.prepareFrame(...args);
  export const renderFrame = (
    ...args: Parameters<typeof surface.renderFrame>
  ) => surface?.renderFrame(...args);
  export const resetFrames = () => surface?.resetFrames();
  export const resize = () => surface?.resize();
</script>

<section
  class="panel game-panel"
  data-panel={definition.id}
  bind:this={element}
>
  <div id="game-stage" class="game-stage">
    <div class="game-viewport">
      <div id="game-frame" class="game-frame">
        <canvas
          id="game-canvas"
          tabindex="0"
          hidden={!rgbEnabled}
          aria-label={`${frame.phase}.${frame.boundaryKind ? ` ${frame.boundaryKind}.` : ""}${frame.detail ? ` ${frame.detail}.` : ""} Focus for human controls: arrows move, Z is B, X is A, Enter is Start, and Shift is Select.`}
        ></canvas>
      </div>
      <div id="game-empty" class="game-empty empty-state">
        This environment has no RGB renderer.
      </div>
    </div>
    <div class="game-frame-status">
      <div class="game-frame-phase" data-frame-phase>{frame.phase}</div>
      <div
        class="game-frame-boundary"
        data-frame-boundary
        hidden={!frame.boundaryKind}
      >
        {frame.boundaryKind}
      </div>
      <div
        class="game-frame-detail"
        data-frame-detail
        hidden={!frame.detail}
        title={frame.detail}
        class:outcome-success={frame.tone === "success"}
        class:outcome-failure={frame.tone === "failure"}
        class:outcome-timeout={frame.tone === "timeout"}
      >
        {frame.detail}
      </div>
    </div>
    <div class="game-overlay-tools">
      <div class="game-actions panel-actions">
        <button
          data-drag-handle
          class="icon-button icon-only panel-drag"
          type="button"
          aria-label="Move game panel"
          title="Move game panel"
          ><svg class="icon" aria-hidden="true"
            ><use href="/assets/tabler-icons.svg#ti-grip-vertical"></use></svg
          ></button
        >
        <button
          data-rgb-toggle
          class="icon-button icon-only"
          type="button"
          aria-label={rgbLabel}
          title={rgbLabel}
          aria-pressed={rgbEnabled}
          disabled={!hasControl}
          ><svg class="icon" aria-hidden="true"
            ><use
              href={`/assets/tabler-icons.svg#ti-${rgbEnabled ? "eye" : "eye-off"}`}
            ></use></svg
          ></button
        >
        <button
          data-fullscreen
          class="icon-button icon-only"
          type="button"
          aria-label="Fullscreen game"
          title="Fullscreen game"
          ><svg class="icon" aria-hidden="true"
            ><use href="/assets/tabler-icons.svg#ti-maximize"></use></svg
          ></button
        >
        <button
          data-panel-menu="game"
          class="icon-button icon-only"
          type="button"
          aria-label="Game panel options"
          title="Game panel options"
          ><svg class="icon" aria-hidden="true"
            ><use href="/assets/tabler-icons.svg#ti-dots-vertical"></use></svg
          ></button
        >
      </div>
      <div class="game-render-speed" data-render-speed title={speed.title}>
        <div data-fps-value>{speed.value}</div>
        <svg viewBox="0 0 120 32" preserveAspectRatio="none" aria-hidden="true"
          ><path data-fps-area d={speed.path}></path></svg
        >
      </div>
    </div>
  </div>
</section>
