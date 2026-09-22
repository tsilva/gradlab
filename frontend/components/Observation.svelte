<script lang="ts">
  import { onMount } from "svelte";
  import Panel from "./Panel.svelte";
  import { createObservationSurface } from "../observation-surface.js";
  let { definition } = $props();
  let element: HTMLDivElement;
  let view = $state.raw<any>({
    exact: false,
    overlay: "none",
    status: {},
    explanation: "",
    legend: [],
    lines: [],
  });
  let surface: ReturnType<typeof createObservationSurface>;
  onMount(() => {
    surface = createObservationSurface(element, (value: any) => (view = value));
    return () => surface.destroy();
  });
  export const render = (snapshot: any) => surface?.render(snapshot);
  export const prepareFrame = (
    ...args: Parameters<typeof surface.prepareFrame>
  ) => surface?.prepareFrame(...args);
  export const renderFrame = (
    ...args: Parameters<typeof surface.renderFrame>
  ) => surface?.renderFrame(...args);
  export const resetFrames = () => surface?.resetFrames();
</script>

<Panel {definition} className="observation-panel">
  <div bind:this={element} style="display:contents">
    <div
      class="diagnostic-state"
      data-diagnostic-state
      data-kind={view.status.kind}
      hidden={view.overlay === "none"}
    >
      <strong data-diagnostic-label>{view.status.label || ""}</strong><span
        data-diagnostic-detail>{view.status.detail || ""}</span
      >
    </div>
    <div class="observation-stage" hidden={!view.exact}>
      <canvas data-observation-canvas></canvas><canvas
        data-diagnostic-canvas
        class="diagnostic-overlay-canvas"
      ></canvas>
    </div>
    <div
      class="diagnostic-context"
      data-diagnostic-context
      hidden={view.overlay === "none"}
    >
      <span data-diagnostic-explanation>{view.explanation}</span>
      <div
        class="cnn-winner-legend"
        data-cnn-winner-legend
        aria-label="CNN winner-map filters"
      >
        {#each view.legend as item (item.index)}<span
            class="cnn-winner-legend-item"
            ><span
              class="cnn-filter-swatch"
              style:--filter-color={item.color}
              aria-hidden="true"
            ></span><span>Filter {item.index}</span></span
          >{/each}
      </div>
    </div>
    <pre
      data-input
      class="compact-pre"
      class:widget-empty={!view.lines.length}>{view.lines.join("\n") ||
        "No data available yet"}</pre>
  </div>
</Panel>
