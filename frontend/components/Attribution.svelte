<script lang="ts">
  import Panel from "./Panel.svelte";
  import { attributionPresentation } from "../../src/gradlab/web_player/panels/diagnostic-overlays.js";
  let { definition, services } = $props();
  let snapshot = $state.raw<any>(null),
    hasControl = $state(false),
    method = $state("gradcam"),
    interval = $state(1);
  let methodInput: HTMLSelectElement, intervalInput: HTMLInputElement;
  let supported: string[] = $derived(
    snapshot?.policy?.attribution?.supported_modes || [],
  );
  let presentation = $derived(
    attributionPresentation(
      snapshot,
      snapshot?.transition?.attribution?.status === "available",
    ),
  );
  export function render(next: any) {
    snapshot = next;
    const state = services.getState();
    hasControl = Boolean(state.hasControl);
    const modes = next?.policy?.attribution?.supported_modes || [],
      shared = next?.session?.attribution || {},
      preference = state.attributionPreference || {};
    const selected = modes.includes(shared.mode)
      ? shared.mode
      : modes.includes(preference.mode)
        ? preference.mode
        : modes[0];
    if (document.activeElement !== methodInput && selected) method = selected;
    if (document.activeElement !== intervalInput)
      interval = modes.includes(shared.mode)
        ? shared.interval || (selected === "occlusion" ? 8 : 1)
        : preference.interval || (selected === "occlusion" ? 8 : 1);
  }
  function send() {
    const config = { mode: method, interval };
    services.setAttributionPreference?.(config);
    services.command("set_attribution", config);
  }
</script>

<Panel {definition} className="attribution-panel">
  <div class="attribution-controls">
    <label
      >Method <select
        data-attribution-method
        bind:this={methodInput}
        bind:value={method}
        disabled={!hasControl || !supported.length}
        onchange={(event) => {
          method = event.currentTarget.value;
          interval = method === "occlusion" ? 8 : 1;
          send();
        }}
        ><option value="gradcam" disabled={!supported.includes("gradcam")}
          >Grad-CAM</option
        ><option value="occlusion" disabled={!supported.includes("occlusion")}
          >Occlusion</option
        ></select
      ></label
    ><label
      >Every N steps <input
        data-attribution-interval
        bind:this={intervalInput}
        type="number"
        min="1"
        step="1"
        bind:value={interval}
        inputmode="numeric"
        disabled={!hasControl || !supported.length}
        onchange={(event) => {
          if (event.currentTarget.validity.valid) {
            interval = Number(event.currentTarget.value);
            send();
          }
        }}
      /></label
    >
  </div>
  <div
    class="attribution-state"
    data-attribution-state
    data-kind={presentation.kind}
  >
    <strong data-attribution-label>{presentation.label}</strong><span
      data-attribution-detail>{presentation.detail}</span
    >
  </div>
  <div class="attribution-explanation">
    <strong>Shared spatial view</strong><span
      >The Input panel overlays the selected method on the exact model input
      while this panel is enabled.</span
    ><span
      >Attribution associates input regions with the selected policy action
      without changing the shared trajectory.</span
    >
  </div>
  <p class="panel-foot">
    Grad-CAM is fast and convolution-specific. Occlusion is slower, but directly
    measures the selected action score after masking input regions.
  </p>
</Panel>
