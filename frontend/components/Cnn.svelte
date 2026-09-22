<script lang="ts">
  import { onDestroy } from "svelte";
  import Panel from "./Panel.svelte";
  import CnnTile from "./CnnTile.svelte";
  import {
    cnnFrameIdentity,
    cnnPresentation,
    sameFrameIdentity,
  } from "../../src/gradlab/web_player/panels/diagnostic-overlays.js";
  import { peakRegionLabel } from "../../src/gradlab/web_player/panels/cnn.js";
  let { definition, services } = $props();
  let snapshot = $state.raw<any>(null),
    bitmap = $state.raw<ImageBitmap | null>(null),
    identity = $state.raw<any>(null);
  let layer = $state(""),
    interval = $state(1),
    topK = $state(12),
    hasControl = $state(false);
  let layerInput: HTMLSelectElement,
    intervalInput: HTMLInputElement,
    topInput: HTMLSelectElement;
  let request = 0,
    disposed = false;
  let layers: any[] = $derived(snapshot?.policy?.cnn?.layers || []);
  let inspection = $derived(snapshot?.transition?.cnn?.inspection);
  let exact = $derived(sameFrameIdentity(identity, cnnFrameIdentity(snapshot)));
  let presentation = $derived(cnnPresentation(snapshot, exact));
  const response = (value: any) => {
    const n = Number(value);
    return !Number.isFinite(n)
      ? "—"
      : n === 0
        ? "0"
        : Math.abs(n) >= 1000 || Math.abs(n) < 0.001
          ? n.toExponential(2)
          : n.toFixed(3);
  };
  function clear() {
    bitmap?.close();
    bitmap = null;
    identity = null;
  }
  export function render(next: any) {
    snapshot = next;
    hasControl = Boolean(services.getState().hasControl);
    const shared = next?.session?.cnn || {};
    if (document.activeElement !== layerInput)
      layer = shared.layer_id || next?.policy?.cnn?.layers?.[0]?.id || "";
    if (document.activeElement !== intervalInput)
      interval = shared.interval || 1;
    if (document.activeElement !== topInput) topK = shared.top_k || 12;
    if (!sameFrameIdentity(identity, cnnFrameIdentity(next))) clear();
  }
  export async function renderFrame(
    kind: number,
    blob: Blob | null,
    metadata: any = {},
  ) {
    if (kind !== 4) return false;
    const incoming = {
      sequence: Number(metadata.sequence),
      generation: Number(metadata.generation),
    };
    // A late frame must never clear a newer exact atlas.
    if (!sameFrameIdentity(incoming, cnnFrameIdentity(snapshot))) return true;
    if (blob && sameFrameIdentity(incoming, identity)) return true;
    const current = ++request;
    if (!blob) {
      clear();
      return true;
    }
    const next = await createImageBitmap(blob);
    if (
      disposed ||
      current !== request ||
      metadata.isCurrent?.() === false ||
      !sameFrameIdentity(incoming, cnnFrameIdentity(snapshot))
    ) {
      next.close();
      return true;
    }
    clear();
    bitmap = next;
    identity = incoming;
    return true;
  }
  export function resetFrames() {
    request++;
    clear();
  }
  onDestroy(() => {
    disposed = true;
    resetFrames();
  });
  function send() {
    services.command("set_cnn_inspection", {
      enabled: true,
      layer_id: layer,
      interval,
      top_k: Number(topK),
    });
  }
</script>

<Panel {definition} className="cnn-panel">
  <div class="cnn-controls">
    <label
      >Layer <select
        data-cnn-layer
        bind:this={layerInput}
        bind:value={layer}
        disabled={!hasControl || !layers.length}
        onchange={(event) => {
          layer = event.currentTarget.value;
          send();
        }}
        >{#each layers as item (item.id)}<option value={item.id}
            >{item.label}</option
          >{/each}</select
      ></label
    ><label
      >Every N steps <input
        data-cnn-interval
        bind:this={intervalInput}
        type="number"
        min="1"
        step="1"
        bind:value={interval}
        inputmode="numeric"
        disabled={!hasControl || !layers.length}
        onchange={(event) => {
          if (event.currentTarget.validity.valid) {
            interval = Number(event.currentTarget.value);
            send();
          }
        }}
      /></label
    ><label
      >Top filters <select
        data-cnn-top-k
        bind:this={topInput}
        bind:value={topK}
        disabled={!hasControl || !layers.length}
        onchange={(event) => {
          topK = Number(event.currentTarget.value);
          send();
        }}
        >{#each [4, 8, 12, 16, 24, 32] as count}<option value={count}
            >{count}</option
          >{/each}</select
      ></label
    >
  </div>
  <div class="cnn-state" data-cnn-state data-kind={presentation.kind}>
    <strong data-cnn-label>{presentation.label}</strong><span data-cnn-detail
      >{presentation.detail}</span
    >
  </div>
  <div class="cnn-explanation">
    <strong>Shared spatial view</strong><span
      >The Input panel displays the selected layer's winner map over the exact
      model input.</span
    ><span
      >Peak regions below use the layer's exact stride and receptive field in
      input pixels.</span
    >
  </div>
  <div class="cnn-filter-grid" data-cnn-filters>
    {#each inspection?.filters || [] as item (item.filter_index)}<article
        class="cnn-filter-card"
        style:--filter-color={item.color}
      >
        <header>
          <span class="cnn-filter-swatch"></span><strong
            >Filter {item.filter_index}</strong
          ><small>rank {item.rank}</small>
        </header>
        <div class="cnn-filter-visuals">
          <figure>
            <CnnTile
              bitmap={exact ? bitmap : null}
              atlas={inspection.atlas}
              tile={item.kernel_tile}
              smooth={false}
            />
            <figcaption>Kernel weights</figcaption>
          </figure>
          <figure>
            <CnnTile
              bitmap={exact ? bitmap : null}
              atlas={inspection.atlas}
              tile={item.activation_tile}
              smooth={true}
            />
            <figcaption>Activation</figcaption>
          </figure>
        </div>
        <dl>
          <div>
            <dt>Peak</dt>
            <dd>{response(item.peak_response)}</dd>
          </div>
          <div>
            <dt>Mean +</dt>
            <dd>{response(item.mean_positive_response)}</dd>
          </div>
          <div>
            <dt>Coverage</dt>
            <dd>
              {Number.isFinite(Number(item.positive_coverage))
                ? `${Math.round(Number(item.positive_coverage) * 100)}%`
                : "—"}
            </dd>
          </div>
        </dl>
        <p class="cnn-region">
          Strongest at {peakRegionLabel(item.peak_input_region)}
        </p>
      </article>{/each}
  </div>
  <p class="panel-foot">
    Kernel tiles show every learned input-channel plane within the filter's
    convolution group: amber is positive, cyan is negative. Activation tiles are
    normalized within each filter; ranking and winner colors use unnormalized
    responses. These views explain the representation, not why the policy
    selected its action.
  </p>
</Panel>
