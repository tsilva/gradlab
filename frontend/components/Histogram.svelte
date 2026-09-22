<script lang="ts">
  import { getContext } from "svelte";
  const active = getContext<() => boolean>("panel-active") ?? (() => true);
  import { onMount } from "svelte";
  import { descriptorFor } from "../../src/gradlab/web_player/panels/telemetry.js";
  import {
    selectedPoint,
    histogramSelectedLabel,
  } from "../../src/gradlab/web_player/panels/telemetry-panel.js";
  import {
    scalarActionIndex,
    discreteActionLabels,
    formatActionValue,
  } from "../../src/gradlab/web_player/panels/action-contract.js";
  import { drawHistogram } from "../../src/gradlab/web_player/panels/shared.js";
  let { block, context }: { block: any; context: any } = $props();
  let canvas: HTMLCanvasElement,
    size = $state(0);
  let descriptor = $derived(descriptorFor(block.metric));
  let data = $derived.by(() => {
    const values = descriptor?.history
      ? context.history
          .map(descriptor.history)
          .filter((value: any) => value !== null && value !== undefined)
      : [];
    const indices = values.map(scalarActionIndex),
      numeric = indices.every(
        (value: number | null) => value !== null && value >= 0,
      );
    const names: (string | null)[] = numeric
      ? discreteActionLabels(context.snapshot)
      : [...new Set<string>(values.map(String))].sort();
    if (numeric)
      while (names.length <= Math.max(-1, ...indices))
        names.push(formatActionValue(names.length, context.snapshot));
    if (!names.length) names.push("—");
    const counts = Array.from({ length: names.length }, () => 0);
    values.forEach((value: any, i: number) => {
      const index = numeric ? indices[i] : names.indexOf(String(value));
      if (index >= 0) counts[index] = (counts[index] || 0) + 1;
    });
    const point = context.view?.inspection
      ? selectedPoint(context.history, context.snapshot, context.view)
      : null;
    const selected =
      point && descriptor?.history ? descriptor.history(point) : null;
    const highlight =
      selected == null
        ? null
        : numeric
          ? scalarActionIndex(selected)
          : names.indexOf(String(selected));
    const label = histogramSelectedLabel(names, highlight);
    return {
      names,
      counts,
      highlight: label === null ? null : highlight,
      caption: values.length
        ? `${values.length} values in the retained episode${label === null ? "." : ` · selected ${label}.`}`
        : "No data available yet",
      empty: !values.length,
    };
  });
  onMount(() => {
    const observer = new ResizeObserver(() => size++);
    observer.observe(canvas);
    size++;
    return () => observer.disconnect();
  });
  $effect(() => {
    if (!active()) return;
    size;
    if (canvas && !data.empty)
      drawHistogram(canvas, data.counts, data.names, {
        highlightIndex: data.highlight,
      });
  });
</script>

<section class="telemetry-block telemetry-plot">
  <div class="chart-heading">
    <span>{block.title || descriptor?.label || "Histogram"}</span>
  </div>
  <canvas
    class="chart telemetry-chart"
    bind:this={canvas}
    hidden={data.empty}
    aria-label={`${descriptor?.label || "Metric"} histogram`}
  ></canvas>
  <p class="panel-foot" class:widget-empty={data.empty}>{data.caption}</p>
  {#if block.foot}<p class="panel-foot">{block.foot}</p>{/if}
</section>
