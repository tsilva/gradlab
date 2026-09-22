<script lang="ts">
  import { getContext } from "svelte";
  const active = getContext<() => boolean>("panel-active") ?? (() => true);
  import { onMount } from "svelte";
  import { chartPoints } from "../../src/gradlab/web_player/panels/chart-status.js";
  import {
    descriptorFor,
    seriesForMetric,
  } from "../../src/gradlab/web_player/panels/telemetry.js";
  import { themeColor } from "../../src/gradlab/web_player/panels/shared.js";
  import { selectedRange } from "../../src/gradlab/web_player/chart-range.js";
  import { rewardContribution } from "../../src/gradlab/web_player/panels/reward-discount.js";
  import { REWARD_LABELS } from "../../src/gradlab/web_player/panels/reward-inspector.js";
  import {
    lineLegendPresentation,
    lineLegendPresentationAtIndex,
    lineBlockAvailability,
    lineBlockFootPresentation,
    lineCursorSequence,
  } from "../../src/gradlab/web_player/panels/telemetry-panel.js";
  import { createLineSurface } from "../line-surface.js";
  let {
    block,
    definition,
    context,
    services,
    showLegend = true,
    chartLabel = "",
    seekable = true,
  }: {
    block: any;
    definition: any;
    context: any;
    services: any;
    showLegend?: boolean;
    chartLabel?: string;
    seekable?: boolean;
  } = $props();
  let canvas = $state<HTMLCanvasElement>(null!);
  let tooltipWidth = $state(0),
    tooltipHeight = $state(0);
  let paint: ReturnType<typeof createLineSurface> | undefined;
  let geometry = $state.raw<any>(null),
    sizeRevision = $state(0),
    pointerY = $state<number | null>(null);
  let drag = $state.raw<any>(null),
    dragEnd = $state(0);
  let suppressClick = false,
    clickTimer: ReturnType<typeof setTimeout> | undefined;
  let descriptors = $derived(block.metrics.map(descriptorFor).filter(Boolean));
  let points = $derived(chartPoints(context.history, context.view));
  let steps = $derived(points.map((point: any) => point.step));
  let reward = $derived(
    definition.id === "step-reward" &&
      block.metrics.length === 2 &&
      block.metrics.includes("reward/provider") &&
      block.metrics.includes("reward/shaped"),
  );
  let reference = $derived(
    reward ? (context.view?.rewardReference?.step ?? null) : null,
  );
  let gamma = $derived(
    context.snapshot?.session?.value_discount ??
      context.snapshot?.session?.critic_comparison?.discount,
  );
  let series = $derived.by(() => {
    const result = descriptors.map((descriptor: any) => ({
      label: reward
        ? REWARD_LABELS[descriptor.key as keyof typeof REWARD_LABELS]
        : descriptor.shortLabel,
      values: seriesForMetric(descriptor.key, points),
      color: themeColor(descriptor.color || "chartBar"),
    }));
    if (
      reward &&
      typeof gamma === "number" &&
      Number.isFinite(gamma) &&
      gamma >= 0 &&
      gamma <= 1
    )
      result.push({
        label: "Discounted contribution",
        values: points.map(
          (point: any) =>
            rewardContribution(point, reference, gamma)?.contribution ?? NaN,
        ),
        color: themeColor("seriesAmber"),
        dash: [4, 3],
      });
    return result;
  });
  let hover = $derived(context.view?.chartHoverStep);
  let cursor = $derived(
    Number.isFinite(hover) ? hover : context.snapshot?.transition?.step,
  );
  let hoverIndex = $derived(
    Number.isFinite(hover) && points.length
      ? points.reduce(
          (best: number, point: any, index: number) =>
            Math.abs(point.step - hover) < Math.abs(points[best].step - hover)
              ? index
              : best,
          0,
        )
      : null,
  );
  let legend = $derived(
    hoverIndex !== null
      ? lineLegendPresentationAtIndex(descriptors, points, hoverIndex)
      : lineLegendPresentation(descriptors, context.history, context.view),
  );
  let availability = $derived(
    lineBlockAvailability(descriptors, context.snapshot, context.history),
  );
  let foot = $derived(
    lineBlockFootPresentation(
      block,
      availability.unavailable,
      availability.notice,
    ),
  );
  onMount(() => {
    paint = createLineSurface(canvas);
    const observer = new ResizeObserver(() => sizeRevision++);
    observer.observe(canvas);
    sizeRevision++;
    return () => {
      observer.disconnect();
      clearTimeout(clickTimer);
    };
  });
  $effect(() => {
    if (!active()) return;
    sizeRevision;
    const current = series,
      x = steps,
      step = cursor,
      ref = reference;
    if (paint)
      geometry = paint(current, x, {
        cursorStep: step,
        referenceStep: ref,
        dimBeforeStep: ref,
      });
  });
  const xAt = (event: PointerEvent | MouseEvent) =>
    ((event.clientX - canvas.getBoundingClientRect().left) *
      canvas.clientWidth) /
    canvas.getBoundingClientRect().width;
  function down(event: PointerEvent) {
    const plot = geometry?.plot,
      x = xAt(event);
    if (event.button !== 0 || !plot || x < plot.left || x > plot.right) return;
    drag = {
      x,
      id: event.pointerId,
      plot,
      first: points[0]?.step,
      last: points.at(-1)?.step,
    };
    dragEnd = x;
    canvas.setPointerCapture(event.pointerId);
  }
  function move(event: PointerEvent) {
    const bounds = canvas.getBoundingClientRect(),
      plot = geometry?.plot;
    pointerY =
      ((event.clientY - bounds.top) * canvas.clientHeight) / bounds.height;
    if (drag) dragEnd = xAt(event);
    if (plot && points.length && bounds.width > 0)
      services.setChartHoverStep?.(
        points[0].step +
          Math.max(
            0,
            Math.min(1, (xAt(event) - plot.left) / (plot.right - plot.left)),
          ) *
            (points.at(-1).step - points[0].step),
      );
  }
  function up(event: PointerEvent) {
    if (!drag) return;
    const range = selectedRange(
      drag.plot,
      drag.x,
      xAt(event),
      drag.first,
      drag.last,
    );
    suppressClick = Math.abs(drag.x - xAt(event)) >= 5;
    drag = null;
    canvas.releasePointerCapture(event.pointerId);
    if (range) services.setChartRange?.(range);
  }
  function leave() {
    pointerY = null;
    services.setChartHoverStep?.(null);
  }
  function click(event: MouseEvent) {
    if (suppressClick) {
      suppressClick = false;
      return;
    }
    if (context.view?.chartRange) {
      services.setChartRange?.(null);
      return;
    }
    if (!seekable) return;
    const sequence = lineCursorSequence(
      points,
      geometry?.plot,
      xAt(event),
      geometry?.pointCount,
    );
    if (sequence !== null) {
      const point = points.find((point: any) => point.sequence === sequence);
      clearTimeout(clickTimer);
      clickTimer = setTimeout(
        () =>
          point && services.inspectStep
            ? services.inspectStep(point.step)
            : services.inspectSequence?.(sequence),
        250,
      );
    }
  }
</script>

<section
  class="telemetry-block telemetry-plot"
  class:reward-history={reward}
  data-telemetry-status={availability.status}
  style="position:relative"
>
  {#if reward}<div class="reward-reference-controls">
      <button
        type="button"
        disabled={!Number.isInteger(context.snapshot?.transition?.step)}
        onclick={() =>
          services.setRewardReference(context.snapshot.transition.step)}
        >Set return reference to cursor</button
      >
    </div>{/if}
  <!-- svelte-ignore a11y_no_noninteractive_element_interactions a11y_click_events_have_key_events (the playbar supplies keyboard navigation for the same chart cursor) -->
  <canvas
    class="chart telemetry-chart"
    bind:this={canvas}
    aria-label={chartLabel ||
      `${descriptors.map((item: any) => item.label).join(" and ")} history`}
    style="touch-action:none"
    onpointerdown={down}
    onpointermove={move}
    onpointerup={up}
    onpointerleave={leave}
    onpointercancel={() => {
      drag = null;
      leave();
    }}
    onclick={click}
    ondblclick={() => {
      clearTimeout(clickTimer);
      services.setChartRange?.(null);
    }}
  ></canvas>
  <div class="legend" class:reward-history-legend={reward} hidden={!showLegend}>
    {#if reward}
      {#each [["Native reward", "seriesViolet"], ["Shaped reward", "seriesTeal"], ["Discounted contribution", "seriesAmber"]] as [label, color], index}<span
          class:discounted-series={index === 2}
          style:--legend-color={themeColor(color)}>{label}</span
        >{/each}
    {:else}{#each descriptors as descriptor, index (descriptor.key)}<span
          style:--legend-color={themeColor(descriptor.color || "chartBar")}
          >{descriptor.shortLabel} =
          <strong class="legend-value">{legend[index]?.value ?? "—"}</strong
          ></span
        >{/each}{/if}
  </div>
  {#if hoverIndex !== null && geometry?.plot}
    <div
      class="chart-tooltip"
      role="tooltip"
      bind:offsetWidth={tooltipWidth}
      bind:offsetHeight={tooltipHeight}
      style:left={`${Math.max(0, Math.min(canvas?.clientWidth - tooltipWidth, geometry.plot.left + ((hover - steps[0]) / Math.max(1, steps.at(-1) - steps[0])) * (geometry.plot.right - geometry.plot.left) + 12))}px`}
      style:top={`${Math.max(0, Math.min(canvas?.clientHeight - tooltipHeight, (pointerY ?? geometry.plot.top) + 12))}px`}
    >
      <strong>Step {steps[hoverIndex]}</strong>
      {#each series as item (item.label)}<div class="chart-tooltip-row">
          <span class="chart-tooltip-swatch" style:background-color={item.color}
          ></span><span>{item.label}</span><span class="chart-tooltip-value"
            >{Number.isFinite(item.values[hoverIndex])
              ? Number(item.values[hoverIndex].toPrecision(6)).toString()
              : "Unavailable"}</span
          >
        </div>{/each}
    </div>
  {/if}
  {#if drag}<div
      class="chart-drag-selection"
      style:left={`${canvas.offsetLeft + Math.max(drag.plot.left, Math.min(drag.x, dragEnd))}px`}
      style:top={`${canvas.offsetTop + drag.plot.top}px`}
      style:width={`${Math.max(0, Math.min(drag.plot.right, Math.max(drag.x, dragEnd)) - Math.max(drag.plot.left, Math.min(drag.x, dragEnd)))}px`}
      style:height={`${drag.plot.bottom - drag.plot.top}px`}
    ></div>{/if}
  <p class="panel-foot" class:warning={foot.warning} hidden={!foot.text}>
    {foot.text}
  </p>
</section>
