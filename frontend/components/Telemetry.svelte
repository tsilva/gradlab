<script lang="ts">
  import Panel from "./Panel.svelte";
  import LineChart from "./LineChart.svelte";
  import RewardTable from "./RewardTable.svelte";
  import Stats from "./Stats.svelte";
  import Distribution from "./Distribution.svelte";
  import PolicyDecision from "./PolicyDecision.svelte";
  import RewardBreakdown from "./RewardBreakdown.svelte";
  import Histogram from "./Histogram.svelte";
  import Namespace from "./Namespace.svelte";
  import {
    usesChartHistory,
    chartHistoryBlock,
  } from "../../src/gradlab/web_player/panels/chart-status.js";
  import { policyDecisionLayoutEnabled } from "../../src/gradlab/web_player/panels/telemetry-panel.js";
  let { definition, services } = $props();
  let context = $state.raw<any>({ snapshot: null, history: [], view: {} });
  function update(snapshot: any, history: any, view: any) {
    if (
      snapshot === context.snapshot &&
      history === context.history &&
      Object.keys(view).length === Object.keys(context.view).length &&
      Object.entries(view).every(([key, value]) => context.view[key] === value)
    )
      return;
    context = { snapshot, history, view };
  }
  export function render(snapshot: any, view = context.view) {
    update(snapshot, view.history ?? context.history, view);
  }
  export function renderHistory(
    history: any,
    snapshot = context.snapshot,
    view = context.view,
  ) {
    update(snapshot, history, view);
  }
  let chart = $derived(context.view.chartStatus),
    hasChart = $derived(usesChartHistory({ ...definition, type: "telemetry" }));
  let decision = $derived(policyDecisionLayoutEnabled(definition));
  let empty = $derived(
    chart?.status === "idle" ||
      (chart?.status === "ready" && chart.data?.length === 0),
  );
</script>

<Panel
  {definition}
  className={`${decision ? "policy-decision-panel" : ""} ${definition.id === "step-reward" ? "step-reward-panel" : ""}`}
  chartStatus={hasChart ? chart?.status : undefined}
>
  {#if hasChart}<div
      class="chart-status"
      role="status"
      aria-live="polite"
      class:widget-empty={empty}
      hidden={!chart ||
        (!empty && ["ready", "refreshing"].includes(chart.status))}
    >
      <span
        >{empty
          ? "No data available yet"
          : chart?.status === "error"
            ? chart.error || "Unable to load episode charts"
            : chart?.status === "recovering"
              ? "Recovering chart history…"
              : chart?.status === "loading"
                ? "Loading chart history…"
                : ""}</span
      ><button
        type="button"
        hidden={chart?.status !== "error"}
        onclick={() => services.retryChartHistory?.()}>Retry</button
      >
    </div>{/if}
  <div class="telemetry-blocks">
    {#if decision}<PolicyDecision
        {definition}
        {context}
      />{:else}{#each definition.config.blocks as block, index (index)}
        {@const waiting =
          chart &&
          (chart.data === null ||
            (chart.status === "ready" && chart.data?.length === 0)) &&
          chartHistoryBlock(block)}
        <div style={waiting ? "display:none" : "display:contents"}>
          {#if block.kind === "line"}<LineChart
              {block}
              {definition}
              {context}
              {services}
            />
          {:else if block.kind === "reward-table"}{#key context.view.rewardReference?.episode}<RewardTable
                {context}
                {services}
              />{/key}
          {:else if block.kind === "stats"}<Stats {block} {context} />
          {:else if block.kind === "histogram"}<Histogram {block} {context} />
          {:else if block.kind === "distribution"}<Distribution
              {block}
              {context}
            />
          {:else if block.kind === "reward-breakdown"}<RewardBreakdown
              {block}
              {definition}
              {context}
              {services}
            />
          {:else}<Namespace {block} {definition} {context} {services} />{/if}
        </div>
      {/each}{/if}
  </div>
</Panel>
