<script lang="ts">
  import {
    descriptorCatalog,
    descriptorFor,
    descriptorValue,
    formatTelemetryValue,
  } from "../../src/gradlab/web_player/panels/telemetry.js";
  import { selectedPoint } from "../../src/gradlab/web_player/panels/telemetry-panel.js";
  import { formatActionValue } from "../../src/gradlab/web_player/panels/action-contract.js";
  import LineChart from "./LineChart.svelte";
  let {
    block,
    definition,
    context,
    services,
  }: { block: any; definition: any; context: any; services: any } = $props();
  let selected = $state("");
  let descriptors = $derived(
    [...descriptorCatalog(context.snapshot, context.history).values()]
      .filter((descriptor) => descriptor.namespace === block.namespace)
      .sort((a, b) => a.label.localeCompare(b.label)),
  );
  let metric = $derived(
    descriptors.some(
      (descriptor) => descriptor.key === (selected || block.metric),
    )
      ? selected || block.metric
      : descriptors[0]?.key || "",
  );
  let label = $derived(
    block.namespace === "signal" ? "Chart signal" : "Chart reward component",
  );
  let lineBlock = $derived({ metrics: metric ? [metric] : [] });
  let point = $derived(
    selectedPoint(context.history, context.snapshot, context.view),
  );
</script>

<section class="telemetry-block telemetry-namespace">
  <div class="signal-toolbar" hidden={!descriptors.length}>
    <label
      ><span class="signal-toolbar-label">{label}</span><select
        aria-label={label}
        value={metric}
        onchange={(event) => {
          selected = event.currentTarget.value;
          services.updatePanelConfig?.(definition.id, {
            blocks: definition.config.blocks.map((candidate: any) =>
              candidate === block
                ? { ...candidate, metric: selected }
                : candidate,
            ),
          });
        }}
        >{#each descriptors as descriptor (descriptor.key)}<option
            value={descriptor.key}>{descriptor.label}</option
          >{/each}</select
      ></label
    >
  </div>
  {#if descriptors.length}<LineChart
      block={lineBlock}
      {definition}
      {context}
      {services}
      showLegend={false}
      chartLabel={`${label} history`}
      seekable={false}
    />
    <div class="telemetry-namespace-table">
      <table>
        <tbody
          >{#each descriptors as descriptor (descriptor.key)}{@const value =
              descriptorValue(descriptor, {
                snapshot: context.snapshot,
                point,
              })}<tr
              ><td>{descriptor.label}</td><td
                >{descriptor.type === "categorical" && value != null
                  ? formatActionValue(value, context.snapshot)
                  : formatTelemetryValue(value, descriptor)}</td
              ></tr
            >{/each}</tbody
        >
      </table>
    </div>{:else}<div class="widget-empty">No data available yet</div>{/if}
  <p class="panel-foot">
    {block.foot ||
      (block.namespace === "signal"
        ? "Post-action environment signals."
        : "Post-action reward components.")}
  </p>
</section>
