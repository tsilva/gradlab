<script lang="ts">
  import {
    descriptorFor,
    descriptorAvailability,
    descriptorValue,
    formatTelemetryValue,
  } from "../../src/gradlab/web_player/panels/telemetry.js";
  import { selectedPoint } from "../../src/gradlab/web_player/panels/telemetry-panel.js";
  import { formatActionValue } from "../../src/gradlab/web_player/panels/action-contract.js";
  let { block, context }: { block: any; context: any } = $props();
  let point = $derived(
    selectedPoint(context.history, context.snapshot, context.view),
  );
  let rows = $derived(
    block.metrics
      .map((key: string) => {
        const descriptor = descriptorFor(key),
          availability = descriptorAvailability(descriptor, {
            snapshot: context.snapshot,
            point,
          });
        const value = descriptorValue(descriptor, {
          snapshot: context.snapshot,
          point,
        });
        return {
          key,
          label: descriptor?.shortLabel || key,
          status: availability.status,
          value:
            availability.status === "available"
              ? descriptor?.type === "categorical" && value != null
                ? formatActionValue(value, context.snapshot)
                : formatTelemetryValue(value, descriptor)
              : availability.message,
        };
      })
      .filter((row: any) => row.status !== "unsupported"),
  );
</script>

<section class="telemetry-block telemetry-stats">
  {#if block.title}<div class="chart-heading">
      <span>{block.title}</span>
    </div>{/if}
  <div class="stat-grid">
    {#each rows as row (row.key)}<div class="stat">
        <span class="stat-label">{row.label}</span><span class="stat-value"
          >{row.value}</span
        >
      </div>{/each}
  </div>
  <p class="panel-foot" hidden={!block.foot}>{block.foot || ""}</p>
</section>
