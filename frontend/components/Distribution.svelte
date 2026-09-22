<script lang="ts">
  import {
    descriptorFor,
    descriptorAvailability,
    descriptorValue,
  } from "../../src/gradlab/web_player/panels/telemetry.js";
  import {
    distributionBlockTitle,
    actionComparisonPresentation,
  } from "../../src/gradlab/web_player/panels/telemetry-panel.js";
  import { formatActionValue } from "../../src/gradlab/web_player/panels/action-contract.js";
  import ActionComparison from "./ActionComparison.svelte";
  let { block, context }: { block: any; context: any } = $props();
  let descriptor = $derived(descriptorFor(block.metric));
  let availability = $derived(
    descriptorAvailability(descriptor, { snapshot: context.snapshot }),
  );
  let decision = $derived(
    descriptorValue(descriptor, { snapshot: context.snapshot }),
  );
  let comparison = $derived(
    actionComparisonPresentation(context.snapshot, context.history, decision),
  );
  let available = $derived(
    ["available", "not-yet-observed"].includes(availability.status),
  );
  let foot = $derived.by(() => {
    const messages = [block.foot];
    const semantics =
      context.snapshot?.session?.action_contract?.policy?.semantics;
    if (semantics?.status === "unavailable")
      messages.push(
        `Action semantics unavailable: ${semantics.reason || "the provider did not declare them"}.`,
      );
    if (available && comparison)
      for (const state of [comparison.history, comparison.step])
        if (state.message) messages.push(state.message);
    return messages.filter(Boolean).join(" ");
  });
</script>

<section
  class="telemetry-block telemetry-distribution"
  hidden={availability.status === "unsupported"}
  data-telemetry-status={!available
    ? availability.status
    : !decision
      ? "not-yet-observed"
      : "available"}
>
  {#if distributionBlockTitle(block, descriptor)}<div class="chart-heading">
      <span>{distributionBlockTitle(block, descriptor)}</span>
    </div>{/if}
  <div class="action-comparison-layout">
    <div class="action-comparison-legend" hidden={!available || !comparison}>
      <div class="action-comparison-legend-series">
        <span class="step">Step action probability</span><span class="episode"
          >Window: policy choices</span
        ><span class="environment">Window: environment actions</span>
      </div>
    </div>
    {#if !available}<div
        class={`action-probabilities empty-state ${availability.status}`}
      >
        {availability.message}
      </div>
    {:else if !decision}<div
        class="action-probabilities empty-state widget-empty"
      >
        No data available yet
      </div>
    {:else if !Array.isArray(decision.probabilities)}<div
        class="distribution-summary"
      >
        Executed {formatActionValue(
          context.snapshot?.transition?.executed_action,
          context.snapshot,
        )} · mean {JSON.stringify(decision.mean)} · std {JSON.stringify(
          decision.stddev,
        )}
      </div>
    {:else if comparison}<div class="action-comparison">
        <ActionComparison rows={comparison.rows} />
      </div>{/if}
  </div>
  <p
    class="panel-foot"
    hidden={!foot}
    class:warning={available &&
      comparison &&
      [comparison.history.status, comparison.step.status].some((status) =>
        ["contract-incomparable", "protocol-error"].includes(status),
      )}
  >
    {foot}
  </p>
</section>
