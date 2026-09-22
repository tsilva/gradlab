<script lang="ts">
  import {
    policyDecisionPresentation,
    ordinal,
  } from "../../src/gradlab/web_player/panels/telemetry-panel.js";
  import Stats from "./Stats.svelte";
  import Distribution from "./Distribution.svelte";
  import ActionComparison from "./ActionComparison.svelte";
  let { definition, context }: { definition: any; context: any } = $props();
  let data = $derived<any>(
    policyDecisionPresentation(context.snapshot, context.history, context.view),
  );
  let foot = $derived(
    [
      definition.config.blocks[0].foot,
      definition.config.blocks[1].foot,
      data.foot,
    ]
      .filter(Boolean)
      .join(" "),
  );
</script>

<section
  class="telemetry-block policy-decision-content"
  data-telemetry-status={data.discrete ? "available" : undefined}
>
  {#if !context.snapshot?.transition}<div class="widget-empty">
      No data available yet
    </div>
  {:else if !data.discrete}<div class="policy-decision-fallback">
      <Stats block={definition.config.blocks[0]} {context} /><Distribution
        block={definition.config.blocks[1]}
        {context}
      />
    </div>
  {:else}<div class="policy-decision-discrete">
      <div
        class="policy-decision-hero"
        class:selected-is-highest={data.selectedIsHighest === true}
        class:selected-below-highest={data.selectedIsHighest === false}
      >
        <span class="policy-decision-context-label">Policy chose</span>
        <div class="policy-decision-hero-line">
          <strong
            class="policy-decision-action"
            class:selected-is-highest={data.selectedIsHighest === true}
            class:selected-below-highest={data.selectedIsHighest === false}
            >{data.action}</strong
          ><strong class="policy-decision-probability"
            >{data.stepProbability === null
              ? "—"
              : `${(data.stepProbability * 100).toFixed(1)}%`}</strong
          >
        </div>
        <div class="policy-decision-mode-line">
          <span class="policy-decision-mode">{data.mode}</span><span
            class="policy-decision-rank"
            class:selected-is-highest={data.selectedIsHighest === true}
            class:selected-below-highest={data.selectedIsHighest === false}
            title={data.rank === null
              ? "The selected action cannot be ranked because its probability is unavailable."
              : `Selected action ranks ${ordinal(data.rank)} of ${data.choiceCount} by this step's action probabilities.`}
            >{data.rank === null
              ? "Rank unavailable"
              : `${ordinal(data.rank)} of ${data.choiceCount} choices`}</span
          >
        </div>
        <div class="policy-decision-context-label">
          Probability of choosing this command
        </div>
        <div
          class="policy-decision-execution"
          class:overridden={Boolean(data.overrideRuleId)}
        >
          <span class="policy-decision-context-label"
            >Command sent to environment</span
          ><strong class="policy-decision-effective-action"
            >{data.effectiveAction}</strong
          ><span
            class="policy-decision-override-reason"
            hidden={!data.overrideRuleId}
            >{data.overrideRuleId
              ? `Override: ${data.overrideRuleId}`
              : ""}</span
          >
        </div>
        <div
          class="policy-decision-context-label"
          hidden={!data.environmentActionNote}
        >
          {data.environmentActionNote || ""}
        </div>
      </div>
      <div
        class="policy-decision-comparison"
        role="table"
        aria-label="Selected-step probabilities and trailing-window policy and environment action frequencies"
      >
        <div class="policy-decision-comparison-header" role="row">
          {#each [["Action", "action"], ["", "bars"], ["STEP", "step"], ["POLICY", "episode"], ["ENV", "environment"]] as [label, className]}<span
              class={className}
              role="columnheader">{label}</span
            >{/each}
        </div>
        <div class="policy-decision-comparison-rows" role="rowgroup">
          <ActionComparison rows={data.rows} compact />
        </div>
      </div>
      <div class="policy-decision-stats">
        {#each data.stats as stat}<div class="policy-decision-stat">
            <span>{stat.label}</span><strong>{stat.value}</strong>
          </div>{/each}
      </div>
      <p class="panel-foot" class:warning={data.warning} hidden={!foot}>
        {foot}
      </p>
    </div>{/if}
</section>
