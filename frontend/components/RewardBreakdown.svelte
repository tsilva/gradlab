<script lang="ts">
  import {
    rewardBreakdownPresentation,
    signedContributionLabel,
    magnitudeShareLabel,
  } from "../../src/gradlab/web_player/panels/reward-breakdown.js";
  import { rewardSummaryCards } from "../../src/gradlab/web_player/panels/telemetry-panel.js";
  let {
    block,
    definition,
    context,
    services,
  }: { block: any; definition: any; context: any; services: any } = $props();
  let scope = $state("episode");
  $effect(() => {
    scope = block.scope === "step" ? "step" : "episode";
  });
  let data = $derived<any>(rewardBreakdownPresentation({ ...context, scope }));
  let maximum = $derived(
    Math.max(0, ...(data.rows || []).map((row: any) => Math.abs(row.impact))),
  );
  const number = (value: any) =>
    value == null || !Number.isFinite(Number(value))
      ? "—"
      : `${Number(value) > 0 ? "+" : ""}${Number(value).toLocaleString(undefined, { minimumFractionDigits: Math.abs(Number(value)) > 0 && Math.abs(Number(value)) < 0.001 ? 4 : 0, maximumFractionDigits: 4 })}`;
  const sign = (value: number) =>
    value > 0 ? "positive" : value < 0 ? "negative" : "zero";
</script>

<section
  class="telemetry-block telemetry-reward-breakdown"
  data-telemetry-status={data.status}
>
  <div class="reward-analysis-toolbar" class:titleless={!block.title}>
    {#if block.title}<span class="chart-heading">{block.title}</span>{/if}<label
      >Scope <select
        aria-label="Reward analysis scope"
        bind:value={scope}
        onchange={(event) => {
          scope = event.currentTarget.value;
          services.updatePanelConfig?.(definition.id, {
            blocks: definition.config.blocks.map((candidate: any) =>
              candidate === block ? { ...candidate, scope } : candidate,
            ),
          });
        }}
        ><option value="step">Selected step</option><option value="episode"
          >Episode to cursor</option
        ></select
      ></label
    >
  </div>
  <div
    class={`reward-analysis-state empty-state ${data.status}`}
    class:widget-empty={data.status === "not-yet-observed"}
    hidden={data.status === "available"}
  >
    {data.status === "not-yet-observed"
      ? "No data available yet"
      : data.message}
  </div>
  {#if data.status === "available"}<div class="reward-analysis-content">
      <div class="table-scroll reward-ledger-scroll">
        <table class="reward-ledger-table">
          <thead
            ><tr
              >{#each ["Component", "Raw", "Final impact", "Signed contribution", "Activity share"] as label}<th
                  scope="col">{label}</th
                >{/each}</tr
            ></thead
          ><tbody
            >{#each data.rows as row}<tr class={`reward-ledger-row ${row.kind}`}
                ><th scope="row"
                  ><div class="reward-ledger-identity">
                    <span
                      class={`reward-sign ${sign(row.impact)}`}
                      aria-label={`${sign(row.impact)} impact`}
                      >{row.impact > 0 ? "+" : row.impact < 0 ? "−" : "0"}</span
                    ><span>{row.label}</span>
                  </div>
                  <div class="reward-zero-bar" aria-hidden="true">
                    <span
                      class={sign(row.impact)}
                      style:--reward-bar-size={`${maximum > 0 ? (50 * Math.abs(row.impact)) / maximum : 0}%`}
                    ></span>
                  </div></th
                ><td>{number(row.raw)}</td><td>{number(row.impact)}</td><td
                  >{signedContributionLabel(row.signedContribution)}</td
                ><td>{magnitudeShareLabel(row.magnitudeShare)}</td></tr
              >{/each}</tbody
          >
        </table>
      </div>
      <div class="reward-ledger-summary">
        {#each rewardSummaryCards(data) as [label, value, kind]}<div
            class={String(kind)}
          >
            <span>{label}</span><strong>{number(value)}</strong>
          </div>{/each}
      </div>
    </div>{/if}
  {#if block.foot}<p
      class="panel-foot"
      class:warning={["protocol-error", "partial-history"].includes(
        data.status,
      )}
    >
      {block.foot}
    </p>{/if}
</section>
