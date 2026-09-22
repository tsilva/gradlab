<script lang="ts">
  let { rows, compact = false }: { rows: any[]; compact?: boolean } = $props();
  const probability = (value: number | null) =>
    value === null ? "—" : `${(value * 100).toFixed(1)}%`;
  const values = (row: any) =>
    [
      ["step", row.stepProbability],
      ["episode", row.policyFrequency],
      ["environment", row.environmentFrequency],
    ] as [string, number | null][];
</script>

{#snippet track(name: string, series: string, value: number | null)}
  <div
    class={`action-comparison-track ${series}`}
    role="progressbar"
    aria-label={`${name} ${series === "episode" ? "window policy-choice frequency" : series === "environment" ? "window environment-action frequency" : "step action probability"}`}
    aria-valuemin="0"
    aria-valuemax="100"
    aria-valuenow={value === null ? undefined : value * 100}
    aria-valuetext={value === null ? "Unavailable" : undefined}
  >
    <div
      class="action-comparison-fill"
      style:width={`${value === null ? 0 : value * 100}%`}
    ></div>
  </div>
{/snippet}
{#each rows as row (row.name)}
  {@const label = [
    row.selected ? "selected action" : "",
    row.highest ? "highest step probability" : "",
  ].filter(Boolean)}
  <div
    class={compact ? "policy-decision-comparison-row" : "action-comparison-row"}
    class:selected={row.selected}
    class:highest={compact && row.highest}
    class:executed={row.executed}
    role={compact ? "row" : undefined}
  >
    <span
      class={compact
        ? "policy-decision-action-label"
        : "action-comparison-label"}
      title={compact && label.length
        ? `${row.name} — ${label.join("; ")}`
        : row.name}
      aria-label={compact && label.length
        ? `${row.name} — ${label.join("; ")}`
        : row.name}
      role={compact ? "rowheader" : undefined}>{row.name}</span
    >
    <div
      class={compact ? "policy-decision-bars" : "action-comparison-bars"}
      role={compact ? "cell" : undefined}
    >
      {#each values(row) as [series, value]}
        {#if compact}{@render track(row.name, series, value)}{:else}<div
            class={`action-comparison-bar ${series}`}
          >
            {@render track(row.name, series, value)}<span
              class="action-comparison-amount">{probability(value)}</span
            >
          </div>{/if}
      {/each}
    </div>
    {#if compact}{#each values(row) as [series, value]}<span
          class={`policy-decision-amount ${series}`}
          role="cell">{probability(value)}</span
        >{/each}{/if}
  </div>
{/each}
