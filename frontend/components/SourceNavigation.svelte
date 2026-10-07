<script lang="ts">
  import { untrack } from "svelte";
  import SourceIcon from "./SourceIcon.svelte";
  let { controller, kind, presentation = null }: { controller: any; kind: string; presentation?: any } = $props();
  let value = $state.raw<any>(untrack(() => presentation));
  export function render(next: any) {
    value = next;
  }
</script>

{#if kind === "breadcrumbs"}
  {#each value || [] as item}
    <button
      type="button"
      class="quiet"
      title={item.title}
      disabled={item.current}
      onclick={() => item.route && controller.navigate(item.route)}
      >{item.label}</button
    >
  {/each}
{:else if value}
  <button
    type="button"
    class="quiet checkpoint-navigation-button icon-only"
    aria-label="Previous checkpoint"
    data-checkpoint-previous
    disabled={value.previousDisabled}
    title={value.previousTitle}
    onclick={() => controller.selectAdjacentCheckpoint("previous")}
  >
    <SourceIcon name="arrow-left" />
  </button>
  <span
    class="checkpoint-navigation-position"
    data-checkpoint-position
    aria-label="Checkpoint position"
    aria-live="polite"
    title={value.title}>{value.position}</span
  >
  <button
    type="button"
    class="quiet checkpoint-navigation-button icon-only"
    aria-label="Next checkpoint"
    data-checkpoint-next
    disabled={value.nextDisabled}
    title={value.nextTitle}
    onclick={() => controller.selectAdjacentCheckpoint("next")}
  >
    <SourceIcon name="arrow-right" />
  </button>
{/if}
