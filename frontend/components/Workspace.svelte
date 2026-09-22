<script lang="ts">
  import { setContext, untrack } from "svelte";
  import PanelSlot from "./PanelSlot.svelte";
  let { host } = $props();
  setContext(
    "panel-services",
    untrack(() => host.options.services),
  );
</script>

{#each host.entries as entry (entry.key)}<PanelSlot {host} {entry} />{/each}
