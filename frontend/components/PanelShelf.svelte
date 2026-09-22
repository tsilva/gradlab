<script lang="ts">
  let { services } = $props();
  let entries = $state.raw<[string, any][]>([]);
  export function renderShelf() {
    const workspace = services.getWorkspace(),
      windowId = services.getWindowId();
    entries = Object.entries<any>(workspace.panels)
      .filter(
        ([, panel]) =>
          !panel.placement.visible || panel.placement.window !== windowId,
      )
      .sort((a, b) => a[1].title.localeCompare(b[1].title));
  }
</script>

{#each entries as [id, panel] (id)}{@const label = panel.placement.visible
    ? `Move ${panel.title} to this window`
    : `Show ${panel.title}`}<button
    type="button"
    class="shelf-item"
    aria-label={label}
    title={label}
    onclick={() => services.onReveal(id)}
    ><span>{panel.title}</span>{#if panel.placement.visible}<small
        >Other window</small
      >{/if}</button
  >{:else}<span class="empty-state">Every panel is visible in this window.</span
  >{/each}
