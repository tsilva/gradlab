<script lang="ts">
  import { onMount, setContext } from "svelte";
  import { panelComponents } from "../loaders";
  let { host, entry } = $props();
  setContext(
    "panel-active",
    () => entry.definition.enabled && !host.options.isSuspended?.(entry.id),
  );
  let gridItem: HTMLDivElement, instance: Record<string, any>;
  let Component = $derived(panelComponents[entry.definition.type]);
  onMount(() => {
    const element = gridItem.querySelector<HTMLElement>(".panel")!;
    element.classList.add("grid-stack-item-content");
    const observer = new ResizeObserver(() => host.invoke(entry.id, "resize"));
    observer.observe(element);
    host.register(entry.id, { component: instance, element, gridItem });
    return () => {
      observer.disconnect();
      host.unregister(entry.id, gridItem);
    };
  });
</script>

<div class="grid-stack-item" data-panel={entry.id} bind:this={gridItem}>
  <Component
    definition={entry.definition}
    services={host.options.services}
    bind:this={instance}
  />
</div>
