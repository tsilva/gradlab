<script lang="ts">
  import { getContext } from "svelte";
  import type { Snippet } from "svelte";
  let {
    definition,
    className = "",
    chartStatus,
    children,
  }: {
    definition: any;
    className?: string;
    chartStatus?: string;
    children: Snippet;
  } = $props();
  const services = getContext<any>("panel-services");
  let capture = $derived(
    definition.id === "cnn"
      ? "CNN features"
      : definition.id === "attribution"
        ? "attribution"
        : null,
  );
  let toggleLabel = $derived(
    capture ? `${capture} capture` : `${definition.label} data processing`,
  );
</script>

<section
  class={`panel ${className}`}
  class:panel-disabled={!definition.enabled}
  data-panel={definition.id}
  data-chart-status={chartStatus}
  aria-labelledby={`${definition.id}-panel-heading`}
>
  <header class="panel-header">
    <button
      data-drag-handle
      class="icon-button icon-only panel-drag"
      type="button"
      aria-label={`Move ${definition.label} panel`}
      title={`Move ${definition.label} panel`}
      ><svg class="icon" aria-hidden="true"
        ><use href="/assets/tabler-icons.svg#ti-grip-vertical"></use></svg
      ></button
    >
    <div class="panel-title">
      <h2 id={`${definition.id}-panel-heading`}>{definition.label}</h2>
    </div>
    {#if definition.switchable}<label
        class="panel-processing-toggle"
        title={`Enable or disable ${toggleLabel}`}
        ><input
          type="checkbox"
          role="switch"
          data-panel-enabled={definition.id}
          checked={definition.enabled}
          aria-label={toggleLabel}
          title={`Enable or disable ${toggleLabel}`}
          onchange={(event) =>
            services.setPanelEnabled(
              definition.id,
              event.currentTarget.checked,
            )}
        /></label
      >{/if}
    <button
      data-panel-menu={definition.id}
      class="icon-button icon-only"
      type="button"
      aria-label={`${definition.label} panel options`}
      title={`${definition.label} panel options`}
      ><svg class="icon" aria-hidden="true"
        ><use href="/assets/tabler-icons.svg#ti-dots-vertical"></use></svg
      ></button
    >
  </header>
  {@render children()}
</section>
