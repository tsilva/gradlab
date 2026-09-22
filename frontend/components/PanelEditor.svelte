<script lang="ts">
  import { tick } from "svelte";
  import {
    defaultBlockForKind,
    editorFieldsForBlock,
  } from "../../src/gradlab/web_player/panels/manager.js";
  import {
    descriptorCatalog,
    compatibleMetricKeys,
    metricOptions,
  } from "../../src/gradlab/web_player/panels/telemetry.js";
  let { services } = $props();
  let dialog: HTMLDialogElement, titleInput: HTMLInputElement;
  let editingId = $state<string | null>(null),
    title = $state("Telemetry"),
    heading = $state("Add telemetry panel"),
    blocks = $state<any[]>([]);
  let context = $state.raw<any>({ snapshot: null, history: [] });
  let catalog = $derived(descriptorCatalog(context.snapshot, context.history));
  const labels: Record<string, string> = {
    stats: "Stats",
    line: "Line chart",
    histogram: "Histogram",
    distribution: "Distribution",
    "namespace-explorer": "Metric explorer",
    "reward-breakdown": "Reward breakdown",
    "reward-table": "Reward table",
  };
  export async function openEditor(id: string | null = null) {
    const panel = id ? services.getWorkspace().panels[id] : null;
    if (panel && panel.type !== "telemetry") {
      services.showToast(
        "Only telemetry panels have editable visualizations.",
        true,
      );
      return;
    }
    editingId = id;
    heading = panel ? `Edit ${panel.title}` : "Add telemetry panel";
    title = panel?.title || "Telemetry";
    blocks = structuredClone(
      panel?.config?.blocks || [{ kind: "line", metrics: ["reward/shaped"] }],
    );
    context = services.getContext();
    await tick();
    dialog.showModal();
    titleInput.focus();
    titleInput.select();
  }
  function options(block: any) {
    return block.kind === "namespace-explorer"
      ? [...catalog.values()].filter((descriptor) => descriptor.namespace)
      : metricOptions(
          catalog,
          block.kind,
          context.snapshot,
          ["stats", "line"].includes(block.kind)
            ? block.metrics || []
            : [block.metric || ""],
        );
  }
  function save(event: SubmitEvent) {
    event.preventDefault();
    const value = title.trim();
    if (!value) {
      services.showToast("Give the panel a title.", true);
      titleInput.focus();
      return;
    }
    if (!blocks.length) {
      services.showToast("Add at least one visualization.", true);
      return;
    }
    const invalid = blocks.find((block) =>
      block.kind === "reward-table"
        ? false
        : block.kind === "stats"
          ? !block.metrics?.length
          : block.kind === "line"
            ? !block.metrics?.length ||
              !compatibleMetricKeys(block.metrics, catalog)
            : block.kind === "namespace-explorer"
              ? !block.namespace
              : block.kind === "reward-breakdown"
                ? !["step", "episode"].includes(block.scope)
                : !block.metric,
    );
    if (invalid) {
      services.showToast(
        invalid.kind === "line"
          ? "A line chart can only combine scalar metrics with the same unit."
          : "Choose metrics for every visualization.",
        true,
      );
      return;
    }
    const panel = {
      title: value,
      config: { blocks: JSON.parse(JSON.stringify(blocks)) },
    };
    if (editingId) services.onUpdate(editingId, panel);
    else services.onCreate(panel);
    dialog.close();
  }
</script>

<dialog
  id="panel-editor"
  class="panel-editor"
  aria-labelledby="panel-editor-heading"
  bind:this={dialog}
>
  <form id="panel-editor-form" method="dialog" onsubmit={save}>
    <header>
      <div>
        <span class="eyebrow">PANEL BUILDER</span>
        <h2 id="panel-editor-heading">{heading}</h2>
      </div>
    </header>
    <label class="panel-editor-title"
      ><span>Panel title</span><input
        id="panel-editor-title"
        type="text"
        maxlength="80"
        required
        bind:this={titleInput}
        bind:value={title}
      /></label
    >
    <div id="panel-editor-blocks" class="panel-editor-blocks">
      {#each blocks as block, index (index)}{@const fields =
          editorFieldsForBlock(block)}{@const multiple = [
          "stats",
          "line",
        ].includes(block.kind)}
        <fieldset class="panel-editor-block">
          <legend>Visualization {index + 1}</legend><label
            >Type <select
              value={block.kind}
              onchange={(event) =>
                (blocks[index] = defaultBlockForKind(
                  event.currentTarget.value,
                ))}
              >{#each Object.entries(labels) as [value, label]}<option {value}
                  >{label}</option
                >{/each}</select
            ></label
          ><label
            >Block title (optional) <input
              type="text"
              maxlength="80"
              value={block.title || ""}
              oninput={(event) => {
                if (event.currentTarget.value)
                  block.title = event.currentTarget.value;
                else delete block.title;
              }}
            /></label
          >
          {#if fields.namespace}<label
              >Namespace <select
                value={block.namespace || "signal"}
                onchange={(event) => {
                  block.namespace = event.currentTarget.value;
                  block.metric = "";
                }}
                ><option value="signal">Environment signals</option><option
                  value="reward-component">Reward components</option
                ></select
              ></label
            >{/if}
          {#if fields.scope}<label
              >Scope <select
                value={block.scope || "episode"}
                onchange={(event) => (block.scope = event.currentTarget.value)}
                ><option value="step">Selected step</option><option
                  value="episode">Episode to cursor</option
                ></select
              ></label
            >{/if}
          {#if fields.metric}<label
              >{multiple ? "Metrics" : "Metric"}
              <select
                {multiple}
                size={multiple ? 6 : 1}
                onchange={(event) => {
                  if (multiple)
                    block.metrics = [
                      ...event.currentTarget.selectedOptions,
                    ].map((item) => item.value);
                  else block.metric = event.currentTarget.value;
                }}
              >
                {#if !multiple}<option value="" selected={!block.metric}
                    >Choose a metric</option
                  >{/if}
                {#each options(block) as descriptor (descriptor.key)}<option
                    value={descriptor.key}
                    selected={multiple
                      ? block.metrics?.includes(descriptor.key)
                      : block.metric === descriptor.key}
                    hidden={block.kind === "namespace-explorer" &&
                      Boolean(descriptor.namespace) &&
                      descriptor.namespace !== block.namespace}
                    >{descriptor.label}</option
                  >{/each}
              </select></label
            >{/if}<button
            type="button"
            class="quiet danger"
            onclick={() => blocks.splice(index, 1)}>Remove visualization</button
          >
        </fieldset>
      {/each}
    </div>
    <button
      id="panel-editor-add-block"
      class="quiet button-with-icon"
      type="button"
      onclick={() => blocks.push({ kind: "line", metrics: ["reward/shaped"] })}
      ><svg class="icon" aria-hidden="true"
        ><use href="/assets/tabler-icons.svg#ti-plus"></use></svg
      ><span>Add visualization</span></button
    >
    <footer class="panel-editor-actions">
      <button
        id="panel-editor-cancel"
        class="quiet"
        type="button"
        onclick={() => dialog.close()}>Cancel</button
      ><button class="primary" type="submit">Save panel</button>
    </footer>
  </form>
</dialog>
