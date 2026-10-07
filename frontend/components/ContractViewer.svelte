<script lang="ts">
  import { tick, untrack } from "svelte";
  import SourceIcon from "./SourceIcon.svelte";
  import { decoratedText } from "../../src/gradlab/web_player/documents/viewer.js";
  let {
    controller: c,
    presentation = null,
  }: { controller: any; presentation?: any } = $props();
  let v = $state.raw<any>(untrack(() => presentation));
  let baseScroll = $state<HTMLElement>();
  let resolvedScroll = $state<HTMLElement>();
  let content = $state<HTMLElement>();
  let syncing = false;
  let lastQuery = "";
  export function render(next: any) {
    v = next;
    if (v.query && v.query !== lastQuery) {
      void tick().then(() =>
        content
          ?.querySelector("mark")
          ?.scrollIntoView({ block: "center", inline: "center" }),
      );
    }
    lastQuery = v.query;
  }
  function syncScroll(source?: HTMLElement, target?: HTMLElement) {
    if (!source || !target) return;
    if (syncing) return;
    syncing = true;
    target.scrollTop = source.scrollTop;
    requestAnimationFrame(() => {
      syncing = false;
    });
  }
</script>

{#snippet syntax(part: any)}
  {#if part.className}<span class={part.className}>{part.text}</span
    >{:else}{part.text}{/if}
{/snippet}
{#snippet emphasized(part: any)}
  {#if part.emphasis}<span class="contract-diff-inline"
      >{@render syntax(part)}</span
    >{:else}{@render syntax(part)}{/if}
{/snippet}
{#snippet text(value: string, view: string, emphasis = [])}
  {#each decoratedText(value, view, v.query, emphasis) as part}{#if part.match}<mark
        >{@render emphasized(part)}</mark
      >{:else}{@render emphasized(part)}{/if}{/each}
{/snippet}
{#snippet lines(side: string)}
  {#each v.rows as row}
    {@const item = row[side]}
    <div class={`contract-diff-row contract-diff-${item?.change || "spacer"}`}>
      <span class="contract-diff-line-number" aria-hidden="true"
        >{item?.number || ""}</span
      ><code class="contract-diff-line-code"
        >{#if item}{@render text(item.text, "base", item.emphasis)}{/if}</code
      >
    </div>
  {/each}
{/snippet}

{#if v}<div class="contract-viewer-shell">
    <header class="contract-viewer-header">
      <div>
        <span class="eyebrow">RESOLVED CONTRACT</span>
        <h2 id="contract-viewer-heading">
          {v.current?.title || "Goal and recipe YAML"}
        </h2>
        <p
          id="contract-viewer-status"
          class="contract-viewer-status"
          aria-live="polite"
        >
          {v.error
            ? "Could not load contract"
            : v.current
              ? [
                  v.current.availability === "static-preview"
                    ? "Static preview"
                    : v.current.availability,
                  v.current.variant_id ? `variant ${v.current.variant_id}` : "",
                ]
                  .filter(Boolean)
                  .join(" · ")
              : "Unavailable"}
        </p>
      </div>
      <button
        id="contract-viewer-close"
        class="quiet icon-only"
        type="button"
        aria-label="Close contract viewer"
        title="Close"
        onclick={() => c.close()}><SourceIcon name="x" /></button
      >
    </header>
    <div class="contract-viewer-toolbar">
      <div
        id="contract-document-tabs"
        class="contract-tabs"
        role="tablist"
        aria-label="Contract document"
      >
        {#each ["goal", "recipe"] as kind}
          <button
            type="button"
            class="contract-tab"
            role="tab"
            data-value={kind}
            disabled={!v.documents[kind]}
            aria-selected={kind === v.documentKind}
            tabindex={kind === v.documentKind ? 0 : -1}
            onclick={() => c.selectDocument(kind)}
            >{kind === "goal" ? "Goal" : "Recipe"}</button
          >
        {/each}
      </div>
      <label
        id="contract-recipe-picker-label"
        class="contract-recipe-picker"
        hidden={!v.recipeItems.length}
        ><span>Recipe</span><select
          id="contract-recipe-picker"
          value={v.selectedRecipeId}
          onchange={(event) =>
            void c
              .selectRecipe(event.currentTarget.value)
              .catch((error: any) =>
                c.showToast(String(error?.message || error), true),
              )}
        >
          {#each v.recipeItems as item}<option value={item.recipe_id}
              >{item.title || item.recipe_id || "Recipe"}</option
            >{/each}</select
        ></label
      >
      <div
        id="contract-view-tabs"
        class="contract-tabs contract-view-tabs"
        role="tablist"
        aria-label="Contract view"
      >
        {#each [["changes", "Changes"], ["base", "Base"], ["resolved", "Resolved"]] as [view, label]}
          <button
            type="button"
            class="contract-tab"
            role="tab"
            data-value={view}
            disabled={view === "changes"
              ? !v.hasChanges
              : !v.current?.views?.[view]}
            aria-selected={view === v.view}
            tabindex={view === v.view ? 0 : -1}
            onclick={() => {
              c.view = view;
              c.render();
            }}>{label}</button
          >
        {/each}
      </div>
      <details
        id="contract-search-disclosure"
        class="contract-search-disclosure"
        open={v.searchOpen}
        ontoggle={(event) => {
          c.searchOpen = event.currentTarget.open;
        }}
      >
        <summary><SourceIcon name="search" /><span>Search</span></summary>
        <label class="contract-search"
          ><SourceIcon name="search" /><input
            id="contract-search-input"
            type="search"
            autocomplete="off"
            placeholder="Search YAML or diff"
            aria-label="Search YAML or diff"
            value={v.query}
            oninput={(event) => {
              c.query = event.currentTarget.value;
              c.render();
            }}
          /><span id="contract-search-count" aria-live="polite">{v.count}</span
          ></label
        >
      </details>
      <button
        id="contract-copy"
        class="quiet button-with-icon"
        type="button"
        disabled={!v.text}
        onclick={() => c.copyCurrent()}
        ><SourceIcon name="copy" /><span>Copy</span></button
      >
    </div>
    <div
      id="contract-viewer-message"
      class="contract-viewer-message"
      hidden={!v.error && !v.current?.message}
    >
      {v.error || v.current?.message || ""}
    </div>
    <div
      id="contract-viewer-loading"
      class="contract-viewer-loading"
      hidden={!v.loading}
    >
      <span class="spinner" aria-hidden="true"></span><span
        >Loading contract…</span
      >
    </div>
    <div
      bind:this={content}
      id="contract-viewer-content"
      class="contract-viewer-content"
      aria-busy={v.loading}
      hidden={v.view === "changes" ? !v.hasChanges : !v.text}
    >
      <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
      <pre
        id="contract-single-content"
        class="contract-single-content"
        tabindex="0"
        hidden={v.view === "changes"}><code>{@render text(v.text, v.view)}</code
        ></pre>
      <div
        id="contract-diff-content"
        class="contract-diff-content"
        aria-label="Side-by-side contract changes"
        hidden={v.view !== "changes" || Boolean(v.diffError)}
      >
        <section
          class="contract-diff-pane"
          aria-labelledby="contract-diff-base-heading"
        >
          <header id="contract-diff-base-heading" class="contract-diff-heading">
            <strong>Base</strong><span id="contract-diff-base-name"
              >{v.current?.kind === "recipe"
                ? "recipe"
                : "goal"}-base.yaml</span
            >
          </header>
          <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
          <div
            bind:this={baseScroll}
            id="contract-diff-base-scroll"
            class="contract-diff-scroll"
            tabindex="0"
            aria-label={`Base YAML, ${v.current?.kind === "recipe" ? "recipe" : "goal"}-base.yaml`}
            onscroll={() => syncScroll(baseScroll, resolvedScroll)}
          >
            <div id="contract-diff-base-lines" class="contract-diff-lines">
              {@render lines("base")}
            </div>
          </div>
        </section>
        <section
          class="contract-diff-pane contract-diff-resolved"
          aria-labelledby="contract-diff-resolved-heading"
        >
          <header
            id="contract-diff-resolved-heading"
            class="contract-diff-heading"
          >
            <strong>Resolved</strong><span id="contract-diff-resolved-name"
              >{v.current?.kind === "recipe"
                ? "recipe"
                : "goal"}-resolved.yaml</span
            >
          </header>
          <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
          <div
            bind:this={resolvedScroll}
            id="contract-diff-resolved-scroll"
            class="contract-diff-scroll"
            tabindex="0"
            aria-label={`Resolved YAML, ${v.current?.kind === "recipe" ? "recipe" : "goal"}-resolved.yaml`}
            onscroll={() => syncScroll(resolvedScroll, baseScroll)}
          >
            <div id="contract-diff-resolved-lines" class="contract-diff-lines">
              {@render lines("resolved")}
            </div>
          </div>
        </section>
      </div>
      <div
        id="contract-diff-error"
        class="contract-diff-error"
        role="alert"
        hidden={!v.diffError}
      >
        {v.diffError}
      </div>
    </div>
  </div>{/if}
