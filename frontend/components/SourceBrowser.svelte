<script lang="ts">
  import { untrack } from "svelte";
  import SourceIcon from "./SourceIcon.svelte";
  import {
    checkpointCanEvaluate,
    checkpointEvaluationPresentation,
    checkpointMetricDescription,
    checkpointMetricIsLoading,
    environmentSuccessStatus,
    formatDate,
    formatGoalConfigurationDate,
    formatGoalDiffValue,
    formatMetricValue,
    goalConfigurationPresentation,
    goalConfigurationSummary,
    groupGoalConfigurations,
    recipeVariantPresentation,
    runEvaluationEvidenceStatus,
    runFinishPresentation,
    runStatePresentation,
    runTrainingEvidenceStatus,
    sortEnvironmentItems,
    successBadgeLabels,
    toggleEnvironmentFavorite,
    writeEnvironmentFavorites,
  } from "../../src/gradlab/web_player/sources/browser.js";

  let {
    controller: c,
    presentation = null,
  }: { controller: any; presentation?: any } = $props();
  let v = $state.raw<any>(untrack(() => presentation));
  let searchInput = $state<HTMLInputElement>();
  export function render(next: any) {
    v = next;
  }
  const refresh = () => {
    c.loadedKey = "";
    void c.load({ force: true, quiet: Boolean(v.items.length) });
  };
  const inspect = (operation: Promise<any>) =>
    void operation.catch((error) =>
      c.showToast(String(error?.message || error), true),
    );
  const openRun = (run: any, variantId = v.route.goal_variant_id) =>
    c.navigate({
      level: "runs",
      goal_variant_id: variantId,
      run_id: String(run.run_id || ""),
      checkpoint_id: "",
    });
  const openRow = (event: MouseEvent, action: () => void) => {
    if (
      !(event.target as HTMLElement).closest("button, details, a, input") &&
      v.hasControl
    )
      action();
  };
  function toggleCheckpoint(id: string, selected: boolean) {
    if (selected) c.selectedCheckpoints.add(id);
    else c.selectedCheckpoints.delete(id);
    c.renderView();
  }
  function toggleAll(selected: boolean) {
    v.items.filter(checkpointCanEvaluate).forEach((item: any) => {
      if (selected) c.selectedCheckpoints.add(item.checkpoint_id);
      else c.selectedCheckpoints.delete(item.checkpoint_id);
    });
    c.renderView();
  }
  function sortColumn(column: any) {
    c.sort = { metric: column.metric, direction: nextSort(column) };
    c.renderView();
  }
  function nextSort(column: any) {
    return v.sort.metric === column.metric
      ? v.sort.direction === "ascending"
        ? "descending"
        : "ascending"
      : column.direction === "min"
        ? "ascending"
        : "descending";
  }
</script>

{#snippet skeleton(label = "Loading table value")}
  <span class="table-skeleton" role="status" aria-label={label}></span>
{/snippet}

{#snippet evidence(status: any, badge = "", prefix = "goal-run-success")}
  <td
    class={`${prefix} ${status.className}`}
    title={badge
      ? `${badge}: ${status.label}. ${status.description}`
      : status.description}
    aria-label={badge
      ? `${badge}: ${status.label}. ${status.description}`
      : status.description}
    aria-busy={status.label === "Loading…" ? true : undefined}
  >
    {#if status.label === "Loading…"}{@render skeleton(
        status.description,
      )}{:else}{status.label}{/if}
  </td>
{/snippet}

{#snippet loadingRows(count: number)}
  {#if v.loading && !v.items.length}
    {#each Array(5) as _}<tr
        >{#each Array(count) as _}<td>{@render skeleton()}</td>{/each}</tr
      >{/each}
  {/if}
{/snippet}

{#snippet loading(message: string)}
  <div class="source-loading">
    <span class="spinner" aria-hidden="true"></span>
    <p>{message}</p>
  </div>
{/snippet}

{#snippet metricHeading(column: any, checkpoints: boolean)}
  {@const active = v.sort.metric === column.metric}
  {@const fullLabel = column.fullLabel || column.label}
  <button
    type="button"
    class="source-sort"
    aria-label={`Sort by ${fullLabel}, ${nextSort(column)}`}
    title={`${fullLabel} · ${checkpoints ? checkpointMetricDescription(column) : column.direction === "min" ? "Lower is better" : "Higher is better"}`}
    onclick={() => sortColumn(column)}
  >
    <span class="source-sort-label"
      ><span>
        {#each String(column.label).split("/") as part, index}{#if index}/{#if checkpoints}<wbr
              />{/if}{/if}{part}{/each}
      </span></span
    >
    <span
      class="source-sort-indicator"
      aria-hidden="true"
      hidden={checkpoints && !active}
    >
      {active ? (v.sort.direction === "ascending" ? "↑" : "↓") : "↕"}
    </span>
  </button>
{/snippet}

{#snippet checkpointMetric(item: any, column: any)}
  {@const pending = checkpointMetricIsLoading(item, column)}
  <td
    class={`checkpoint-metric-cell${column.evidence === "evaluation" ? " checkpoint-eval-metric-cell" : ""}`}
    aria-busy={pending ? true : undefined}
  >
    {#if pending}{@render skeleton(
        `Loading ${column.fullLabel || column.label}`,
      )}
    {:else}<span
        >{formatMetricValue(column.metric, item.metrics?.[column.metric])}</span
      >{/if}
  </td>
{/snippet}

{#snippet checkpoints()}
  {@const t = v.table}
  <div class="source-evaluation-actions">
    {#if v.selectedCheckpoints.size}<span
        >{v.selectedCheckpoints.size.toLocaleString()} selected</span
      >{/if}
    <button
      type="button"
      class="quiet button-with-icon"
      onclick={() => inspect(c.inspectRun())}
    >
      <SourceIcon name="code" /><span>Inspect run YAML</span>
    </button>
    <button
      type="button"
      class="primary button-with-icon"
      disabled={!v.hasControl || !v.selectedCheckpoints.size || v.evaluating}
      onclick={() => c.evaluateSelected()}
      ><SourceIcon name="player-play" /><span>
        {v.evaluating
          ? "Adding to queue…"
          : v.selectedCheckpoints.size
            ? `Evaluate ${v.selectedCheckpoints.size.toLocaleString()}`
            : "Evaluate selected"}
      </span></button
    >
  </div>
  <div class="source-table-scroll">
    <table
      class="source-table checkpoint-table"
      style:min-width={`${t.totalWidth}ch`}
    >
      <colgroup
        >{#each t.widths as width}<col
            style:width={`${(width / t.totalWidth) * 100}%`}
          />{/each}</colgroup
      >
      <thead>
        <tr class="checkpoint-group-row">
          <th
            scope="col"
            rowspan="2"
            class="source-selection-cell checkpoint-fixed-header"
          >
            <input
              type="checkbox"
              aria-label="Select all eligible checkpoints"
              checked={v.items.some(checkpointCanEvaluate) &&
                v.items
                  .filter(checkpointCanEvaluate)
                  .every((item: any) =>
                    v.selectedCheckpoints.has(item.checkpoint_id),
                  )}
              disabled={!v.items.some(checkpointCanEvaluate) || v.evaluating}
              onchange={(event) => toggleAll(event.currentTarget.checked)}
            />
          </th>
          <th scope="col" rowspan="2" class="checkpoint-fixed-header">Step</th>
          {#if t.training.length || t.trainingPlaceholder}<th
              scope="colgroup"
              colspan={t.training.length || 1}
              class="checkpoint-train-group">Train</th
            >{/if}
          <th
            scope="colgroup"
            colspan={t.evaluation.length + 1}
            class="checkpoint-eval-group">Eval</th
          >
        </tr>
        <tr>
          {#each t.training as column}<th
              scope="col"
              class="checkpoint-key-header checkpoint-train-header"
              aria-sort={v.sort.metric === column.metric
                ? v.sort.direction
                : "none"}>{@render metricHeading(column, true)}</th
            >{/each}
          {#if t.trainingPlaceholder}<th
              scope="col"
              class="checkpoint-train-header"
            ></th>{/if}
          <th scope="col" class="checkpoint-eval-header">Status</th>
          {#each t.evaluation as column}<th
              scope="col"
              class="checkpoint-key-header checkpoint-eval-header"
              aria-sort={v.sort.metric === column.metric
                ? v.sort.direction
                : "none"}>{@render metricHeading(column, true)}</th
            >{/each}
        </tr>
      </thead>
      <tbody aria-busy={v.loading && !v.items.length ? true : undefined}>
        {#each t.items as item (item.checkpoint_id)}
          {@const status = checkpointEvaluationPresentation(item)}
          <!-- svelte-ignore a11y_no_noninteractive_element_to_interactive_role -->
          <tr
            class:selected={v.selectedCheckpoints.has(item.checkpoint_id)}
            tabindex={v.hasControl ? 0 : -1}
            role="button"
            aria-disabled={!v.hasControl}
            onclick={(event) => openRow(event, () => c.selectCheckpoint(item))}
            onkeydown={(event) => {
              if (
                event.target === event.currentTarget &&
                ["Enter", " "].includes(event.key) &&
                v.hasControl
              ) {
                event.preventDefault();
                c.selectCheckpoint(item);
              }
            }}
          >
            <td class="source-selection-cell"
              ><input
                type="checkbox"
                checked={v.selectedCheckpoints.has(item.checkpoint_id)}
                disabled={!checkpointCanEvaluate(item) || v.evaluating}
                aria-label={checkpointCanEvaluate(item)
                  ? `Select ${item.checkpoint_id} for evaluation`
                  : `${item.checkpoint_id} cannot be evaluated again`}
                onchange={(event) =>
                  toggleCheckpoint(
                    item.checkpoint_id,
                    event.currentTarget.checked,
                  )}
              /></td
            >
            <td class="checkpoint-step-cell"
              ><div class="checkpoint-step">
                <span title={item.checkpoint_id}
                  >{Number(item.step).toLocaleString()}</span
                >
                {#each item.representative_media || [] as media}<a
                    href={media.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    aria-label={`${media.kind === "monitoring" ? "Monitoring video" : "Evaluation video"} at step ${item.step}`}
                  >
                    {media.kind === "monitoring"
                      ? "Monitoring video"
                      : "Evaluation video"}</a
                  >{/each}
                {#if /final/i.test(String(item.purpose || ""))}<span
                    class="checkpoint-purpose-badge"
                    aria-label={`Checkpoint purpose: ${item.purpose}`}
                    >{item.purpose}</span
                  >{/if}
              </div></td
            >
            {#each t.training as column}{@render checkpointMetric(
                item,
                column,
              )}{/each}
            {#if t.trainingPlaceholder}<td>{@render skeleton()}</td>{/if}
            <td class="checkpoint-eval-status-cell"
              ><span
                class={`checkpoint-eval-status ${status.tone}`}
                title={status.title || undefined}
              >
                {status.label}{#if status.progress}<small
                    >{status.progress}</small
                  >{/if}</span
              ></td
            >
            {#each t.evaluation as column}{@render checkpointMetric(
                item,
                column,
              )}{/each}
          </tr>
        {/each}
        {@render loadingRows(t.columns.length)}
      </tbody>
    </table>
  </div>
{/snippet}

{#snippet runs()}
  {@const t = v.table}
  <div
    class="source-table-scroll"
    class:training-leader={t.efficiency?.evidence === "training"}
  >
    <table class="source-table">
      <thead
        ><tr
          >{#each t.columns as column}<th
              scope="col"
              aria-sort={column.metric
                ? v.sort.metric === column.metric
                  ? v.sort.direction
                  : "none"
                : undefined}
            >
              {#if column.metric}{@render metricHeading(
                  column,
                  false,
                )}{:else}{column.label}{/if}</th
            >{/each}</tr
        ></thead
      >
      <tbody aria-busy={v.loading && !v.items.length ? true : undefined}>
        {#each t.items as item (item.run_id)}
          {@const finish = runFinishPresentation(item)}
          {@const state = runStatePresentation(item)}
          {@const variant = recipeVariantPresentation(item)}
          {@const leader = t.efficiency?.item?.run_id === item.run_id}
          <!-- svelte-ignore a11y_no_noninteractive_element_to_interactive_role -->
          <tr
            class:efficiency-leader={leader}
            tabindex={v.hasControl ? 0 : -1}
            role="button"
            aria-disabled={!v.hasControl}
            onclick={(event) => openRow(event, () => openRun(item))}
            onkeydown={(event) => {
              if (
                event.target === event.currentTarget &&
                ["Enter", " "].includes(event.key) &&
                v.hasControl
              ) {
                event.preventDefault();
                openRun(item);
              }
            }}
          >
            <td class="run-cell"
              ><button
                type="button"
                class="run-identity"
                disabled={!v.hasControl}
                onclick={() => openRun(item)}
              >
                <span
                  class={`run-state ${state.tone}`}
                  title={`Run state: ${state.label}`}
                  aria-label={`Run state: ${state.label}`}
                  ><SourceIcon name={state.iconName} /></span
                >
                <div class="run-identity-text">
                  <span class="run-name"
                    >{item.description || item.name || item.run_id}</span
                  ><small>{item.run_id}</small>
                </div>
              </button>
              {#if successBadgeLabels(item).length}<span class="success-badges"
                  >{#each successBadgeLabels(item) as label}<span
                      class={`success-badge ${label.startsWith("train/") ? "training" : "evaluation"}`}
                      title={label === "train/success"
                        ? "A run reached this training goal's success condition"
                        : "A verified evaluation reached this goal's acceptance condition"}
                      >{label}</span
                    >{/each}</span
                >{/if}
              {#if leader}<span class="source-leader-badge"
                  >{t.efficiency.evidence === "evaluation"
                    ? "Most efficient"
                    : "Training lead"}</span
                >{/if}
            </td>
            <td class="recipe-cell"
              ><span>{item.recipe || "—"}</span><small
                class="recipe-variant"
                title={variant.detail}>{variant.summary}</small
              >
              {#if item.recipe_sha256}<small
                  class="recipe-revision"
                  title={`Recipe SHA-256: ${item.recipe_sha256}`}
                  >rev {String(item.recipe_sha256).slice(0, 12)}</small
                >{/if}</td
            >
            <td class="data-cell">{item.seed ?? "—"}</td>
            <td class={`finish-reason ${finish.tone}`}>
              {#if finish.evidence}<span class="finish-status"
                  >{finish.label}</span
                >
                <div class="finish-evidence">
                  {#each [["Observed", finish.evidence.observed], ["Required", finish.evidence.required]] as [label, value]}<div
                      class="finish-evidence-item"
                    >
                      <strong class="finish-evidence-value">{value}</strong
                      ><span class="finish-evidence-label">{label}</span>
                    </div>{/each}
                </div>
                <small class="finish-evidence-metric"
                  >{finish.evidence.metric}</small
                >{#if finish.evidence.step}<small class="finish-evidence-step"
                    >{finish.evidence.step}</small
                  >{/if}
              {:else}<span>{finish.label}</span>{#if finish.detail}<small
                    >{finish.detail}</small
                  >{/if}{/if}
            </td>
            {#each t.metrics as column}<td class="data-cell"
                >{formatMetricValue(
                  column.metric,
                  item.metrics?.[column.metric],
                )}</td
              >{/each}
            <td class="data-cell"
              >{formatDate(item.updated_at || item.created_at)}</td
            >
            <td class="inspection-cell"
              ><button
                type="button"
                class="quiet button-with-icon"
                onclick={() => inspect(c.inspectRun(item.run_id))}
                ><SourceIcon name="code" /><span>Inspect</span></button
              ></td
            >
          </tr>
        {/each}
        {@render loadingRows(t.columns.length)}
      </tbody>
    </table>
  </div>
{/snippet}

{#snippet environmentsAndGoals()}
  {@const environments = v.route.level === "environments"}
  <div class={environments ? "environment-table-scroll" : "goal-table-scroll"}>
    <table class={environments ? "environment-table" : "goal-table"}>
      {#if environments}<colgroup
          ><col class="environment-favorite-column" /><col
            class="environment-name-column"
          /></colgroup
        >
        <colgroup
          ><col class="environment-goals-column" /><col
            class="environment-status-column"
          /><col class="environment-status-column" /></colgroup
        >{/if}
      <thead
        ><tr
          ><th
            scope={environments ? "colgroup" : "col"}
            colspan={environments ? 2 : 1}
            class:environment-heading={environments}
            >{environments ? "Environment" : "Goal"}</th
          >
          {#each [environments ? "Goals" : "Recipes", "train/success", "eval/success"] as label, index}<th
              scope="col"
              class={environments
                ? index
                  ? "environment-status-column"
                  : "environment-goals-column"
                : undefined}>{label}</th
            >{/each}
          {#if !environments}<th scope="col">YAML</th>{/if}</tr
        ></thead
      >
      <tbody aria-busy={v.loading && !v.items.length ? true : undefined}>
        {#each environments ? sortEnvironmentItems(v.items, v.favoriteEnvironments) : v.items as item (item.name || item.goal_id)}
          {@const route = environments
            ? {
                level: "goals",
                environment_id: item.name,
                goal_id: "",
                goal_variant_id: "",
                run_id: "",
                checkpoint_id: "",
              }
            : {
                level: "goal_variants",
                goal_id: item.goal_id,
                goal_variant_id: "",
                run_id: "",
                checkpoint_id: "",
              }}
          <!-- svelte-ignore a11y_click_events_have_key_events a11y_no_noninteractive_element_interactions -->
          <tr
            class={environments ? "environment-row" : "goal-row"}
            onclick={(event) => openRow(event, () => c.navigate(route))}
          >
            {#if environments}
              {@const favorite = v.favoriteEnvironments.has(item.name)}
              <td class="environment-favorite-cell"
                ><button
                  type="button"
                  class="environment-favorite"
                  class:selected={favorite}
                  aria-pressed={favorite}
                  title={`${favorite ? "Remove" : "Add"} ${item.name} ${favorite ? "from" : "to"} favorites`}
                  aria-label={`${favorite ? "Remove" : "Add"} ${item.name} ${favorite ? "from" : "to"} favorites`}
                  onclick={() => {
                    c.favoriteEnvironments = toggleEnvironmentFavorite(
                      v.favoriteEnvironments,
                      item.name,
                    );
                    writeEnvironmentFavorites(c.favoriteEnvironments);
                    c.renderView();
                  }}
                >
                  <SourceIcon
                    name={favorite ? "star-filled" : "star"}
                  /></button
                ></td
              >
            {/if}
            <td
              ><button
                type="button"
                class={environments
                  ? "environment-row-navigation"
                  : "goal-row-navigation"}
                disabled={!v.hasControl}
                title={!environments && item.goal_slug !== item.goal_id
                  ? `Goal slug: ${item.goal_slug}`
                  : undefined}
                onclick={() => c.navigate(route)}
              >
                {#if environments}{item.name}{:else}<div
                    class="goal-row-identity"
                  >
                    <strong>{item.goal_id}</strong><span
                      >{item.title || item.goal_slug}</span
                    >
                  </div>{/if}
              </button></td
            >
            <td
              >{Number(
                environments ? item.goal_count : item.recipe_count,
              ).toLocaleString()}</td
            >
            {@render evidence(
              environmentSuccessStatus(item, "train/success"),
              "",
              environments ? "environment-status" : "goal-status",
            )}
            {@render evidence(
              environmentSuccessStatus(item, "eval/success"),
              "",
              environments ? "environment-status" : "goal-status",
            )}
            {#if !environments}<td
                ><button
                  type="button"
                  class="quiet goal-row-inspect icon-only"
                  title={`Inspect ${item.goal_id} YAML`}
                  aria-label={`Inspect ${item.goal_id} YAML`}
                  onclick={() => inspect(c.inspectGoal(item))}
                  ><SourceIcon name="code" /></button
                ></td
              >{/if}
          </tr>
        {/each}
        {@render loadingRows(5)}
      </tbody>
    </table>
  </div>
{/snippet}

{#snippet configurationRuns(variant: any)}
  {@const page = v.goalVariantRunPages.get(String(variant.variant_id))}
  {@const items = page?.loaded ? page.items : variant.recent_runs || []}
  <section class="goal-configuration-runs">
    {#if !items.length}<p class="goal-configuration-runs-empty">
        No runs use this configuration yet.
      </p>
    {:else}<div class="goal-configuration-run-scroll">
        <table class="goal-configuration-run-table">
          <thead
            ><tr
              >{#each ["Run", "State", "Training", "Evaluation", ""] as label}<th
                  scope="col">{label}</th
                >{/each}</tr
            ></thead
          >
          <tbody
            >{#each items as run (run.run_id)}
              {@const state = runStatePresentation(run)}
              {@const name = [
                run.recipe || run.name || run.run_id || "Run",
                run.seed != null ? `seed ${run.seed}` : "",
              ]
                .filter(Boolean)
                .join(" · ")}
              <!-- svelte-ignore a11y_click_events_have_key_events a11y_no_noninteractive_element_interactions -->
              <tr
                onclick={(event) =>
                  openRow(event, () => openRun(run, variant.variant_id))}
              >
                <td
                  ><button
                    type="button"
                    class="goal-configuration-run-identity"
                    disabled={!v.hasControl}
                    onclick={() => openRun(run, variant.variant_id)}
                    ><strong>{name}</strong></button
                  >
                  <small
                    class="goal-run-activity"
                    title={run.updated_at ? formatDate(run.updated_at) : ""}
                    >{run.updated_at
                      ? formatGoalConfigurationDate(run.updated_at)
                      : "Activity unavailable"}</small
                  >
                  <details class="goal-run-details">
                    <summary>Details</summary>
                    <p>{run.description || "No run description available."}</p>
                    <p>
                      {[run.name, recipeVariantPresentation(run).detail]
                        .filter(Boolean)
                        .join(" · ")}
                    </p>
                    <code>{run.run_id}</code>
                    <button
                      type="button"
                      class="quiet"
                      onclick={() =>
                        void navigator.clipboard
                          .writeText(String(run.run_id || ""))
                          .catch(() =>
                            c.showToast("Could not copy run ID", true),
                          )}>Copy ID</button
                    >
                  </details>
                </td>
                <td
                  class={`goal-run-state ${state.tone}`}
                  title={String(run.state || "unknown").replace(/_/g, " ")}
                  aria-label={String(run.state || "unknown").replace(/_/g, " ")}
                  >{state.label}</td
                >
                {@render evidence(
                  runTrainingEvidenceStatus(run),
                  "train/success",
                )}
                {@render evidence(
                  runEvaluationEvidenceStatus(run),
                  "eval/success",
                )}
                <td
                  ><button
                    type="button"
                    class="quiet icon-only"
                    title="View checkpoints"
                    aria-label={`View checkpoints for ${name}`}
                    disabled={!v.hasControl}
                    onclick={() => openRun(run, variant.variant_id)}
                    ><SourceIcon name="arrow-right" /></button
                  ></td
                >
              </tr>
            {/each}</tbody
          >
        </table>
      </div>{/if}
    {#if page?.error}<p class="source-inline-error">{page.error}</p>{/if}
    {#if c.embeddedGoalRunsHaveMore(variant)}<button
        type="button"
        class="quiet button-with-icon"
        disabled={Boolean(page?.loading)}
        onclick={() =>
          c.loadEmbeddedGoalRuns(variant, {
            append: Boolean(page?.items?.length),
          })}><SourceIcon name="refresh" /><span>Load more runs</span></button
      >{/if}
  </section>
{/snippet}

{#snippet configurations()}
  {@const selected = v.items.find(
    (item: any) => item.variant_id === v.selectedGoalVariantId,
  )}
  <div
    class="goal-configuration-browser"
    aria-busy={!selected && v.loading ? true : undefined}
  >
    <div class="goal-configuration-layout">
      <aside class="goal-configuration-list" aria-label="Goal configurations">
        {#if !selected}<h3>Goal configurations</h3>
          {#each Array(4) as _}<div
              class="goal-configuration-option goal-configuration-placeholder"
            >
              {#each Array(4) as _}{@render skeleton(
                  "Loading goal configuration",
                )}{/each}
            </div>{/each}
        {:else}{#each groupGoalConfigurations(v.items) as group, index}
            {#if index === 0 || group.current !== groupGoalConfigurations(v.items)[index - 1].current}<h3
              >
                {group.current ? "Current" : "History"}
              </h3>{/if}
            <section class="goal-configuration-group">
              <h4>
                <code title={group.revisionId}
                  >{group.revisionId.slice(0, 8)}</code
                >
              </h4>
              {#each [group.defaultVariant, ...group.overrides].filter(Boolean) as variant (variant.variant_id)}
                {@const p = goalConfigurationPresentation(variant)}
                {@const active = variant.variant_id === v.selectedGoalVariantId}
                <div class="goal-configuration-entry" class:selected={active}>
                  <button
                    type="button"
                    id={`goal-configuration-${encodeURIComponent(variant.variant_id)}`}
                    class="goal-configuration-option"
                    class:selected={active}
                    aria-pressed={active}
                    onclick={() => c.selectGoalVariant(variant)}
                  >
                    <strong>{p.behaviorLabel}</strong><span
                      class="goal-configuration-run-count">{p.runLabel}</span
                    >
                    <span class="goal-configuration-option-summary"
                      >{goalConfigurationSummary(variant, p).replace(
                        / changes total$/,
                        " changes",
                      )}</span
                    >
                    <span class="goal-configuration-option-meta"
                      >{#if p.runCount === 0}No runs yet...{:else}
                        <span
                          >First used <time
                            datetime={variant.first_used_at}
                            title={formatDate(variant.first_used_at)}
                            >{p.firstUsedDate}</time
                          ></span
                        >
                        <span
                          >Last activity <time
                            datetime={variant.last_activity_at}
                            title={formatDate(variant.last_activity_at)}
                            >{p.lastActivityDate}</time
                          ></span
                        >{/if}</span
                    >
                  </button>
                </div>
              {/each}
            </section>
          {/each}{/if}
      </aside>
      <section
        class="goal-configuration-panel"
        aria-label="Selected goal configuration"
      >
        {#if !selected}{#each ["goal-configuration-panel-header", "goal-configuration-runs", "goal-configuration-baseline"] as name}<div
              class={`${name} goal-configuration-placeholder`}
            >
              {@render skeleton(
                "Loading configuration details",
              )}{@render skeleton()}
            </div>{/each}
        {:else}
          {@const p = goalConfigurationPresentation(selected)}
          <header class="goal-configuration-panel-header">
            <div class="goal-configuration-identity">
              <h3>
                {p.behaviorLabel} · {p.kind.startsWith("current_")
                  ? "Current"
                  : "Historical"}
              </h3>
              <code title={selected.goal_contract_sha256}
                >{String(selected.goal_contract_sha256 || "").slice(0, 8) ||
                  "unknown"}</code
              >
            </div>
            <button
              type="button"
              class="quiet button-with-icon goal-configuration-inspect"
              aria-label="View goal YAML"
              onclick={() =>
                inspect(
                  p.kind === "current_default"
                    ? c.inspectGoal({ goal_id: v.route.goal_id })
                    : c.inspectGoalVariant(selected),
                )}><SourceIcon name="code" /><span>YAML</span></button
            >
          </header>
          {#if p.kind === "current_default"}<p
              class="goal-configuration-baseline"
            >
              No contract differences. This configuration matches the current
              checked-in goal.
            </p>
          {:else}<details
              class="goal-configuration-differences"
              id="selected-goal-configuration-differences"
            >
              <summary
                >{p.differenceLabel.replace(/changes?$/, (word: string) =>
                  word === "change" ? "difference" : "differences",
                )} from current</summary
              >
              {#if !p.comparisonAvailable}<div
                  class="goal-configuration-diff-empty warning"
                >
                  The exact historical contract is not sufficiently proven, so
                  no field-level comparison is shown.
                </div>
              {:else if !v.goalVariantDiff || v.goalVariantDiff.state === "loading"}{@render loading(
                  "Loading exact contract differences…",
                )}
              {:else if v.goalVariantDiff.state === "error" || v.goalVariantDiff.availability !== "exact"}<div
                  class="goal-configuration-diff-empty warning"
                >
                  {v.goalVariantDiff.message ||
                    "An exact field-level comparison is unavailable."}
                </div>
              {:else if !v.goalVariantDiff.entries.length}<div
                  class="goal-configuration-diff-empty"
                >
                  This configuration has no behavioral differences from the
                  current checked-in goal.
                </div>
              {:else}{#if v.goalVariantDiff.message}<p>
                    {v.goalVariantDiff.message}
                  </p>{/if}
                <div class="goal-configuration-diff-scroll">
                  <table class="goal-configuration-diff-table">
                    <thead
                      ><tr
                        >{#each ["Exact contract path", "Before", "After"] as label}<th
                            scope="col">{label}</th
                          >{/each}</tr
                      ></thead
                    >
                    <tbody
                      >{#each v.goalVariantDiff.entries as entry}{@const kind =
                          ["added", "removed", "changed"].includes(entry.kind)
                            ? entry.kind
                            : "changed"}
                        <tr
                          ><td class="goal-configuration-path"
                            ><code>{entry.path}</code></td
                          ><td class="goal-configuration-value"
                            ><code
                              >{formatGoalDiffValue(entry.before, {
                                unavailable: kind === "added",
                              })}</code
                            ></td
                          >
                          <td
                            class={`goal-configuration-value goal-configuration-after ${kind}`}
                            ><code
                              >{formatGoalDiffValue(entry.after, {
                                unavailable: kind === "removed",
                              })}</code
                            ></td
                          ></tr
                        >
                      {/each}</tbody
                    >
                  </table>
                </div>{/if}
            </details>{/if}
          {@render configurationRuns(selected)}
        {/if}
      </section>
    </div>
  </div>
{/snippet}

{#if v}
  <section class="source-shell">
    <div class="source-head">
      <div class="source-title">
        <span class="eyebrow">PLAYBACK SOURCE</span>
        <h2>{v.heading}</h2>
        {#if v.route.level === "runs" && v.route.run_id}
          {@const status = v.runStatus?.state
            ? runStatePresentation(v.runStatus)
            : {
                iconName:
                  v.loading && !v.loadedKey ? "refresh" : "activity-heartbeat",
                tone: v.loading && !v.loadedKey ? "pending" : "unknown",
                label: v.loading && !v.loadedKey ? "Loading…" : "Unavailable",
              }}
          <div
            class={`source-run-status ${status.tone}`}
            class:loading={status.tone === "pending"}
            aria-label={`Training run state: ${status.label}${v.runStatus?.updated_at ? `. Updated ${formatDate(v.runStatus.updated_at)}` : ""}`}
          >
            <SourceIcon name={status.iconName} /><span>Training run</span
            ><strong>{status.label}</strong>
            {#if v.runStatus?.updated_at}<small
                >Updated {formatDate(v.runStatus.updated_at)}</small
              >{/if}
          </div>
        {/if}
      </div>
      {#if v.app.phase === "selecting"}<button
          type="button"
          class="quiet icon-only"
          class:refreshing={v.refreshing}
          aria-label={v.refreshing ? "Refreshing" : "Refresh"}
          aria-busy={v.refreshing}
          title={v.refreshing ? "Refreshing this list" : "Refresh this list"}
          disabled={v.refreshing}
          onclick={refresh}><SourceIcon name="refresh" /></button
        >{/if}
    </div>
    {#if !v.hasControl}<p class="source-notice">
        Observer window — choose Control here to change the shared run.
      </p>{/if}
    {#if v.app.phase === "error"}<div class="source-centered source-failure">
        <p>{v.app.error || "The checkpoint could not be opened."}</p>
        <div class="source-actions">
          <button
            type="button"
            class="primary button-with-icon"
            disabled={!v.hasControl}
            onclick={() => c.command("retry_source")}
            ><SourceIcon name="refresh" /><span>Retry</span></button
          >
          <button
            type="button"
            class="quiet button-with-icon"
            disabled={!v.hasControl}
            onclick={() => c.browseCurrentSource()}
            ><SourceIcon name="folder-search" /><span>Choose another</span
            ></button
          >
          {#if v.app.has_active_runner}<button
              type="button"
              class="quiet button-with-icon"
              disabled={!v.hasControl}
              onclick={() => c.command("cancel_source")}
              ><SourceIcon name="arrow-left" /><span>Back to current run</span
              ></button
            >{/if}
        </div>
      </div>
    {:else if ["resolving", "verifying", "loading"].includes(v.app.phase)}<div
        class="source-centered"
      >
        {@render loading(v.app.message || "Preparing playback…")}
        {#if v.app.has_active_runner}<button
            type="button"
            class="quiet button-with-icon"
            disabled={!v.hasControl}
            onclick={() => c.command("cancel_source")}
            ><SourceIcon name="arrow-left" /><span>Back to current run</span
            ></button
          >{/if}
      </div>
    {:else}
      {#if v.route.level !== "goal_variants" || v.sourceItems.length > 8 || v.query || v.searchOpen}
        <details
          class="source-search-disclosure"
          open={v.searchOpen}
          ontoggle={(event) => {
            c.searchOpen = event.currentTarget.open;
            if (c.searchOpen) searchInput?.focus({ preventScroll: true });
          }}
        >
          <summary><SourceIcon name="search" />Search</summary>
          <div class="source-search">
            <SourceIcon name="search" />
            <input
              bind:this={searchInput}
              type="search"
              value={v.query}
              autocomplete="off"
              placeholder={v.searchPlaceholder}
              aria-label={v.searchPlaceholder}
              oninput={(event) => c.setSearch(event.currentTarget.value)}
            />
            <button
              type="button"
              class="quiet source-search-close icon-only"
              aria-label="Close search"
              title="Close search"
              onclick={() => {
                c.searchOpen = false;
                if (v.query) c.setSearch("");
                else c.renderView();
              }}><SourceIcon name="x" /></button
            >
          </div>
        </details>
      {/if}
      <div
        class="source-results"
        class:source-results-configurations={v.route.level === "goal_variants"}
        class:source-checkpoint-results={v.route.level === "runs" &&
          v.route.run_id}
        class:loading={v.loading}
        class:loading-empty={v.loading && !v.items.length}
      >
        {#if v.error}<div class="source-inline-error">
            <p>{v.error}</p>
            <button
              type="button"
              class="button-with-icon"
              onclick={() => {
                c.loadedKey = "";
                void c.load();
              }}><SourceIcon name="refresh" /><span>Retry</span></button
            >
          </div>{/if}
        {#if !v.items.length && !v.loading && v.loadedKey}<div
            class="source-empty"
          >
            <strong
              >{v.route.run_id
                ? "No public checkpoints yet"
                : "No matching results"}</strong
            >
            <p>
              {v.route.run_id
                ? "This run has not published a playable checkpoint to public model storage."
                : "Try a broader search."}
            </p>
          </div>
        {:else if !v.error || v.items.length}
          {#if ["environments", "goals"].includes(v.route.level)}{@render environmentsAndGoals()}
          {:else if v.route.level === "goal_variants"}{@render configurations()}
          {:else if v.route.run_id}{@render checkpoints()}
          {:else}{@render runs()}{/if}
          {#if v.nextCursor}<button
              type="button"
              class="source-load-more"
              disabled={v.loading}
              onclick={() => c.load({ append: true })}
              >{v.loading ? "Loading…" : "Load more"}</button
            >{/if}
        {/if}
      </div>
    {/if}
  </section>
{/if}
