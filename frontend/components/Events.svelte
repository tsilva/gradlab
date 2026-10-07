<script lang="ts">
  import { onDestroy } from "svelte";
  import Panel from "./Panel.svelte";
  import { eventAtCursor } from "../../src/gradlab/web_player/panels/events.js";
  import {
    eventColor,
    eventColorFill,
    eventLabels,
  } from "../../src/gradlab/web_player/event-colors.js";
  let { definition, services } = $props();
  let container: HTMLDivElement;
  let points = $state.raw<any[]>([]),
    view = $state.raw<any>({}),
    cursorPoint = $state.raw<any>(null),
    status = $state(""),
    recorded = $state(false);
  let identity: string | null = null,
    nextLast: number | null = null,
    expanded = false,
    pending = false,
    loaded = false,
    revision = 0,
    updated = 0,
    disposed = false;
  let visible = $derived(
    cursorPoint && !points.some((point) => eventAtCursor(point, view))
      ? [...points, cursorPoint].sort((a, b) => b.step - a.step)
      : points,
  );
  async function load(append = false) {
    if (pending || !identity || disposed) return;
    pending = true;
    const request = revision;
    if (!loaded && !status) status = "Loading events…";
    try {
      const result = await services.loadEvents(
        identity.split(":").slice(1).join(":"),
        append ? nextLast : null,
      );
      if (request !== revision || disposed) return;
      if (append || !expanded) nextLast = result.next_last;
      if (append) {
        const existing = new Set(points.map((point) => point.step));
        points = [
          ...points,
          ...result.points.filter((point: any) => !existing.has(point.step)),
        ];
        expanded = true;
      } else if (expanded) {
        const merged = new Map(points.map((point) => [point.step, point]));
        for (const point of result.points) merged.set(point.step, point);
        points = [...merged.values()].sort((a, b) => b.step - a.step);
      } else points = result.points;
      loaded = true;
      status = nextLast === null ? "" : "Scroll down for older events";
      updated = Date.now();
    } catch (error) {
      if (request === revision && !disposed) {
        status = (error as Error).message;
        updated = Date.now();
      }
    } finally {
      pending = false;
      if (request !== revision && !disposed) void load();
    }
  }
  export function renderHistory(
    history: any[],
    snapshot: any = null,
    next: any = {},
  ) {
    view = {
      ...next,
      selectedStep: snapshot?.transition?.step,
      selectedEpisode:
        snapshot?.transition?.episode ?? snapshot?.session?.episode,
    };
    cursorPoint =
      history.find(
        (point) =>
          eventAtCursor(point, view) &&
          (point.boundary || point.events?.length),
      ) || null;
    const episode = services.getState?.().liveSnapshot?.trajectory?.episode_id;
    const key = episode ? `${view.sessionEpoch}:${episode}` : null;
    recorded = Boolean(key);
    const panel = container?.closest<HTMLElement>("[data-panel]");
    if (key !== identity) {
      identity = key;
      revision++;
      expanded = false;
      loaded = false;
      if (panel) panel.scrollTop = 0;
      status = key ? "Loading events…" : "";
      nextLast = null;
      updated = 0;
      points = [];
    }
    if (key) {
      if ((!panel || panel.scrollTop === 0) && Date.now() - updated >= 1000)
        void load();
    } else
      points = history
        .filter((point) => point.boundary || point.events?.length)
        .reverse();
  }
  function scroll(event: Event) {
    const panel = event.target as HTMLElement;
    if (
      nextLast !== null &&
      panel.clientHeight > 0 &&
      panel.scrollHeight - panel.scrollTop - panel.clientHeight <= 1
    )
      void load(true);
  }
  function observe(node: HTMLElement) {
    const panel = node.closest("[data-panel]")!;
    panel.addEventListener("scroll", scroll, { passive: true });
    return {
      destroy() {
        panel.removeEventListener("scroll", scroll);
      },
    };
  }
  onDestroy(() => {
    disposed = true;
    revision++;
  });
</script>

<Panel {definition}>
  <div bind:this={container} use:observe style="display:contents">
    {#if recorded && status && !visible.length}
      <div class="chart-status" role="status">{status}</div>
    {/if}
    <ol data-list class="event-list">
      {#each visible as point (`${point.episode}:${point.step}`)}
        {@const labels = eventLabels(point)}{@const selected = eventAtCursor(
          point,
          view,
        )}
        <li
          class="event-item"
          class:boundary={point.boundary}
          class:selected
          style:--event-colors={eventColorFill(labels)}
        >
          <button
            type="button"
            class="event-jump"
            aria-label={`Inspect ${labels.join(" · ")} at episode ${point.episode}, step ${point.step}`}
            aria-current={selected ? "step" : undefined}
            onclick={() =>
              recorded
                ? services.inspectStep(point.step)
                : services.inspectSequence(point.sequence)}
            ><div class="event-labels">
              {#each labels as label}<span
                  class="event-label"
                  style:--event-color={eventColor(label)}>{label}</span
                >{/each}
            </div>
            <div class="event-meta">step {point.step}</div></button
          >
        </li>
      {:else}{#if !recorded || !status}<li class="empty-state widget-empty">
            No data available yet
          </li>{/if}{/each}
    </ol>
    {#if recorded && status && visible.length}
      <div class="chart-status" role="status">{status}</div>
    {/if}
  </div>
</Panel>
