<script lang="ts">
  import { shellState } from "../shell-state.svelte";
  const p = shellState.publication;
</script>

{#snippet facts(items: any[])}
  {#each items as [term, value]}<dt>{term}</dt>
    <dd>{String(value ?? "—")}</dd>{/each}
{/snippet}

<dialog
  id="publication-dialog"
  class="publication-dialog"
  aria-labelledby="publication-heading"
>
  <div class="publication-shell">
    <header class="publication-header">
      <div>
        <span class="eyebrow">COMBINED RELEASE</span>
        <h2 id="publication-heading">Publish episode</h2>
        <p
          id="publication-status"
          class="publication-status"
          aria-live="polite"
        >
          {p.status}
        </p>
      </div>
      <button
        id="publication-close"
        class="quiet icon-only"
        type="button"
        aria-label="Close publication dialog"
        title="Close"
        ><svg class="icon" aria-hidden="true"
          ><use href="/assets/tabler-icons.svg#ti-x"></use></svg
        ></button
      >
    </header>
    <div class="publication-body">
      <section class="publication-preview">
        <!-- svelte-ignore a11y_media_has_caption -->
        <video
          id="publication-video"
          src={p.videoUrl || undefined}
          controls
          preload="metadata"
        ></video>
        <dl id="publication-capture" class="publication-facts">
          {@render facts(p.capture)}
        </dl>
        <dl
          id="publication-generated"
          class="publication-facts publication-generated"
        >
          {@render facts(p.generated)}
        </dl>
      </section>
      <form id="publication-form" class="publication-form">
        <div class="publication-fields">
          <label
            ><span>Privacy</span><select
              id="publication-privacy"
              bind:value={p.privacy}
              ><option value="public">Public</option><option value="unlisted"
                >Unlisted</option
              ><option value="private">Private</option></select
            ></label
          >
          <label
            ><span>Thumbnail time (seconds)</span><input
              id="publication-thumbnail-time"
              type="number"
              min="0"
              step="0.25"
              bind:value={p.thumbnailTime}
              required
            /></label
          >
          >
          <label class="publication-wide"
            ><span>Extra tags (comma-separated)</span><input
              id="publication-tags"
              bind:value={p.tags}
              maxlength="300"
            /></label
          >
          <label class="publication-wide"
            ><span
              >Operator note <small>clearly labeled as unverified</small></span
            ><textarea
              id="publication-note"
              bind:value={p.note}
              rows="3"
              maxlength="1000"
            ></textarea></label
          >
        </div>
        <div
          id="publication-credentials"
          class="publication-credentials"
          style:white-space="pre-line"
        >
          {p.credentials}
        </div>
        <div id="publication-job" class="publication-job" hidden={!p.job}>
          {#if p.job}<div>
              {p.job.state || "queued"} · {p.job.progress?.phase || "queued"}{p
                .job.message
                ? ` · ${p.job.message}`
                : ""}
            </div>
            {#each Object.entries(p.job.urls || {}) as [label, url]}{#if String(url).startsWith("https://")}<div
                >
                  <a
                    href={String(url)}
                    target="_blank"
                    rel="noopener noreferrer">{label}: {url}</a
                  >
                </div>{/if}{/each}
          {/if}
        </div>
        <footer class="publication-actions">
          <button
            id="publication-authorize-youtube"
            class="quiet"
            type="button"
            hidden={p.youtubeReady}>Authorize YouTube</button
          >
          <button id="publication-check" class="quiet" type="button"
            >Check accounts</button
          >
          <button
            id="publication-retry"
            class="quiet"
            type="button"
            hidden={!["failed", "blocked", "canceled"].includes(p.job?.state)}
            >Retry</button
          >
          <button
            id="publication-cancel"
            class="quiet danger"
            type="button"
            hidden={!p.job ||
              ["succeeded", "failed", "blocked", "canceled"].includes(
                p.job.state,
              )}>Cancel</button
          >
          <button
            id="publication-resolve"
            class="quiet"
            type="button"
            hidden={!(
              p.job?.state === "blocked" &&
              p.job?.progress?.phase === "youtube_uncertain"
            )}>Resolve upload</button
          >
          <button
            id="publication-cleanup"
            class="quiet"
            type="button"
            hidden={!p.job ||
              !["succeeded", "failed", "blocked", "canceled"].includes(
                p.job.state,
              )}>Clean local staging</button
          >
          <button
            id="publication-submit"
            class="primary"
            type="submit"
            disabled={p.submitDisabled}>Publish to both</button
          >
        </footer>
      </form>
    </div>
  </div>
</dialog>
