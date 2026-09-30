<script lang="ts">
  import PublicationDialog from "./PublicationDialog.svelte";
  import { shellState } from "../shell-state.svelte";
  import {
    eventColorFill,
    eventLabels,
  } from "../../src/gradlab/web_player/event-colors.js";
</script>

<header class="app-header">
  <div class="app-brand">
    <span class="app-wordmark eyebrow">GRADLAB</span>
    <span class="header-divider" aria-hidden="true"></span>
    <button
      id="source-back"
      class="quiet button-with-icon header-back"
      type="button"
      aria-label="Back"
      title="Back to previous level"
      hidden
      ><svg class="icon" aria-hidden="true"
        ><use href="/assets/tabler-icons.svg#ti-arrow-left"></use></svg
      ><span>Back</span></button
    >
    <h1 id="page-title" hidden>{shellState.pageTitle}</h1>
    <nav
      id="source-breadcrumbs"
      class="source-breadcrumbs header-breadcrumbs"
      aria-label="Playback source"
      hidden
    ></nav>
  </div>
  <div class="header-status">
    <span class="hotreload-warning" role="status" title="Live frontend changes update this player automatically">Hot reload active</span>
    <button
      id="switch-window"
      class="quiet button-with-icon"
      type="button"
      title={`Open or focus ${shellState.switchWindow.label.toLowerCase()}`}
      hidden={!shellState.switchWindow.visible}
      ><svg class="icon" aria-hidden="true"
        ><use href="/assets/tabler-icons.svg#ti-external-link"></use></svg
      ><span>{shellState.switchWindow.label}</span></button
    >
    <button
      id="trajectory-download"
      class="quiet icon-only"
      type="button"
      aria-label="Download episode"
      title="Download episode"
      hidden
      ><svg
        class="icon"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
        aria-hidden="true"><path d="M12 3v12m-5-5 5 5 5-5M5 16v4h14v-4" /></svg
      ></button
    >
    <button
      id="trajectory-import"
      class="quiet icon-only"
      type="button"
      aria-label="Import episode"
      title="Import episode"
      ><svg
        class="icon"
        aria-hidden="true"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
        ><path d="M4 17v4h16v-4M12 3v12M7 8l5-5 5 5" /></svg
      ></button
    >
    <input
      id="trajectory-file"
      type="file"
      accept=".trj,.zip"
      aria-label="Episode trajectory archive"
      hidden
    />
    <nav
      id="checkpoint-navigation"
      class="checkpoint-navigation"
      aria-label="Checkpoint navigation"
      hidden
    >
      <button
        class="quiet checkpoint-navigation-button icon-only"
        type="button"
        aria-label="Previous checkpoint"
        data-checkpoint-previous
        ><svg class="icon" aria-hidden="true"
          ><use href="/assets/tabler-icons.svg#ti-arrow-left"></use></svg
        ></button
      >
      <span
        class="checkpoint-navigation-position"
        data-checkpoint-position
        aria-label="Checkpoint position"
        aria-live="polite">— / —</span
      >
      <button
        class="quiet checkpoint-navigation-button icon-only"
        type="button"
        aria-label="Next checkpoint"
        data-checkpoint-next
        ><svg class="icon" aria-hidden="true"
          ><use href="/assets/tabler-icons.svg#ti-arrow-right"></use></svg
        ></button
      >
    </nav>
    <span
      id="connection-status"
      class={`sync-status ${shellState.connection.kind}`}
      aria-live="polite"
      hidden={shellState.connection.label === "Synced" &&
        !shellState.connection.kind}>{shellState.connection.label}</span
    >
    <button
      id="more-toggle"
      class="quiet icon-only"
      type="button"
      aria-label="More playback actions"
      title="More playback actions"
      aria-controls="player-menu"
      aria-expanded="false"
      hidden
      ><svg class="icon" aria-hidden="true"
        ><use href="/assets/tabler-icons.svg#ti-dots-vertical"></use></svg
      ></button
    >
  </div>
</header>

<div
  id="checkpoint-loading-mask"
  class="checkpoint-loading-mask"
  role="status"
  aria-live="polite"
  hidden
>
  <div class="checkpoint-loading-status">
    <span class="spinner" aria-hidden="true"></span>
    <strong>Loading checkpoint…</strong>
  </div>
</div>

<div id="player-menu" class="floating-menu player-menu" hidden>
  <button
    id="inspect-active"
    class="quiet button-with-icon"
    type="button"
    title="Inspect active goal and recipe YAML"
    hidden
    ><svg class="icon" aria-hidden="true"
      ><use href="/assets/tabler-icons.svg#ti-code"></use></svg
    ><span>Goal & recipe YAML</span></button
  >
  <button
    id="publish-episode"
    class="quiet button-with-icon"
    type="button"
    title="Publish the completed episode to YouTube and Hugging Face"

     hidden={!shellState.publication.enabled}><svg class="icon" aria-hidden="true"
      ><use href="/assets/tabler-icons.svg#ti-external-link"></use></svg
    ><span>Publish episode</span></button
  >
  <div class="player-menu-customize" data-customize-actions>
    <span class="menu-section-label">Customize</span>
    <button
      id="layouts-toggle"
      class="quiet button-with-icon"
      type="button"
      title="Saved layouts"
      ><svg class="icon" aria-hidden="true"
        ><use href="/assets/tabler-icons.svg#ti-layout-dashboard"></use></svg
      ><span>Saved layouts</span></button
    >
    <button
      id="panels-toggle"
      class="quiet button-with-icon"
      type="button"
      title="Panels"
      aria-controls="panel-shelf"
      aria-expanded="false"
      ><svg class="icon" aria-hidden="true"
        ><use href="/assets/tabler-icons.svg#ti-layout-grid"></use></svg
      ><span>Panels</span></button
    >
    <button
      id="new-window"
      class="quiet button-with-icon"
      type="button"
      title="Open synchronized window"
      ><svg class="icon" aria-hidden="true"
        ><use href="/assets/tabler-icons.svg#ti-external-link"></use></svg
      ><span>Open synchronized window</span></button
    >
  </div>
</div>

<div id="layout-menu" class="floating-menu layout-menu" hidden>
  <label
    >Layout name <input
      id="layout-name-input"
      value="Default layout"
      maxlength="48"
    /></label
  >
  <div class="menu-actions">
    <button
      id="save-layout"
      class="primary button-with-icon"
      type="button"
      title="Save layout"
      ><svg class="icon" aria-hidden="true"
        ><use href="/assets/tabler-icons.svg#ti-device-floppy"></use></svg
      ><span>Save layout</span></button
    >
    <button
      id="reset-layout"
      class="quiet button-with-icon"
      type="button"
      title="Reset default layout"
      ><svg class="icon" aria-hidden="true"
        ><use href="/assets/tabler-icons.svg#ti-restore"></use></svg
      ><span>Reset default</span></button
    >
  </div>
  <div id="saved-layouts" class="saved-layouts">
    {#each shellState.savedLayouts as name (name)}<div class="saved-layout-row">
        <button
          type="button"
          class="quiet"
          title={`Load layout ${name}`}
          onclick={() => shellState.loadLayout(name)}>{name}</button
        ><button
          type="button"
          class="quiet danger"
          title={`Delete layout ${name}`}
          onclick={() => shellState.deleteLayout(name)}>Delete</button
        >
      </div>
    {:else}<span class="empty-state">No named layouts saved yet.</span>{/each}
  </div>
</div>

<div id="panel-menu" class="floating-menu panel-menu" hidden>
  <strong id="panel-menu-title">{shellState.panelMenu.title}</strong>
  <button
    id="panel-new-window"
    class="button-with-icon"
    type="button"
    title="Move panel to a new synchronized window"
    ><svg class="icon" aria-hidden="true"
      ><use href="/assets/tabler-icons.svg#ti-external-link"></use></svg
    ><span>Move to new window</span></button
  >
  <button
    id="panel-dock-main"
    class="button-with-icon"
    type="button"
    title="Dock panel in the main window"
     hidden={!shellState.panelMenu.dockMain}><svg class="icon" aria-hidden="true"
      ><use href="/assets/tabler-icons.svg#ti-layout-dashboard"></use></svg
    ><span>Dock to main window</span></button
  >
  <button
    id="panel-edit"
    class="button-with-icon"
    type="button"
    title="Edit telemetry visualizations"
     hidden={!shellState.panelMenu.telemetry}><svg class="icon" aria-hidden="true"
      ><use href="/assets/tabler-icons.svg#ti-edit"></use></svg
    ><span>Edit</span></button
  >
  <button
    id="panel-duplicate"
    class="button-with-icon"
    type="button"
    title="Duplicate as a telemetry panel"
     hidden={!shellState.panelMenu.telemetry}><svg class="icon" aria-hidden="true"
      ><use href="/assets/tabler-icons.svg#ti-copy"></use></svg
    ><span>Duplicate</span></button
  >
  <button
    id="panel-hide"
    class="button-with-icon"
    type="button"
    title="Hide panel"
    ><svg class="icon" aria-hidden="true"
      ><use href="/assets/tabler-icons.svg#ti-eye-off"></use></svg
    ><span>Hide panel</span></button
  >
  <button
    id="panel-reset-size"
    class="button-with-icon"
    type="button"
    title="Reset panel size"
    ><svg class="icon" aria-hidden="true"
      ><use href="/assets/tabler-icons.svg#ti-restore"></use></svg
    ><span>Reset size</span></button
  >
  <button
    id="panel-remove"
    class="button-with-icon danger"
    type="button"
    title="Remove custom panel"
     hidden={!shellState.panelMenu.removable}><svg class="icon" aria-hidden="true"
      ><use href="/assets/tabler-icons.svg#ti-trash"></use></svg
    ><span>Remove</span></button
  >
</div>

<div
  id="panel-shelf"
  class="floating-menu panel-shelf"
  hidden
  aria-label="Available panels"
>
  <div class="panel-shelf-heading">
    <strong>Panels</strong>
    <button
      id="panel-add"
      class="quiet button-with-icon"
      type="button"
      title="Add telemetry panel"
      ><svg class="icon" aria-hidden="true"
        ><use href="/assets/tabler-icons.svg#ti-plus"></use></svg
      ><span>Add</span></button
    >
  </div>
  <div id="panel-shelf-items" class="panel-shelf-items"></div>
</div>

<main id="source-browser" class="source-browser" hidden></main>
<section
  id="trajectory-controls"
  class="trajectory-controls"
  aria-label="Episode recording"
  hidden
>
  <button id="trajectory-retry" class="quiet" type="button" hidden
    >Retry storage</button
  >
  <span id="trajectory-navigation" hidden>
    <button
      id="trajectory-previous"
      class="quiet icon-only"
      type="button"
      aria-label="Previous recorded transition"
      ><svg class="icon" aria-hidden="true"
        ><use href="/assets/tabler-icons.svg#ti-arrow-left"></use></svg
      ></button
    >
    <label
      >Transition <input
        id="trajectory-seek"
        type="number"
        min="1"
        step="1"
        aria-label="Exact recorded transition"
      /></label
    >
    <button
      id="trajectory-next"
      class="quiet icon-only"
      type="button"
      aria-label="Next recorded transition"
      ><svg class="icon" aria-hidden="true"
        ><use href="/assets/tabler-icons.svg#ti-arrow-right"></use></svg
      ></button
    >
  </span>
</section>
<span id="trajectory-status" class="visually-hidden" role="status"></span>
<main id="dashboard" class="dashboard grid-stack"></main>

<div id="timeline-home" hidden></div>
<section
  id="timeline"
  class="timeline"
  aria-label="Episode inspection"
  aria-busy={shellState.timeline.busy}
>
  <div class="timeline-labels">
    <strong id="timeline-label">{shellState.timeline.label}</strong>
    <button
      id="timeline-zoom"
      type="button"
      aria-label="Reset chart zoom"
      hidden={!shellState.timeline.zoom}
      >{shellState.timeline.zoom
        ? `Steps ${shellState.timeline.zoom.first}–${shellState.timeline.zoom.last} · Reset zoom`
        : ""}</button
    >
    <span
      id="playback-evidence-status"
      class="playback-evidence-status"
      role="status"
      hidden={!shellState.evidence.text}
      title={shellState.evidence.title}>{shellState.evidence.text}</span
    >
  </div>
  <div class="timeline-track">
    <div class="timeline-scrubber-track">
      <input
        id="timeline-scrubber"
        type="range"
        min={shellState.timeline.first}
        max={shellState.timeline.last}
        step="1"
        bind:value={shellState.timeline.selected}
        disabled={shellState.timeline.disabled}
        aria-valuetext={`Step ${shellState.timeline.selected}`}
        style:--timeline-progress={`${shellState.timeline.progress}%`}
        aria-label="Inspect an episode step"
        title="Left/Right: move one step · Space: play or pause"
      />
      <div
        id="timeline-zoom-band"
        hidden={!shellState.timeline.zoom}
        style:left={`${(100 * ((shellState.timeline.zoom?.first ?? 0) - shellState.timeline.first)) / Math.max(1, shellState.timeline.last - shellState.timeline.first)}%`}
        style:width={`${(100 * ((shellState.timeline.zoom?.last ?? 0) - (shellState.timeline.zoom?.first ?? 0))) / Math.max(1, shellState.timeline.last - shellState.timeline.first)}%`}
      >
        <button
          type="button"
          class="timeline-range-handle"
          data-edge="first"
          role="slider"
          aria-valuenow={shellState.timeline.zoom?.first ?? 0}
          aria-valuetext={`Step ${shellState.timeline.zoom?.first ?? 0}`}
          aria-valuemin={shellState.timeline.first}
          aria-valuemax={(shellState.timeline.zoom?.last ?? 0) - 1}
          aria-label="Chart window start"
          title="Drag to adjust chart window start"
        ></button>
        <button
          type="button"
          class="timeline-range-handle"
          data-edge="last"
          role="slider"
          aria-valuenow={shellState.timeline.zoom?.last ?? 0}
          aria-valuetext={`Step ${shellState.timeline.zoom?.last ?? 0}`}
          aria-valuemin={(shellState.timeline.zoom?.first ?? 0) + 1}
          aria-valuemax={shellState.timeline.last}
          aria-label="Chart window end"
          title="Drag to adjust chart window end"
        ></button>
      </div>
      <div id="timeline-markers" class="timeline-markers" aria-hidden="true">
        {#each shellState.timeline.markers as marker (marker.step)}<span
            class="timeline-marker"
            style:left={`${marker.position * 100}%`}
            data-step={marker.step}
            data-count={marker.count}
            style:--event-colors={eventColorFill(eventLabels(marker))}
          ></span>{/each}
      </div>
    </div>
    <div class="timeline-actions">
      <button
        id="timeline-playback-toggle"
        class="icon-only"
        class:primary={shellState.transport.action !== "pause"}
        type="button"
        data-action={shellState.transport.action}
        disabled={shellState.transport.disabled}
        aria-label={shellState.transport.label}
        title={shellState.transport.reason}
        ><svg class="icon" aria-hidden="true"
          ><use
            id="timeline-playback-icon"
            href={`/assets/tabler-icons.svg#ti-${shellState.transport.icon}`}
          ></use></svg
        ></button
      >
      <button
        id="timeline-reset"
        class="quiet icon-only"
        type="button"
        aria-label="Reset episode"
        title={shellState.transport.resetTitle}
        disabled={shellState.transport.resetDisabled}
        ><svg class="icon" aria-hidden="true"
          ><use href="/assets/tabler-icons.svg#ti-refresh"></use></svg
        ></button
      >
      <button
        id="playback-settings-toggle"
        class="quiet icon-only"
        type="button"
        aria-label="Playback settings"
        title="Playback settings"
        aria-controls="playback-settings-menu"
        aria-expanded="false"
        ><svg class="icon" aria-hidden="true"
          ><use href="/assets/tabler-icons.svg#ti-settings"></use></svg
        ></button
      >
    </div>
  </div>
</section>

<div
  id="playback-settings-menu"
  class="floating-menu playback-settings-menu"
  hidden
>
  <div class="floating-menu-heading">
    <strong>Playback settings</strong>
    <button
      id="playback-settings-close"
      class="quiet icon-only"
      type="button"
      aria-label="Close playback settings"
      title="Close"
      ><svg class="icon" aria-hidden="true"
        ><use href="/assets/tabler-icons.svg#ti-x"></use></svg
      ></button
    >
  </div>
  <div id="playback-settings-content"></div>
</div>

<dialog
  id="trajectory-download-dialog"
  class="trajectory-download-dialog"
  aria-labelledby="trajectory-download-heading"
>
  <h2 id="trajectory-download-heading">Download episode</h2>
  <p>
    Save a <strong
      >.trj archive named by environment, checkpoint, and content hash</strong
    >, containing the recorded episode, images, policy inputs and decisions, and
    the exact checkpoint with its configuration.
  </p>
  <p>
    The current episode is captured when you confirm. Playback can continue
    during preparation. Unfinished episodes include only what has been recorded
    so far.
  </p>
  <div id="trajectory-download-progress" hidden>
    <p role="status">
      Preparing your episode… Your download will start automatically.
    </p>
    <progress aria-label="Preparing episode download"></progress>
  </div>
  <p id="trajectory-download-error" role="alert" hidden></p>
  <div class="trajectory-download-actions">
    <button id="trajectory-download-cancel" class="quiet" type="button"
      >Cancel</button
    >
    <button id="trajectory-download-confirm" type="button">Confirm</button>
  </div>
</dialog>

<div id="panel-editor-mount"></div>

<dialog
  id="contract-viewer"
  class="contract-viewer"
  aria-labelledby="contract-viewer-heading"
>
</dialog>

<PublicationDialog />

<div
  id="toast"
  class="toast"
  class:visible={shellState.toast.visible}
  style:border-color={shellState.toast.error ? "var(--red)" : "var(--cyan)"}
  role="status"
  aria-live="polite"
>
  {shellState.toast.message}
</div>
