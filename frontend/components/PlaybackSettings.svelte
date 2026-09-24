<script lang="ts">
  import {
    applyStopConditionSuggestion,
    stopConditionHighlightSegments,
    stopConditionPopoverPlacement,
    stopConditionSuggestions,
  } from "../../src/gradlab/web_player/playback-settings.js";
  import { text } from "../../src/gradlab/web_player/panels/shared.js";
  // Wire snapshots are owned and validated by the existing Playback controller.
  // Retain their identity; never proxy the observation or history buffers.
  let { services, idPrefix = "playback" } = $props();
  let snapshot = $state.raw<any>(null);
  let hasControl = $state(false);
  let fps = $state(0),
    seed = $state(""),
    sampling = $state("stochastic"),
    temperature = $state(1),
    contractMode = $state("training");
  let stopSource = $state("");
  let stopSuggestions = $state<any[]>([]);
  let suggestionRange = $state({ from: 0, to: 0 });
  let activeSuggestion = $state(0);
  let stopSuggestionStyle = $state("");
  let interactingWithSuggestions = false;
  let stopPositionFrame: number | null = null;
  let stopTimer: ReturnType<typeof setTimeout> | null = null;
  let fpsInput: HTMLInputElement,
    seedInput: HTMLInputElement,
    temperatureInput: HTMLInputElement,
    contractInput: HTMLSelectElement,
    stopInput: HTMLTextAreaElement,
    stopHighlight: HTMLPreElement;
  let defaultSeed = $state("");
  let stopSourceKey = "",
    wasAwaiting = false;
  const selectionLabel = (mode: string) =>
    ({
      stochastic: "Stochastic",
      deterministic: "Deterministic",
      epsilon_greedy: "Epsilon-greedy",
      greedy: "Greedy",
      program: "Program",
      route: "Route",
    })[mode] || String(mode || "").replaceAll("_", " ");
  let session = $derived(snapshot?.session || {});
  let selection = $derived(snapshot?.policy?.action_selection || {});
  let supportedModes: string[] = $derived(
    Array.isArray(selection.supported_modes)
      ? selection.supported_modes
      : ["stochastic", "deterministic"],
  );
  let recording = $derived(snapshot?.mode === "recording");
  let dataset = $derived(["dataset", "trajectory"].includes(snapshot?.mode));
  let contract = $derived(session.playback_contract || {});
  let contractModes: string[] = $derived(
    Array.isArray(contract.available_modes)
      ? contract.available_modes
      : ["training"],
  );
  let stopCondition = $derived(session.stop_condition || {});
  let activeStopError = $derived(
    String(stopCondition.source || "") === stopSource
      ? stopCondition.error || null
      : null,
  );
  let stopError = $derived(
    activeStopError
      ? `${activeStopError.message} · column ${Number(activeStopError.offset || 0) + 1}`
      : "",
  );
  let stopHighlights = $derived(
    stopConditionHighlightSegments(
      stopSource,
      activeStopError,
      stopCondition.symbols,
    ),
  );
  let canEditStop = $derived(
    hasControl &&
      !recording &&
      !dataset &&
      snapshot?.run_state === "paused",
  );
  export function updateControl() {
    const state = services.getState();
    hasControl = Boolean(state.hasControl);
    // Control availability follows the live head even during recorded inspection.
    const live = state.liveSnapshot || state.snapshot;
    if (live) snapshot = live;
  }
  export function render(next: any, view: { inspection?: boolean } = {}) {
    updateControl();
    next = view.inspection ? services.getState().liveSnapshot || next : next;
    if (!next) return;
    snapshot = next;
    const nextSession = next.session || {};
    const action = next.policy?.action_selection || {};
    const modes = action.supported_modes || ["stochastic", "deterministic"];
    if (document.activeElement !== fpsInput)
      fps = Number(nextSession.target_fps || 0);
    if (document.activeElement !== temperatureInput)
      temperature = nextSession.sampling_temperature ?? 1;
    const nextSeed = text(nextSession.default_seed, nextSession.seed);
    if (document.activeElement !== seedInput && nextSeed !== defaultSeed) {
      seed = String(nextSeed);
      defaultSeed = nextSeed;
    }
    if (!nextSession.awaiting_next_episode || !wasAwaiting)
      sampling =
        action.requested_mode ||
        nextSession.sampling_mode ||
        action.default_mode ||
        modes[0] ||
        "";
    wasAwaiting = Boolean(nextSession.awaiting_next_episode);
    if (document.activeElement !== contractInput)
      contractMode = nextSession.playback_contract?.mode || "training";
    const nextStopSource = String(nextSession.stop_condition?.source || "");
    if (document.activeElement !== stopInput && nextStopSource !== stopSourceKey) {
      stopSource = nextStopSource;
      stopSourceKey = nextStopSource;
    }
  }
  export const episodeOptions = () => ({
    seed,
    sampling_mode: sampling,
  });
  const refreshSuggestions = () => {
    if (!stopInput) return;
    const result = stopConditionSuggestions(
      stopSource,
      stopInput.selectionStart,
      stopCondition.symbols,
    );
    suggestionRange = { from: result.from, to: result.to };
    stopSuggestions = result.items;
    activeSuggestion = 0;
    positionStopSuggestions();
  };
  const positionStopSuggestions = () => {
    if (!stopInput) return;
    const placement = stopConditionPopoverPlacement(
      stopInput.getBoundingClientRect(),
      { width: window.innerWidth, height: window.innerHeight },
    );
    stopSuggestionStyle = [
      `left:${placement.left}px`,
      `top:${placement.top}px`,
      `width:${placement.width}px`,
      `max-height:${placement.maxHeight}px`,
    ].join(";");
  };
  const scheduleStopSuggestionPosition = () => {
    if (stopPositionFrame != null) cancelAnimationFrame(stopPositionFrame);
    stopPositionFrame = requestAnimationFrame(() => {
      stopPositionFrame = null;
      positionStopSuggestions();
    });
  };
  const syncStopHighlightScroll = () => {
    if (!stopInput || !stopHighlight) return;
    stopHighlight.scrollTop = stopInput.scrollTop;
    stopHighlight.scrollLeft = stopInput.scrollLeft;
  };
  const portalToBody = (node: HTMLElement) => {
    document.body.append(node);
    return { destroy: () => node.remove() };
  };
  const sendStopSource = () => {
    if (stopTimer) clearTimeout(stopTimer);
    stopTimer = setTimeout(() => {
      services.command("set_stop_condition", { source: stopSource });
      stopTimer = null;
    }, 250);
  };
  const flushStopSource = () => {
    if (!stopTimer) return;
    clearTimeout(stopTimer);
    stopTimer = null;
    services.command("set_stop_condition", { source: stopSource });
  };
  const chooseSuggestion = (index: number) => {
    const item = stopSuggestions[index];
    if (!item) return;
    const result = applyStopConditionSuggestion(
      stopSource,
      stopInput.selectionStart,
      suggestionRange.from,
      suggestionRange.to,
      item.value,
    );
    stopSource = result.source;
    stopSuggestions = [];
    sendStopSource();
    requestAnimationFrame(() => {
      stopInput.focus();
      stopInput.setSelectionRange(result.cursor, result.cursor);
    });
  };
  const leaveStopInput = () => {
    requestAnimationFrame(() => {
      if (interactingWithSuggestions) {
        stopInput?.focus({ preventScroll: true });
        return;
      }
      stopSuggestions = [];
      flushStopSource();
    });
  };
  $effect(() => {
    if (!stopSuggestions.length) return;
    scheduleStopSuggestionPosition();
    const reposition = () => scheduleStopSuggestionPosition();
    const finishInteraction = () => {
      interactingWithSuggestions = false;
    };
    window.addEventListener("resize", reposition);
    window.addEventListener("pointerup", finishInteraction);
    document.addEventListener("scroll", reposition, true);
    return () => {
      window.removeEventListener("resize", reposition);
      window.removeEventListener("pointerup", finishInteraction);
      document.removeEventListener("scroll", reposition, true);
    };
  });
  $effect(() => () => {
    if (stopTimer) clearTimeout(stopTimer);
    if (stopPositionFrame != null) cancelAnimationFrame(stopPositionFrame);
  });
</script>

<div class="control-components playback-settings-form">
  <div class="advanced-playback-body">
    <div class="playback-field playback-fps">
      <label for={`${idPrefix}-fps`}>Play FPS</label>
      <input
        id={`${idPrefix}-fps`}
        data-fps
        type="number"
        min={recording ? 1 : 0}
        step="1"
        bind:value={fps}
        bind:this={fpsInput}
        inputmode="decimal"
        disabled={!hasControl}
        oninput={(event) => {
          if (
            event.currentTarget.validity.valid &&
            event.currentTarget.value.trim()
          )
            services.command("set_fps", {
              fps: Number(event.currentTarget.value),
            });
        }}
      />
    </div>
    <div class="playback-field next-episode-seed">
      <label for={`${idPrefix}-seed`}>Reset seed</label><input
        id={`${idPrefix}-seed`}
        data-seed
        inputmode="numeric"
        bind:value={seed}
        bind:this={seedInput}
        disabled={!hasControl || recording || dataset}
      />
    </div>
    <div class="playback-field playback-sampling">
      <label for={`${idPrefix}-sampling`}>Next action selection</label>
      <select
        id={`${idPrefix}-sampling`}
        data-sampling
        aria-describedby={`${idPrefix}-sampling-hint`}
        bind:value={sampling}
        disabled={!hasControl ||
          recording ||
          dataset ||
          supportedModes.length <= 1}
        onchange={(event) =>
          services.command("set_action_selection_mode", {
            mode: event.currentTarget.value,
          })}
      >
        {#each supportedModes as mode (mode)}<option value={mode}
            >{selectionLabel(mode)}</option
          >{/each}
      </select>
    </div>
    <div
      class="playback-field"
      data-temperature-field
      hidden={selection.supports_temperature === false ||
        !supportedModes.includes("stochastic")}
    >
      <label for={`${idPrefix}-temperature`}>Sampling temperature</label>
      <input
        id={`${idPrefix}-temperature`}
        data-temperature
        type="number"
        min="0.01"
        step="any"
        bind:value={temperature}
        bind:this={temperatureInput}
        disabled={!hasControl ||
          recording ||
          dataset ||
          sampling !== "stochastic"}
        onchange={(event) => {
          if (
            event.currentTarget.validity.valid &&
            event.currentTarget.value.trim()
          )
            services.command("set_sampling_temperature", {
              temperature: Number(event.currentTarget.value),
            });
        }}
      />
    </div>
    <p
      id={`${idPrefix}-sampling-hint`}
      class="control-hint"
      data-sampling-hint
      hidden={recording || dataset || supportedModes.length !== 1}
    >
      {supportedModes.length === 1
        ? `${selectionLabel(supportedModes[0])} is fixed by this checkpoint.`
        : ""}
    </p>
    <div
      class="playback-field playback-contract"
      data-playback-contract
      hidden={recording || dataset}
    >
      <label for={`${idPrefix}-contract-mode`}>Environment contract</label>
      <select
        id={`${idPrefix}-contract-mode`}
        data-contract-mode
        bind:value={contractMode}
        bind:this={contractInput}
        disabled={!hasControl || recording || dataset}
        onchange={(event) =>
          services.command("set_contract_mode", {
            mode: event.currentTarget.value,
          })}
      >
        <option value="training" disabled={!contractModes.includes("training")}
          >Training contract</option
        >
        <option
          value="evaluation"
          disabled={!contractModes.includes("evaluation")}
          >Published evaluation</option
        >
        <option
          value="counterfactual"
          disabled={!contractModes.includes("counterfactual")}
          >Counterfactual overrides</option
        >
      </select>
    </div>
    <div class="playback-field stop-condition-editor" hidden={recording || dataset}>
      <label for={`${idPrefix}-stop-condition`}>Stop conditions</label>
      <div
        class="stop-condition-input-shell"
        class:invalid={activeStopError != null}
      >
        <pre
          class="stop-condition-highlight"
          aria-hidden="true"
          bind:this={stopHighlight}
        >{#each stopHighlights as segment}<span
              class={`stop-condition-token ${segment.kind}`}
              class:syntax-error={segment.error}
              data-token-kind={segment.kind}
            >{segment.text}</span>{/each}</pre
        >
        <textarea
          id={`${idPrefix}-stop-condition`}
          data-stop-condition
          rows="4"
          spellcheck="false"
          role="combobox"
          aria-autocomplete="list"
          aria-controls={`${idPrefix}-stop-suggestions`}
          aria-expanded={stopSuggestions.length > 0}
          aria-activedescendant={stopSuggestions.length
            ? `${idPrefix}-stop-suggestion-${activeSuggestion}`
            : undefined}
          aria-invalid={activeStopError != null}
          aria-describedby={stopError
            ? `${idPrefix}-stop-condition-error`
            : undefined}
          bind:this={stopInput}
          bind:value={stopSource}
          disabled={!canEditStop}
          onfocus={refreshSuggestions}
          onblur={leaveStopInput}
          onclick={refreshSuggestions}
          onscroll={syncStopHighlightScroll}
          oninput={() => {
            refreshSuggestions();
            syncStopHighlightScroll();
            sendStopSource();
          }}
          onkeydown={(event) => {
            if (!stopSuggestions.length) return;
            if (event.key === "ArrowDown") {
              event.preventDefault();
              activeSuggestion = (activeSuggestion + 1) % stopSuggestions.length;
            } else if (event.key === "ArrowUp") {
              event.preventDefault();
              activeSuggestion =
                (activeSuggestion - 1 + stopSuggestions.length) % stopSuggestions.length;
            } else if (event.key === "Enter" || event.key === "Tab") {
              event.preventDefault();
              chooseSuggestion(activeSuggestion);
            } else if (event.key === "Escape") {
              event.stopPropagation();
              stopSuggestions = [];
            }
          }}
        ></textarea>
      </div>
      {#if stopSuggestions.length}
        <div
          id={`${idPrefix}-stop-suggestions`}
          class="stop-condition-suggestions"
          data-playback-settings-popover
          role="listbox"
          tabindex="-1"
          style={stopSuggestionStyle}
          use:portalToBody
          onpointerdown={(event) => {
            event.stopPropagation();
            interactingWithSuggestions = true;
          }}
        >
          {#each stopSuggestions as suggestion, index (suggestion.value)}
            <button
              id={`${idPrefix}-stop-suggestion-${index}`}
              type="button"
              role="option"
              tabindex="-1"
              aria-selected={index === activeSuggestion}
              class:active={index === activeSuggestion}
              onmousedown={(event) => event.preventDefault()}
              onmouseenter={() => (activeSuggestion = index)}
              onclick={(event) => {
                event.stopPropagation();
                chooseSuggestion(index);
              }}
            >
              <code>{suggestion.value}</code>
              <span>{suggestion.kind}{suggestion.detail == null
                  ? ""
                  : ` · ${suggestion.detail}`}</span>
            </button>
          {/each}
        </div>
      {/if}
      {#if stopError}
        <p
          id={`${idPrefix}-stop-condition-error`}
          class="control-hint control-error"
          data-stop-condition-status
        >
          {stopError}
        </p>
      {/if}
    </div>
  </div>
</div>
