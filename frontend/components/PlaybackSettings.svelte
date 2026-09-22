<script lang="ts">
  import {
    orderedTerminationConditions,
    terminationOutcomeClass,
    frameSkipPresentation,
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
  let enabledConditions = $state<string[]>([]);
  let fpsInput: HTMLInputElement,
    seedInput: HTMLInputElement,
    temperatureInput: HTMLInputElement,
    contractInput: HTMLSelectElement;
  let defaultSeed = $state("");
  let terminationKey = "",
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
  let conditions = $derived(
    orderedTerminationConditions(session.termination_conditions),
  );
  let canTerminate = $derived(
    hasControl &&
      !recording &&
      !dataset &&
      (Number(session.step || 0) === 0 ||
        Boolean(session.awaiting_next_episode)),
  );
  let frameSkip = $derived(frameSkipPresentation(contract));
  let contractLabel = $derived(
    session.temperature_changed ||
      Number(session.sampling_temperature ?? 1) !== 1
      ? "Counterfactual — not evidence"
      : {
          training: "Training contract",
          evaluation: "Published evaluation",
          counterfactual: "Counterfactual — not evidence",
        }[String(contract.mode || "training")] || selectionLabel(contract.mode),
  );
  let contractHint = $derived.by(() => {
    const messages = [];
    if (session.critic_comparison?.reasons?.length)
      messages.push(
        `Critic comparison unavailable: ${session.critic_comparison.reasons.join("; ")}.`,
      );
    if (contract.evaluation_matches_training === false)
      messages.push(
        `Published evaluation semantics differ from training${contract.mismatch_paths?.length ? ` at ${contract.mismatch_paths.join(", ")}` : ""}.`,
      );
    return (
      messages.join(" ") ||
      "Training-compatible critic comparison is available after a terminal episode."
    );
  });
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
    const key = JSON.stringify(
      orderedTerminationConditions(nextSession.termination_conditions),
    );
    if (key !== terminationKey) {
      terminationKey = key;
      enabledConditions = (nextSession.termination_conditions || [])
        .filter((condition: any) => condition.enabled)
        .map((condition: any) => condition.id);
    }
  }
  export const episodeOptions = () => ({
    seed,
    sampling_mode: sampling,
    enabled_termination_conditions: conditions.length
      ? [...enabledConditions]
      : null,
  });
</script>

<div class="control-components playback-settings-form">
  <div class="playback-glance" aria-live="polite">
    <strong data-playback-glance-contract>{contractLabel}</strong>
    <span data-playback-glance-detail
      >{selectionLabel(sampling || session.sampling_mode)} · seed {text(
        snapshot?.transition?.seed,
        text(session.seed, defaultSeed),
      )}</span
    >
    <span
      data-playback-frame-skip
      hidden={!frameSkip}
      class:contract-mismatch={frameSkip?.differs}
      title={frameSkip
        ? `Training repeated each selected action for ${frameSkip.training} environment frame${frameSkip.training === 1 ? "" : "s"}. This playback repeats each selected action for ${frameSkip.playback} environment frame${frameSkip.playback === 1 ? "" : "s"}.`
        : ""}>{frameSkip?.label || ""}</span
    >
  </div>
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
      <p class="control-hint">
        1 uses the original distribution. Lower values favor likely actions;
        higher values add randomness. Changes apply to the next stochastic
        decision and make playback counterfactual.
      </p>
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
        aria-describedby={`${idPrefix}-contract-hint`}
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
    <p id={`${idPrefix}-contract-hint`} class="control-hint" data-contract-hint>
      {contractHint}
    </p>
    <fieldset
      class="termination-settings"
      data-termination-settings
      hidden={recording || dataset || !conditions.length}
    >
      <legend>Episode termination</legend>
      <p class="control-hint" data-termination-source>
        Defaults: {session.termination_source || "training"}
      </p>
      <div class="termination-options" data-termination-options>
        {#each conditions as condition (condition.id)}
          <label class="termination-option"
            ><input
              type="checkbox"
              value={condition.id}
              bind:group={enabledConditions}
              disabled={!canTerminate}
              onchange={(event) => {
                const enabled = new Set(enabledConditions);
                if (event.currentTarget.checked) enabled.add(condition.id);
                else enabled.delete(condition.id);
                services.command("set_termination_conditions", {
                  enabled: [...enabled],
                });
              }}
            /><span>{condition.label}</span><small
              class={`termination-outcome ${terminationOutcomeClass(condition.outcome)}`}
              >{condition.outcome}</small
            ></label
          >
        {/each}
      </div>
      <p class="control-hint">
        Changes apply immediately before an episode starts.
      </p>
    </fieldset>
  </div>
</div>
