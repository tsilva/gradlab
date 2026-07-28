import { createPanel, text } from "./shared.js";

export function mount({ definition, services }) {
  const element = createPanel({
    id: definition.id,
    label: definition.label,
    tag: "aside",
    className: "control-panel transport",
    headerClass: "control-panel-header",
    body: `
      <p data-session-summary class="session-summary" aria-live="polite">Waiting for session…</p>
      <section class="control-section" aria-labelledby="playback-controls-heading">
        <h3 id="playback-controls-heading" class="control-label">Playback</h3>
        <div class="control-grid transport-grid">
          <button data-command="play" data-playback-toggle data-requires-active-episode class="primary icon-only" aria-label="Play" title="Play current episode"><svg class="icon" aria-hidden="true"><use data-playback-icon href="/assets/tabler-icons.svg#ti-player-play"></use></svg></button>
          <button data-command="step" data-requires-active-episode class="icon-only" aria-label="Step once" title="Step once"><svg class="icon" aria-hidden="true"><use href="/assets/tabler-icons.svg#ti-player-skip-forward"></use></svg></button>
          <button data-command="step-ten" data-requires-active-episode class="icon-only" aria-label="Step 10 times" title="Step 10 times"><svg class="icon" aria-hidden="true"><use href="/assets/tabler-icons.svg#ti-player-track-next"></use></svg></button>
          <button data-command="continue-event" data-requires-active-episode class="icon-only" aria-label="Continue to next event" title="Continue to next event"><svg class="icon" aria-hidden="true"><use href="/assets/tabler-icons.svg#ti-activity-heartbeat"></use></svg></button>
        </div>
        <details class="playback-settings">
          <summary>Playback settings</summary>
          <div class="playback-settings-body">
            <div class="playback-fps">
              <label for="playback-fps">Play FPS</label>
              <input id="playback-fps" data-fps type="number" min="0" step="1" value="0" inputmode="decimal" aria-describedby="playback-fps-hint">
            </div>
            <p id="playback-fps-hint" class="control-hint">0 runs playback uncapped · changes apply immediately</p>
            <div class="playback-contract" data-playback-contract>
              <label for="playback-contract-mode">Environment contract</label>
              <select id="playback-contract-mode" data-contract-mode aria-describedby="playback-contract-hint">
                <option value="training">Training contract</option>
                <option value="evaluation">Published evaluation</option>
                <option value="counterfactual">Counterfactual · clipping off</option>
              </select>
              <button data-command="set-contract-mode" class="quiet icon-only" aria-label="Apply environment contract" title="Apply environment contract and start a new shared session"><svg class="icon" aria-hidden="true"><use href="/assets/tabler-icons.svg#ti-check"></use></svg></button>
            </div>
            <p id="playback-contract-hint" class="control-hint" data-contract-hint>Training-time policy semantics are the default.</p>
            <fieldset class="termination-settings" data-termination-settings>
              <legend>Episode termination</legend>
              <p class="control-hint" data-termination-source></p>
              <div class="termination-options" data-termination-options></div>
              <button data-command="apply-termination" data-apply-termination class="quiet control-wide" type="button">Apply and reset episode</button>
              <p class="control-hint">Changes are available before the first step or between episodes.</p>
            </fieldset>
          </div>
        </details>
      </section>
      <section class="control-section next-episode-settings" aria-labelledby="next-episode-heading">
        <h3 id="next-episode-heading" class="control-label">Episode</h3>
        <div class="next-episode-settings-body">
          <label class="next-episode-seed" for="next-episode-seed">Seed
            <input id="next-episode-seed" data-seed inputmode="numeric">
          </label>
          <div class="playback-sampling">
            <label for="playback-sampling">Action selection</label>
            <select id="playback-sampling" data-sampling aria-describedby="playback-sampling-hint">
              <option value="stochastic">Stochastic</option>
              <option value="deterministic">Deterministic</option>
            </select>
          </div>
          <p id="playback-sampling-hint" class="control-hint">Applies to the next playback episode only</p>
          <button data-command="reset-episode" data-reset-episode class="quiet button-with-icon control-wide" aria-label="Reset episode" title="Reset to the configured seed and pause"><svg class="icon" aria-hidden="true"><use href="/assets/tabler-icons.svg#ti-refresh"></use></svg><span>Reset episode</span></button>
          <button data-command="next-episode" data-next-episode class="primary button-with-icon control-wide" aria-label="Play next episode" title="Available after the current episode ends"><svg class="icon" aria-hidden="true"><use href="/assets/tabler-icons.svg#ti-player-play"></use></svg><span>Play next episode</span></button>
          <p data-next-episode-hint class="control-hint">Available after the current episode ends</p>
        </div>
      </section>
    `,
  });

  const seed = element.querySelector("[data-seed]");
  const fps = element.querySelector("[data-fps]");
  const sampling = element.querySelector("[data-sampling]");
  const contractMode = element.querySelector("[data-contract-mode]");
  const contractSettings = element.querySelector("[data-playback-contract]");
  const contractHint = element.querySelector("[data-contract-hint]");
  const playbackToggle = element.querySelector("[data-playback-toggle]");
  const playbackIcon = playbackToggle.querySelector("[data-playback-icon]");
  const resetEpisode = element.querySelector("[data-reset-episode]");
  const nextEpisode = element.querySelector("[data-next-episode]");
  const nextEpisodeSettings = element.querySelector(".next-episode-settings");
  const nextEpisodeHint = element.querySelector("[data-next-episode-hint]");
  const seedField = element.querySelector(".next-episode-seed");
  const playbackSampling = element.querySelector(".playback-sampling");
  const terminationSettings = element.querySelector("[data-termination-settings]");
  const terminationOptions = element.querySelector("[data-termination-options]");
  const terminationSource = element.querySelector("[data-termination-source]");
  const applyTermination = element.querySelector("[data-apply-termination]");
  let wasAwaitingNextEpisode = false;
  const selectionLabel = (mode) => ({
    stochastic: "Stochastic",
    deterministic: "Deterministic",
    epsilon_greedy: "Epsilon-greedy",
    greedy: "Greedy",
    program: "Program",
  })[mode] || String(mode).replaceAll("_", " ");
  const commands = {
    pause: () => services.pauseCurrentPlayback(),
    play: () => services.playFromCurrentPosition(),
    step: () => services.command("step", { count: 1 }),
    "step-ten": () => services.command("step", { count: 10 }),
    "continue-event": () => services.command("continue", { target: "any" }),
    "reset-episode": () => services.command("reset_episode", {
      seed: seed.value,
    }),
    "next-episode": () => services.command("next_episode", {
      sampling_mode: sampling.value,
      driver: "policy",
    }),
    "set-contract-mode": () => services.command("set_contract_mode", {
      mode: contractMode.value,
    }),
    "apply-termination": () => services.command("set_termination_conditions", {
      enabled: [...terminationOptions.querySelectorAll("input:checked")]
        .map((input) => input.value),
    }),
  };
  element.querySelectorAll("[data-command]").forEach((button) => {
    button.addEventListener("click", () => commands[button.dataset.command]());
  });
  fps.addEventListener("input", () => {
    if (!fps.validity.valid || fps.value.trim() === "") return;
    services.command("set_fps", { fps: Number(fps.value) });
  });
  const updateControl = () => {
    const state = services.getState();
    fps.disabled = !state.hasControl;
    element.querySelectorAll("button:not([data-panel-menu]):not([data-drag-handle]):not(.panel-resize)")
      .forEach((button) => { button.disabled = !state.hasControl; });
    const session = state.liveSnapshot?.session || state.snapshot?.session || {};
    element.querySelectorAll("[data-requires-active-episode]")
      .forEach((button) => {
        button.disabled = !state.hasControl || Boolean(session.awaiting_next_episode);
      });
    playbackToggle.disabled = !state.hasControl || (
      Boolean(session.awaiting_next_episode)
      && !state.replayingInspection
      && !services.canReplayInspection()
    );
    const recording = (state.liveSnapshot?.mode || state.snapshot?.mode) === "recording";
    contractMode.disabled = recording || !state.hasControl;
    const canChangeTermination = (
      !recording
      && state.hasControl
      && (Number(session.step || 0) === 0 || Boolean(session.awaiting_next_episode))
    );
    applyTermination.disabled = !canChangeTermination;
    terminationOptions.querySelectorAll("input").forEach((input) => {
      input.disabled = !canChangeTermination;
    });
    const canPrepareNextEpisode = (
      !recording && state.hasControl && Boolean(session.can_start_next_episode)
    );
    const canResetEpisode = (
      !recording
      && state.hasControl
      && (
        !Boolean(session.awaiting_next_episode)
        || Boolean(session.can_start_next_episode)
      )
    );
    resetEpisode.disabled = !canResetEpisode;
    nextEpisode.disabled = !canPrepareNextEpisode;
    seed.disabled = !canResetEpisode;
    const supportedModes = (
      state.liveSnapshot?.policy?.action_selection?.supported_modes
      || state.snapshot?.policy?.action_selection?.supported_modes
      || []
    );
    sampling.disabled = !canPrepareNextEpisode || supportedModes.length <= 1;
    nextEpisodeSettings.classList.toggle(
      "available",
      canPrepareNextEpisode || canResetEpisode,
    );
  };

  const renderPlaybackToggle = (runState) => {
    const running = services.getState().replayingInspection
      || ["playing", "stepping", "continuing"].includes(runState);
    const command = running ? "pause" : "play";
    playbackToggle.title = running
      ? "Pause after the current transition"
      : (services.canReplayInspection()
        ? "Replay from the selected step"
        : "Play current episode");
    if (playbackToggle.dataset.command === command) return;
    const label = running ? "Pause" : "Play";
    playbackToggle.dataset.command = command;
    playbackToggle.classList.toggle("primary", !running);
    playbackToggle.setAttribute("aria-label", label);
    playbackIcon.setAttribute("href", `/assets/tabler-icons.svg#ti-player-${command}`);
  };

  return {
    element,
    updateControl,
    render(snapshot, view = {}) {
      if (view.inspection) snapshot = services.getState().liveSnapshot || snapshot;
      if (!snapshot) { updateControl(); return; }
      const session = snapshot.session || {};
      const actionSelection = snapshot.policy?.action_selection || {};
      const supportedModes = Array.isArray(actionSelection.supported_modes)
        ? actionSelection.supported_modes
        : ["stochastic", "deterministic"];
      const selectionKey = JSON.stringify(supportedModes);
      if (sampling.dataset.modes !== selectionKey) {
        sampling.dataset.modes = selectionKey;
        sampling.replaceChildren(...supportedModes.map((mode) => {
          const option = document.createElement("option");
          option.value = mode;
          option.textContent = selectionLabel(mode);
          return option;
        }));
      }
      if (document.activeElement !== fps) fps.value = Number(session.target_fps || 0);
      const awaitingNextEpisode = Boolean(session.awaiting_next_episode);
      if (!awaitingNextEpisode || !wasAwaitingNextEpisode) {
        if (document.activeElement !== seed) seed.value = text(session.seed, "");
        sampling.value = actionSelection.requested_mode
          || session.sampling_mode
          || actionSelection.default_mode
          || supportedModes[0]
          || "";
      }
      wasAwaitingNextEpisode = awaitingNextEpisode;
      element.querySelector("[data-session-summary]").textContent = `${snapshot.run_state.toUpperCase()} · ${snapshot.driver.toUpperCase()}`;
      renderPlaybackToggle(snapshot.run_state);
      const recording = snapshot.mode === "recording";
      const dataset = snapshot.mode === "dataset";
      const playbackContract = session.playback_contract || {};
      const availableContractModes = Array.isArray(playbackContract.available_modes)
        ? playbackContract.available_modes
        : ["training"];
      if (document.activeElement !== contractMode) {
        contractMode.value = playbackContract.mode || "training";
      }
      [...contractMode.options].forEach((option) => {
        option.disabled = !availableContractModes.includes(option.value);
      });
      contractSettings.hidden = recording || dataset;
      const comparisonReasons = Array.isArray(session.critic_comparison?.reasons)
        ? session.critic_comparison.reasons
        : [];
      const mismatchPaths = Array.isArray(playbackContract.mismatch_paths)
        ? playbackContract.mismatch_paths
        : [];
      const contractMessages = [];
      if (comparisonReasons.length) {
        contractMessages.push(`Critic comparison unavailable: ${comparisonReasons.join("; ")}.`);
      }
      if (playbackContract.legacy_mismatch) {
        contractMessages.push(
          `Legacy warning: published evaluation semantics differ from training${
            mismatchPaths.length ? ` at ${mismatchPaths.join(", ")}` : ""
          }.`,
        );
      }
      contractHint.textContent = contractMessages.join(" ")
        || "Training-compatible critic comparison is available after a terminal episode.";
      const terminationConditions = Array.isArray(session.termination_conditions)
        ? session.termination_conditions
        : [];
      terminationSettings.hidden = recording || dataset || terminationConditions.length === 0;
      terminationSource.textContent = `Defaults: ${session.termination_source || "training"}`;
      const terminationKey = JSON.stringify(terminationConditions);
      if (terminationOptions.dataset.key !== terminationKey) {
        terminationOptions.dataset.key = terminationKey;
        terminationOptions.replaceChildren(...terminationConditions.map((condition) => {
          const label = document.createElement("label");
          label.className = "termination-option";
          const input = document.createElement("input");
          input.type = "checkbox";
          input.value = condition.id;
          input.checked = Boolean(condition.enabled);
          const name = document.createElement("span");
          name.textContent = condition.label;
          const outcome = document.createElement("small");
          outcome.textContent = condition.outcome;
          label.append(input, name, outcome);
          return label;
        }));
      }
      nextEpisodeSettings.hidden = recording || dataset;
      seedField.hidden = recording;
      playbackSampling.hidden = recording;
      const samplingHint = element.querySelector("#playback-sampling-hint");
      samplingHint.textContent = supportedModes.length === 1
        ? `${selectionLabel(supportedModes[0])} is fixed by this checkpoint.`
        : "Applies to the next playback episode only; departures are playback-only.";
      nextEpisode.hidden = recording;
      resetEpisode.hidden = recording;
      nextEpisodeHint.hidden = recording;
      nextEpisode.title = session.can_start_next_episode
        ? "Start the prepared next episode"
        : (session.awaiting_next_episode
          ? "The configured episode limit has been reached"
          : "Available after the current episode ends");
      nextEpisodeHint.textContent = session.can_start_next_episode
        ? "Starts the prepared policy episode"
        : (session.awaiting_next_episode
          ? "The configured episode limit has been reached"
          : "Available after the current episode ends");
      resetEpisode.title = session.awaiting_next_episode
        ? (session.can_start_next_episode
          ? "Reset the prepared next episode to the configured seed and pause"
          : "The configured episode limit has been reached")
        : "Reset the current episode to the configured seed and pause";
      fps.min = recording ? "1" : "0";
      element.querySelector("#playback-fps-hint").textContent = recording
        ? "Recording FPS must be at least 1 · changes apply immediately"
        : "0 runs playback uncapped · changes apply immediately";
      ["step", "step-ten", "continue-event"].forEach((name) => {
        element.querySelector(`[data-command="${name}"]`).hidden = recording;
      });
      updateControl();
    },
  };
}
