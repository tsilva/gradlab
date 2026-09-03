import { text } from "./panels/shared.js";

function selectionLabel(mode) {
  return ({
    stochastic: "Stochastic",
    deterministic: "Deterministic",
    epsilon_greedy: "Epsilon-greedy",
    greedy: "Greedy",
    program: "Program",
    route: "Route",
  })[mode] || String(mode || "").replaceAll("_", " ");
}

const TERMINATION_OUTCOME_ORDER = new Map([
  ["success", 0],
  ["failure", 1],
  ["timeout", 2],
]);

export function orderedTerminationConditions(conditions) {
  return (Array.isArray(conditions) ? conditions : [])
    .map((condition, index) => ({ condition, index }))
    .sort((left, right) => (
      (TERMINATION_OUTCOME_ORDER.get(String(left.condition?.outcome || "").toLowerCase()) ?? 3)
      - (TERMINATION_OUTCOME_ORDER.get(String(right.condition?.outcome || "").toLowerCase()) ?? 3)
      || left.index - right.index
    ))
    .map(({ condition }) => condition);
}

export function terminationOutcomeClass(outcome) {
  const normalized = String(outcome || "").toLowerCase();
  return TERMINATION_OUTCOME_ORDER.has(normalized) ? `outcome-${normalized}` : "";
}

export function frameSkipPresentation(playbackContract) {
  const values = playbackContract?.frame_skip;
  if (!values || typeof values !== "object") return null;
  const training = Number(values.training);
  const playback = Number(values.playback);
  if (
    !Number.isInteger(training)
    || training < 1
    || !Number.isInteger(playback)
    || playback < 1
  ) return null;
  return {
    training,
    playback,
    differs: training !== playback,
    label: `Frame skip · training ${training} · playback ${playback}`,
  };
}

export function mountPlaybackSettings({ services, idPrefix = "playback" }) {
  const element = document.createElement("div");
  element.className = "control-components playback-settings-form";
  element.innerHTML = `
    <div class="playback-glance" aria-live="polite">
      <strong data-playback-glance-contract>Training contract</strong>
      <span data-playback-glance-detail>Stochastic · seed —</span>
      <span data-playback-frame-skip hidden></span>
    </div>
    <div class="advanced-playback-body">
      <div class="playback-field playback-fps">
        <label for="${idPrefix}-fps">Play FPS</label>
        <input id="${idPrefix}-fps" data-fps type="number" min="0" step="1" value="0" inputmode="decimal">
      </div>
      <div class="playback-field next-episode-seed">
        <label for="${idPrefix}-seed">Reset seed</label>
        <input id="${idPrefix}-seed" data-seed inputmode="numeric">
      </div>
      <div class="playback-field playback-sampling">
        <label for="${idPrefix}-sampling">Next action selection</label>
        <select id="${idPrefix}-sampling" data-sampling aria-describedby="${idPrefix}-sampling-hint">
          <option value="stochastic">Stochastic</option>
          <option value="deterministic">Deterministic</option>
        </select>
      </div>
      <p id="${idPrefix}-sampling-hint" class="control-hint" data-sampling-hint hidden></p>
      <div class="playback-field playback-contract" data-playback-contract>
        <label for="${idPrefix}-contract-mode">Environment contract</label>
        <select id="${idPrefix}-contract-mode" data-contract-mode aria-describedby="${idPrefix}-contract-hint">
          <option value="training">Training contract</option>
          <option value="evaluation">Published evaluation</option>
          <option value="counterfactual">Counterfactual overrides</option>
        </select>
        <button data-apply-contract class="quiet button-with-icon" type="button" aria-label="Apply environment contract" title="Apply environment contract and start a new shared session"><svg class="icon" aria-hidden="true"><use href="/assets/tabler-icons.svg#ti-check"></use></svg><span>Apply</span></button>
      </div>
      <p id="${idPrefix}-contract-hint" class="control-hint" data-contract-hint>Training-time policy semantics are the default.</p>
      <fieldset class="termination-settings" data-termination-settings>
        <legend>Episode termination</legend>
        <p class="control-hint" data-termination-source></p>
        <div class="termination-options" data-termination-options></div>
        <p class="control-hint">Selections apply with Reset or Next episode.</p>
      </fieldset>
    </div>
  `;

  const seed = element.querySelector("[data-seed]");
  const fps = element.querySelector("[data-fps]");
  const sampling = element.querySelector("[data-sampling]");
  const samplingHint = element.querySelector("[data-sampling-hint]");
  const contractMode = element.querySelector("[data-contract-mode]");
  const contractSettings = element.querySelector("[data-playback-contract]");
  const applyContract = element.querySelector("[data-apply-contract]");
  const contractHint = element.querySelector("[data-contract-hint]");
  const terminationSettings = element.querySelector("[data-termination-settings]");
  const terminationOptions = element.querySelector("[data-termination-options]");
  const terminationSource = element.querySelector("[data-termination-source]");
  const glanceContract = element.querySelector("[data-playback-glance-contract]");
  const glanceDetail = element.querySelector("[data-playback-glance-detail]");
  const frameSkip = element.querySelector("[data-playback-frame-skip]");
  let wasAwaitingNextEpisode = false;

  const enabledTerminationConditions = () => {
    const inputs = [...terminationOptions.querySelectorAll("input")];
    return inputs.length
      ? inputs.filter((input) => input.checked).map((input) => input.value)
      : null;
  };

  const episodeOptions = () => ({
    seed: seed.value,
    sampling_mode: sampling.value,
    enabled_termination_conditions: enabledTerminationConditions(),
  });

  applyContract.addEventListener("click", () => services.command("set_contract_mode", {
    mode: contractMode.value,
  }));
  fps.addEventListener("input", () => {
    if (!fps.validity.valid || fps.value.trim() === "") return;
    services.command("set_fps", { fps: Number(fps.value) });
  });

  const updateControl = () => {
    const state = services.getState();
    const session = state.liveSnapshot?.session || state.snapshot?.session || {};
    const mode = state.liveSnapshot?.mode || state.snapshot?.mode;
    const readOnly = !state.hasControl;
    const recording = mode === "recording";
    const dataset = mode === "dataset";
    fps.disabled = readOnly;
    seed.disabled = readOnly || recording || dataset;
    sampling.disabled = readOnly || recording || dataset || sampling.options.length <= 1;
    contractMode.disabled = readOnly || recording || dataset;
    applyContract.disabled = readOnly || recording || dataset;
    const canChangeTermination = (
      !recording
      && !dataset
      && !readOnly
      && (Number(session.step || 0) === 0 || Boolean(session.awaiting_next_episode))
    );
    terminationOptions.querySelectorAll("input").forEach((input) => {
      input.disabled = !canChangeTermination;
    });
  };

  return {
    element,
    episodeOptions,
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
      const defaultSeed = text(session.default_seed, session.seed);
      if (document.activeElement !== seed && seed.dataset.defaultSeed !== defaultSeed) {
        seed.value = defaultSeed;
        seed.dataset.defaultSeed = defaultSeed;
      }
      if (!awaitingNextEpisode || !wasAwaitingNextEpisode) {
        sampling.value = actionSelection.requested_mode
          || session.sampling_mode
          || actionSelection.default_mode
          || supportedModes[0]
          || "";
      }
      wasAwaitingNextEpisode = awaitingNextEpisode;
      const recording = snapshot.mode === "recording";
      const dataset = snapshot.mode === "dataset";
      const playbackContract = session.playback_contract || {};
      const frameSkipDetails = frameSkipPresentation(playbackContract);
      frameSkip.hidden = frameSkipDetails === null;
      frameSkip.classList.toggle("contract-mismatch", Boolean(frameSkipDetails?.differs));
      frameSkip.textContent = frameSkipDetails?.label || "";
      frameSkip.title = frameSkipDetails
        ? `Training repeated each selected action for ${
          frameSkipDetails.training
        } environment frame${frameSkipDetails.training === 1 ? "" : "s"}. This playback repeats each selected action for ${
          frameSkipDetails.playback
        } environment frame${frameSkipDetails.playback === 1 ? "" : "s"}.`
        : "";
      const availableContractModes = Array.isArray(playbackContract.available_modes)
        ? playbackContract.available_modes
        : ["training"];
      if (document.activeElement !== contractMode) {
        contractMode.value = playbackContract.mode || "training";
      }
      const contractLabel = {
        training: "Training contract",
        evaluation: "Published evaluation",
        counterfactual: "Counterfactual — not evidence",
      }[playbackContract.mode || "training"] || selectionLabel(playbackContract.mode);
      glanceContract.textContent = contractLabel;
      glanceDetail.textContent = `${selectionLabel(sampling.value || session.sampling_mode)} · seed ${
        text(snapshot.transition?.seed, text(session.seed, defaultSeed))
      }`;
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
      if (playbackContract.evaluation_matches_training === false) {
        contractMessages.push(
          `Published evaluation semantics differ from training${
            mismatchPaths.length ? ` at ${mismatchPaths.join(", ")}` : ""
          }.`,
        );
      }
      contractHint.textContent = contractMessages.join(" ")
        || "Training-compatible critic comparison is available after a terminal episode.";
      const terminationConditions = orderedTerminationConditions(
        session.termination_conditions,
      );
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
          outcome.className = `termination-outcome ${
            terminationOutcomeClass(condition.outcome)
          }`.trim();
          outcome.textContent = condition.outcome;
          label.append(input, name, outcome);
          return label;
        }));
      }
      samplingHint.hidden = recording || dataset || supportedModes.length !== 1;
      samplingHint.textContent = supportedModes.length === 1
        ? `${selectionLabel(supportedModes[0])} is fixed by this checkpoint.`
        : "";
      fps.min = recording ? "1" : "0";
      updateControl();
    },
  };
}
