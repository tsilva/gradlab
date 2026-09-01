from __future__ import annotations

# ruff: noqa: E402

import argparse
import os
from collections import deque
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

from gradlab.local_paths import configure_matplotlib_cache, default_runs_dir

configure_matplotlib_cache()

import numpy as np
import torch
from gradlab.action_contract import (
    action_contract_meanings,
    action_contract_payload,
    action_value_for_controls,
    configured_action_meanings,
    configured_action_name,
)
from gradlab.batch_runtime import StepDiagnostics
from gradlab.cli_parser import ExactArgumentParser
from gradlab.env import (
    info_value_from_state_name,
    state_name_candidates_from_level_id,
    task_conditioning,
    task_max_episode_steps,
    task_reward,
    task_termination,
)
from gradlab.eval_metrics import (
    batch_metrics_for_lane,
    drain_runtime_records,
    episode_records,
    episode_result_from_record,
    is_level_complete,
)
from gradlab.model_sources import (
    DEFAULT_PUBLIC_MODELS_BASE_URL,
    positional_model_source_arg,
)
from gradlab.play_attribution import (
    ATTRIBUTION_MODES,
    AttributionError,
    PolicyActionAttributor,
    attribution_capability,
)
from gradlab.play_cnn import (
    CNN_INTERVAL_DEFAULT,
    CNN_TOP_K_DEFAULT,
    CNN_TOP_K_MAX,
    CNNInspection,
    CNNInspectionError,
    PolicyCNNInspector,
    cnn_inspection_capability,
)
from gradlab.play_debug import PolicyDecision
from gradlab.play_processing import (
    PLAYER_PROCESSING_FEATURES,
    normalize_player_processing,
)
from gradlab.policy_observation import (
    model_observation,
    task_info_value_from_info,
    task_info_vars,
    task_state_names,
)
from gradlab.policy_runtime import PolicyRuntime, reset_policy_state
from gradlab.play_termination import (
    configured_termination_ids,
    termination_condition_payload,
    with_enabled_termination_conditions,
)
from gradlab.seeds import DEFAULT_EVAL_SEED, EVAL_SEED_START
from gradlab.env_registry import environment_spec


PLAYBACK_DEVICE = "cpu"


ANSI_RESET = "\033[0m"
ANSI_STYLES = {
    "bold": "\033[1m",
    "dim": "\033[2m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "blue": "\033[34m",
    "yellow": "\033[33m",
    "magenta": "\033[35m",
}
ATTRIBUTION_INTERVAL_DEFAULTS = {"gradcam": 1, "occlusion": 8}


def _color(text: str, style: str) -> str:
    if os.environ.get("NO_COLOR"):
        return text
    return f"{ANSI_STYLES[style]}{text}{ANSI_RESET}"


def _summary_line(icon: str, label: str, value: str, style: str) -> str:
    return f"  {_color(icon, style)} {_color(label + ':', 'dim')} {value}"


def _format_sequence(value) -> str:
    if not value:
        return "-"
    if isinstance(value, str):
        return value
    return ",".join(str(item) for item in value)


def fast_env_image_obs(obs) -> np.ndarray:
    if isinstance(obs, Mapping):
        key = "observation" if "observation" in obs else "image"
        if key not in obs:
            raise ValueError(
                f"dict fast env obs is missing 'observation' or 'image'; keys={tuple(obs)}"
            )
        obs = obs[key]
    return np.asarray(obs)


def fast_env_obs(obs: np.ndarray) -> np.ndarray:
    arr = fast_env_image_obs(obs)
    if arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[1] == 4:
        return arr
    if arr.ndim == 3 and arr.shape[0] == 4:
        return arr[None, ...]
    raise ValueError(
        f"expected channel-first fast env obs with 4 stacked frames, got shape {arr.shape}"
    )


def fast_env_frames(obs: np.ndarray) -> deque[np.ndarray]:
    arr = fast_env_image_obs(obs)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] == 4:
        return deque([arr[idx, ..., None] for idx in range(arr.shape[0])], maxlen=4)
    raise ValueError(
        f"expected channel-first fast env obs with 4 stacked frames, got shape {arr.shape}"
    )


def task_conditioning_change_message(
    *,
    episode: int,
    step: int,
    old_task: object,
    new_task: object,
    task_index: int,
    task_count: int,
) -> str:
    task_vector = [1 if index == task_index else 0 for index in range(task_count)]
    return (
        "task_conditioning_change "
        f"episode={episode} step={step} old={old_task!r} new={new_task!r} "
        f"index={task_index} one_hot={task_vector}"
    )


def task_state_from_info(info: dict, task_states: tuple[str, ...]) -> str | None:
    level_id = info.get("level_id")
    if not isinstance(level_id, str) or not level_id:
        return None
    for level_state in state_name_candidates_from_level_id(level_id):
        if level_state in task_states:
            return level_state
    return None


def playback_should_end_episode(terminated: bool, truncated: bool, completed: bool) -> bool:
    # Completion is shown in playback output, but GUI playback keeps going unless
    # the environment actually terminates or truncates the episode.
    del completed
    return bool(terminated or truncated)


def vector_env_frame(env) -> np.ndarray:
    images = env.get_images()
    if not images or images[0] is None:
        raise RuntimeError("native vector provider did not return an RGB frame for lane 0")
    return np.asarray(images[0]).copy()


def _heatmap_color(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.clip(np.asarray(heatmap, dtype=np.float32), 0.0, 1.0)
    red = np.clip(1.6 * heatmap, 0.0, 1.0)
    green = np.clip(1.8 * heatmap - 0.25, 0.0, 1.0)
    blue = np.clip(1.0 - heatmap, 0.0, 1.0) * 0.35
    return np.stack([red, green, blue], axis=2) * 255.0


def render_obs_stack(
    frames: deque[np.ndarray],
    scale: int,
) -> np.ndarray:
    if scale < 1:
        raise ValueError("obs viewer scale must be >= 1")
    panels = []
    for frame in frames:
        gray = frame[..., 0]
        panel = np.repeat(gray[..., None], 3, axis=2)
        if scale != 1:
            panel = np.repeat(np.repeat(panel, scale, axis=0), scale, axis=1)
        panels.append(panel)
    return np.concatenate(panels, axis=1)


def render_attribution_stack(
    frames: tuple[np.ndarray, ...] | deque[np.ndarray],
    heatmap: np.ndarray,
    scale: int = 1,
) -> np.ndarray:
    """Render an opacity-independent RGBA map aligned with the observation stack."""

    if scale < 1:
        raise ValueError("obs viewer scale must be >= 1")
    values = np.asarray(heatmap, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"attribution heatmap must be 2D, got shape {values.shape}")
    if scale != 1:
        values = np.repeat(np.repeat(values, scale, axis=0), scale, axis=1)
    values = np.clip(values, 0.0, 1.0)
    color = _heatmap_color(values).astype(np.uint8)
    alpha = np.rint(values * 255.0).astype(np.uint8)[..., None]
    panel = np.concatenate([color, alpha], axis=2)
    rendered = []
    for frame in frames:
        height, width = np.asarray(frame).shape[:2]
        expected = (height * scale, width * scale)
        if values.shape != expected:
            raise ValueError(
                "attribution heatmap shape does not match observation frame: "
                f"{values.shape} vs {expected}"
            )
        rendered.append(panel.copy())
    return np.concatenate(rendered, axis=1)


def add_play_source_args(parser: argparse.ArgumentParser) -> None:
    def positional_play_source_arg(value: str) -> str:
        from gradlab.play_catalog import is_wandb_url

        if is_wandb_url(value):
            return value
        return positional_model_source_arg(value)

    parser.add_argument(
        "artifact_ref",
        nargs="?",
        type=positional_play_source_arg,
        help=(
            "W&B run URL, immutable public checkpoint manifest, or Hugging Face model ref. "
            "Use --model for a local checkpoint or --run for an gradlab public run."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Local gradlab policy path. The artifact must have model.json and recipe.json sidecars.",
    )
    parser.add_argument(
        "--recipe",
        help=(
            "Play the newest completed local run for a built-in <goal-path>/<recipe> "
            "reference or recipe YAML."
        ),
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=default_runs_dir(),
        help="Local run root searched by --recipe; defaults to ~/.config/gradlab/runs.",
    )
    parser.add_argument(
        "--rom-path",
        type=Path,
        help=(
            "Use a provider-compatible raw .nes ROM, or a local ViZDoom IWAD as a "
            "visible counterfactual when it differs from training."
        ),
    )
    parser.add_argument(
        "--run",
        help=(
            "Immutable gradlab run ID or exact public checkpoint manifest URL. A run ID "
            "resolves its promoted checkpoint, or its highest-step final checkpoint when "
            "no promotion exists, without W&B or private R2 credentials."
        ),
    )
    parser.add_argument(
        "--public-models-base-url",
        default=DEFAULT_PUBLIC_MODELS_BASE_URL,
        help="Public models bucket URL. Defaults to gradlab's checked-in public endpoint.",
    )
    parser.add_argument(
        "--public-model-root",
        default=str(default_runs_dir() / "public_models"),
        help="Local cache for public run checkpoints.",
    )
    parser.set_defaults(
        hf_revision=None,
        hf_model_root=str(default_runs_dir() / "hf_models"),
    )


def nonnegative_int_arg(value: str) -> int:
    parsed = int(value)
    if parsed < 0 or parsed > 65535:
        raise argparse.ArgumentTypeError("must be in [0, 65535]")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = ExactArgumentParser(
        prog="gradlab play",
        description=(
            "Browse repository goals and control-plane runs, then inspect a public "
            "checkpoint in the interactive web player"
        ),
    )
    add_play_source_args(parser)
    parser.add_argument(
        "--episodes", type=int, default=0, help="Number of episodes; use 0 to run forever"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_EVAL_SEED,
        help=(
            "Base playback seed. The default lives in the eval/play-reserved seed "
            f"range >= {EVAL_SEED_START}; overrides must stay in that range."
        ),
    )
    parser.add_argument(
        "--device",
        default=PLAYBACK_DEVICE,
        choices=[PLAYBACK_DEVICE],
        help="Inference device; playback currently runs on CPU only.",
    )
    parser.add_argument(
        "--env-provider",
        help=(
            "Run the artifact's unchanged evaluation contract through an equivalent provider. "
            "The provider must support the recorded game and constructor arguments."
        ),
    )
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument(
        "--port",
        type=nonnegative_int_arg,
        default=0,
        help="Loopback dashboard port; use 0 to select an available port automatically.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Print the play and stats dashboard URLs without opening the default browser.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Start paused for interactive transition debugging.",
    )
    parser.add_argument(
        "--continuous-play",
        action="store_true",
        help=(
            "Ignore the recipe's task success, failure, stall, and step-limit boundaries. "
            "This is a semantic deviation intended only for continuous interactive play."
        ),
    )
    parser.add_argument(
        "--resume-cell",
        help=(
            "Resume a cell-graph representative by node ID. The model must have "
            "been exported with state_archive.export.snapshots=retained."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable model-download and player-startup progress bars.",
    )
    return parser


def resolved_play_launch_lines(
    args: argparse.Namespace,
    *,
    argv: list[str],
    artifact_ref: str | None,
    policy_config,
    display_config,
) -> list[str]:
    states = policy_config.states or ((policy_config.state,) if policy_config.state else ())
    return [
        _color("▶ resolved play launch", "bold"),
        _summary_line("›", "argv", " ".join(argv) if argv else "-", "cyan"),
        _summary_line("◇", "artifact", artifact_ref or "-", "magenta"),
        _summary_line("▣", "model", args.model, "magenta"),
        _summary_line(
            "●",
            "policy/eval env",
            f"{policy_config.env_provider} game={policy_config.game} "
            f"state={policy_config.state or '-'} states={_format_sequence(states)}",
            "green",
        ),
        _summary_line(
            "○",
            "viewer source",
            f"{display_config.env_provider} game={display_config.game} "
            f"state={display_config.state or '-'} shared_with_policy=True",
            "blue",
        ),
        _summary_line(
            "▶",
            "policy",
            f"device={args.device} stochastic=True "
            f"seed={args.seed} episodes={args.episodes} "
            f"max_steps={task_max_episode_steps(policy_config)} "
            f"debug={getattr(args, 'debug', False)} "
            f"resume_cell={getattr(args, 'resume_cell', None) or '-'} "
            "interface=web "
            f"respect_task_termination={getattr(args, 'respect_task_termination', False)}",
            "green",
        ),
        _summary_line(
            "▤",
            "preprocessing",
            f"frame_skip={policy_config.frame_skip} max_pool={policy_config.max_pool_frames} "
            f"sticky={policy_config.sticky_action_prob} "
            f"obs={_format_sequence(policy_config.obs_resize)} "
            f"crop={_format_sequence(policy_config.obs_crop)} "
            f"crop_mode={policy_config.obs_crop_mode} crop_fill={policy_config.obs_crop_fill} "
            f"resize={policy_config.obs_resize_algorithm}",
            "yellow",
        ),
        _summary_line(
            "⚙",
            "action/reward",
            f"action_set={configured_action_name(policy_config)} "
            f"reward_mode={task_reward(policy_config).get('reward_mode')} "
            f"reward_scale={task_reward(policy_config).get('reward_scale')} "
            f"reward_clip={task_reward(policy_config).get('reward_clip')}",
            "yellow",
        ),
        _summary_line(
            "◆",
            "task/events",
            f"task_conditioning={bool(task_conditioning(policy_config).get('enabled'))} "
            f"task_info_vars={_format_sequence(task_info_vars(policy_config))} "
            f"termination_events={_format_sequence((*task_termination(policy_config).get('failure', ()), *task_termination(policy_config).get('success', ())))}",
            "cyan",
        ),
        _summary_line(
            "✓",
            "source of truth",
            "one policy/eval env supplies both viewers, observations, rewards, dones, and info",
            "green",
        ),
    ]


def optional_vector_env_frame(env) -> np.ndarray | None:
    try:
        return vector_env_frame(env)
    except AttributeError, NotImplementedError, RuntimeError, TypeError, ValueError:
        return None


def optional_fast_env_frames(obs) -> deque[np.ndarray] | None:
    try:
        return fast_env_frames(obs)
    except KeyError, TypeError, ValueError:
        return None


def playback_model_observation(
    model,
    policy_obs,
    config,
    *,
    active_task_state: str | None,
    active_info_value: tuple[int | str, ...] | None,
):
    spaces = getattr(getattr(model, "observation_space", None), "spaces", None)
    if isinstance(spaces, dict) and {"image", "task"}.issubset(spaces):
        return model_observation(
            model,
            fast_env_obs(policy_obs),
            config,
            active_task_state=active_task_state,
            active_info_value=active_info_value,
        )
    if isinstance(policy_obs, Mapping):
        return policy_obs
    try:
        return fast_env_obs(policy_obs)
    except ValueError:
        return np.asarray(policy_obs)


def _observation_shape(value) -> str:
    if isinstance(value, Mapping):
        return (
            "{"
            + ", ".join(f"{name}:{np.asarray(item).shape}" for name, item in value.items())
            + "}"
        )
    return str(np.asarray(value).shape)


@dataclass(frozen=True)
class _PlaybackTransition:
    sequence: int
    episode: int
    step: int
    seed: int | None
    start_id: str | None
    model_obs: object
    decision: PolicyDecision | None
    action_source: str
    executed_action: object
    diagnostics: StepDiagnostics | None
    info: dict[str, object]
    before_frame: np.ndarray | None
    after_frame: np.ndarray | None
    before_frames: tuple[np.ndarray, ...]
    after_frames: tuple[np.ndarray, ...]
    attribution: np.ndarray | None
    pre_task: object
    next_task: object
    reward: float
    total_reward: float
    max_x_pos: int
    terminated: bool
    truncated: bool
    completed: bool
    boundary: bool
    after_frame_role: str = "after_action_observation"
    attribution_status: str = "off"
    attribution_mode: str = "none"
    attribution_generation: int = 0
    attribution_reason: str | None = "disabled"
    cnn_inspection: CNNInspection | None = None
    cnn_status: str = "off"
    cnn_layer_id: str | None = None
    cnn_generation: int = 0
    cnn_reason: str | None = "disabled"

    @property
    def events(self) -> tuple[str, ...]:
        return self.diagnostics.events if self.diagnostics is not None else ()


class _PlaybackSession:
    """The sole mutable owner of one playback trajectory."""

    def __init__(
        self,
        *,
        model,
        env,
        config,
        initial_seed: int,
        policy_runtime: PolicyRuntime | None = None,
        policy_provenance: Mapping[str, object] | None = None,
        env_factory=None,
        termination_base_config=None,
        termination_source: str = "training",
        attributor_factory=PolicyActionAttributor,
        cnn_inspector_factory=PolicyCNNInspector,
    ):
        self.model = model
        self.policy_runtime = policy_runtime
        self._policy_runtime_error: Exception | None = (
            None
            if policy_runtime is not None
            else RuntimeError("policy runtime identity was not provided")
        )
        self.env = env
        self.config = config
        self.initial_seed = initial_seed
        self.attributor: PolicyActionAttributor | None = None
        self._attributor_factory = attributor_factory
        algorithm_id = (
            None if policy_runtime is None else str(policy_runtime.capabilities.algorithm_id)
        )
        self.attribution_capability = attribution_capability(model, algorithm_id)
        self.attribution_mode = "none"
        self.attribution_interval = ATTRIBUTION_INTERVAL_DEFAULTS["gradcam"]
        self.attribution_status = "off"
        self.attribution_error: str | None = None
        self.attribution_generation = 0
        self.attribution_last_computed_sequence: int | None = None
        self.cnn_inspector: PolicyCNNInspector | None = None
        self._cnn_inspector_factory = cnn_inspector_factory
        self.cnn_capability = cnn_inspection_capability(model)
        self.cnn_enabled = False
        self.cnn_layer_id = self.cnn_capability.get("default_layer_id")
        self.cnn_interval = CNN_INTERVAL_DEFAULT
        self.cnn_top_k = CNN_TOP_K_DEFAULT
        self.cnn_status = "off"
        self.cnn_error: str | None = None
        self.cnn_generation = 0
        self.cnn_last_computed_sequence: int | None = None
        self.processing_features = PLAYER_PROCESSING_FEATURES
        self.policy_provenance = dict(policy_provenance or {})
        self.env_factory = env_factory
        self.termination_base_config = termination_base_config or config
        self.termination_source = termination_source
        self.info_vars = task_info_vars(config)
        self.conditioning_enabled = bool(task_conditioning(config).get("enabled"))
        self.configured_task_states = task_state_names(config) if self.conditioning_enabled else ()
        self.action_contract: Mapping[str, object] | None = None
        self._refresh_action_contract()
        from gradlab.policy_execution import verify_policy_execution_contract

        verify_policy_execution_contract(self.model, self.env)

        self.policy_obs = None
        self.current_frame: np.ndarray | None = None
        self.frames: deque[np.ndarray] | None = None
        self.active_task_state: str | None = None
        self.active_info_value: tuple[int | str, ...] | None = None
        self.active_seed: int | None = initial_seed
        self.episode = 1
        self.step_index = 0
        self.sequence = 0
        self.total_reward = 0.0
        self.max_x_pos = 0
        self.interactive = False
        self.last_transition: _PlaybackTransition | None = None

    def _refresh_action_contract(self) -> None:
        runtime = getattr(self.env, "runtime", None)
        contract = getattr(runtime, "action_contract", None)
        if isinstance(contract, Mapping):
            self.action_contract = contract
            self.action_contract_payload = action_contract_payload(contract)
            self.action_names = action_contract_meanings(contract)
            return
        self.action_contract = None
        self.action_contract_payload = None
        try:
            self.action_names = configured_action_meanings(self.config)
        except ValueError:
            self.action_names = ()

    @property
    def termination_conditions(self) -> list[dict[str, object]]:
        return termination_condition_payload(self.termination_base_config, self.config)

    def set_termination_conditions(self, enabled_ids: list[str] | tuple[str, ...]) -> None:
        if self.env_factory is None:
            raise RuntimeError("this playback session cannot reconfigure termination conditions")
        enabled = tuple(str(condition_id) for condition_id in enabled_ids)
        configured = set(configured_termination_ids(self.termination_base_config))
        unknown = sorted(set(enabled) - configured)
        if unknown:
            raise ValueError(f"unknown termination condition(s): {', '.join(unknown)}")
        next_config = with_enabled_termination_conditions(
            self.termination_base_config,
            enabled,
        )
        if next_config.task == self.config.task:
            return

        seed = self.initial_seed if self.active_seed is None else int(self.active_seed)
        previous_env = self.env
        previous_config = self.config
        next_env = self.env_factory(next_config, seed)
        try:
            from gradlab.policy_runtime import bind_policy_action_space

            bind_policy_action_space(
                self.model,
                next_env.action_space,
                getattr(getattr(next_env, "runtime", None), "action_contract", None),
            )
            from gradlab.policy_execution import verify_policy_execution_contract

            verify_policy_execution_contract(self.model, next_env)
            self.env = next_env
            self.config = next_config
            self._refresh_action_contract()
            self.restart(seed, reset_episode_index=False)
        except Exception:
            self.env = previous_env
            self.config = previous_config
            next_env.close()
            raise
        previous_env.close()

    @property
    def active_task(self):
        return (
            self.active_info_value if self.active_info_value is not None else self.active_task_state
        )

    @property
    def model_obs(self):
        return playback_model_observation(
            self.model,
            self.policy_obs,
            self.config,
            active_task_state=self.active_task_state,
            active_info_value=self.active_info_value,
        )

    def _set_initial_conditioning(self, reset_info: Mapping[str, object]) -> None:
        self.active_task_state = (
            (self.config.state or self.configured_task_states[0])
            if self.configured_task_states
            else None
        )
        self.active_info_value = None
        if self.info_vars:
            self.active_info_value = task_info_value_from_info(reset_info, self.config)
            if self.active_info_value is None:
                self.active_info_value = info_value_from_state_name(
                    self.active_task_state or "",
                    self.info_vars,
                )
        elif self.configured_task_states:
            self._update_conditioning(reset_info)

    def _update_conditioning(self, info: Mapping[str, object]) -> None:
        if self.info_vars:
            next_value = None
            if "level_hi" in info and "level_lo" in info:
                next_value = (int(info["level_hi"]), int(info["level_lo"]))
            if next_value is None:
                next_value = task_info_value_from_info(info, self.config)
            if next_value is not None:
                self.active_info_value = next_value
            return
        if not self.configured_task_states:
            return
        candidate = info.get("start_id")
        if isinstance(candidate, str) and candidate in self.configured_task_states:
            self.active_task_state = candidate
            return
        mutable_info = dict(info)
        if "level_hi" in mutable_info and "level_lo" in mutable_info:
            mutable_info["level_id"] = (
                f"{int(mutable_info['level_hi'])}-{int(mutable_info['level_lo'])}"
            )
        next_state = task_state_from_info(mutable_info, self.configured_task_states)
        if next_state is not None:
            self.active_task_state = next_state

    def restart(
        self,
        seed: int | None = None,
        *,
        reset_episode_index: bool = True,
    ) -> None:
        episode = 1 if reset_episode_index else self.episode
        seed = self.initial_seed if seed is None else seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if bool(getattr(self.model, "use_sde", False)):
            self.model.policy.reset_noise()
        self.env.seed(seed)
        self.policy_obs = self.env.reset()
        reset_policy_state(self.model)
        reset_info = dict(self.env.reset_infos[0])
        self._set_initial_conditioning(reset_info)
        self.active_seed = seed
        self.episode = episode
        self.step_index = 0
        self.total_reward = 0.0
        self.max_x_pos = 0
        self.interactive = False
        self.last_transition = None
        self.current_frame = optional_vector_env_frame(self.env)
        self.frames = optional_fast_env_frames(self.policy_obs)

    def resume_cell(
        self,
        node_id: str,
        *,
        entry_document: Mapping[str, object],
        payload: bytes,
    ) -> None:
        """Restore an explicitly embedded cell snapshot and continue its route."""

        runtime = getattr(self.env, "runtime", None)
        archive = getattr(runtime, "state_archive", None)
        resume_node = getattr(self.model, "resume_node", None)
        if runtime is None or archive is None or not callable(resume_node):
            raise RuntimeError("this playback session cannot resume cell snapshots")
        entry = archive.import_entry(entry_document, payload)
        self.policy_obs = runtime.restore_archive_entries(
            np.asarray([True], dtype=np.bool_),
            (entry.entry_id,),
        )
        reset_policy_state(self.model)
        resume_node(str(node_id), lane=0)
        mark_resumed = getattr(self.env, "mark_policy_resumed", None)
        if callable(mark_resumed):
            mark_resumed()
        reset_info = dict(self.env.reset_infos[0])
        self._set_initial_conditioning(reset_info)
        runtime_state = entry.runtime_state
        self.active_seed = (
            self.initial_seed
            if runtime_state is None or runtime_state.episode_seed is None
            else runtime_state.episode_seed
        )
        self.episode = self.episode if runtime_state is None else runtime_state.episode_index + 1
        self.step_index = 0 if runtime_state is None else runtime_state.episode_length
        self.total_reward = 0.0 if runtime_state is None else runtime_state.episode_return
        self.max_x_pos = 0
        self.interactive = False
        self.last_transition = None
        self.current_frame = optional_vector_env_frame(self.env)
        self.frames = optional_fast_env_frames(self.policy_obs)

    def reset_episode(self, seed: int | None = None) -> None:
        """Abandon the active trajectory and return to step zero."""

        if self.step_index > 0:
            self.episode += 1
        active_seed = self.active_seed if seed is None else seed
        self.restart(active_seed, reset_episode_index=False)

    @property
    def policy_capabilities(self) -> dict[str, object]:
        if self.policy_runtime is None:
            return {
                "algorithm_id": None,
                "action_selection": {
                    "supported_modes": [],
                    "default_mode": None,
                },
                "introspection": [],
            }
        return self.policy_runtime.capabilities.payload(
            attribution_available=bool(self.attribution_capability.get("supported_modes")),
        )

    def set_processing(self, features: tuple[str, ...] | frozenset[str]) -> None:
        self.processing_features = normalize_player_processing(features)

    @property
    def attribution_state(self) -> dict[str, object]:
        return {
            "mode": self.attribution_mode,
            "interval": self.attribution_interval,
            "status": self.attribution_status,
            "error": self.attribution_error,
            "generation": self.attribution_generation,
            "last_computed_sequence": self.attribution_last_computed_sequence,
        }

    @property
    def cnn_state(self) -> dict[str, object]:
        return {
            "enabled": self.cnn_enabled,
            "layer_id": self.cnn_layer_id,
            "interval": self.cnn_interval,
            "top_k": self.cnn_top_k,
            "status": self.cnn_status,
            "error": self.cnn_error,
            "generation": self.cnn_generation,
            "last_computed_sequence": self.cnn_last_computed_sequence,
        }

    def _cnn_for_transition(
        self,
        transition: _PlaybackTransition,
    ) -> _PlaybackTransition:
        if not transition.before_frames:
            return replace(
                transition,
                cnn_inspection=None,
                cnn_status="not_computed",
                cnn_layer_id=self.cnn_layer_id,
                cnn_generation=0,
                cnn_reason="no_image_observation",
            )
        self.cnn_generation += 1
        generation = self.cnn_generation
        try:
            if self.cnn_inspector is None:
                self.cnn_inspector = self._cnn_inspector_factory(self.model)
            inspection = self.cnn_inspector.inspect(
                transition.model_obs,
                layer_id=str(self.cnn_layer_id),
                top_k=self.cnn_top_k,
                generation=generation,
            )
        except Exception as exc:
            message = f"CNN inspection failed: {exc}"
            self.cnn_status = "error"
            self.cnn_error = message
            return replace(
                transition,
                cnn_inspection=None,
                cnn_status="error",
                cnn_layer_id=self.cnn_layer_id,
                cnn_generation=generation,
                cnn_reason=message,
            )
        self.cnn_last_computed_sequence = transition.sequence
        return replace(
            transition,
            cnn_inspection=inspection,
            cnn_status="available",
            cnn_layer_id=self.cnn_layer_id,
            cnn_generation=generation,
            cnn_reason=None,
        )

    def configure_cnn_inspection(
        self,
        *,
        enabled: bool,
        layer_id: str | None = None,
        interval: int | None = None,
        top_k: int | None = None,
    ) -> _PlaybackTransition | None:
        if not isinstance(enabled, bool):
            raise ValueError("CNN inspection enabled must be a boolean")
        if not enabled:
            self.cnn_enabled = False
            self.cnn_status = "off"
            self.cnn_error = None
            if self.last_transition is not None:
                self.last_transition = replace(
                    self.last_transition,
                    cnn_inspection=None,
                    cnn_status="off",
                    cnn_layer_id=self.cnn_layer_id,
                    cnn_generation=0,
                    cnn_reason="disabled",
                )
            return self.last_transition

        layers = {
            str(item.get("id"))
            for item in self.cnn_capability.get("layers", ())
            if isinstance(item, Mapping) and item.get("id") is not None
        }
        if not layers:
            reason = self.cnn_capability.get("unavailable_reason")
            raise ValueError(str(reason or "CNN inspection is unavailable"))
        resolved_layer = str(layer_id or self.cnn_layer_id or "")
        if resolved_layer not in layers:
            raise ValueError(f"unknown convolutional layer {resolved_layer!r}")
        if isinstance(interval, bool):
            raise ValueError("CNN inspection interval must be a positive integer")
        if isinstance(interval, float) and not interval.is_integer():
            raise ValueError("CNN inspection interval must be a positive integer")
        resolved_interval = self.cnn_interval if interval is None else int(interval)
        if resolved_interval < 1:
            raise ValueError("CNN inspection interval must be >= 1")
        if isinstance(top_k, bool):
            raise ValueError(f"CNN inspection top_k must be in [1, {CNN_TOP_K_MAX}]")
        if isinstance(top_k, float) and not top_k.is_integer():
            raise ValueError(f"CNN inspection top_k must be in [1, {CNN_TOP_K_MAX}]")
        resolved_top_k = self.cnn_top_k if top_k is None else int(top_k)
        if not 1 <= resolved_top_k <= CNN_TOP_K_MAX:
            raise ValueError(f"CNN inspection top_k must be in [1, {CNN_TOP_K_MAX}]")

        self.cnn_enabled = True
        self.cnn_layer_id = resolved_layer
        self.cnn_interval = resolved_interval
        self.cnn_top_k = resolved_top_k
        self.cnn_status = "active"
        self.cnn_error = None
        try:
            if self.cnn_inspector is None:
                self.cnn_inspector = self._cnn_inspector_factory(self.model)
        except Exception as exc:
            self.cnn_status = "error"
            self.cnn_error = f"CNN inspection failed to initialize: {exc}"
            raise CNNInspectionError(self.cnn_error) from exc

        if self.last_transition is not None:
            self.last_transition = self._cnn_for_transition(self.last_transition)
            if self.cnn_status == "error":
                raise CNNInspectionError(self.cnn_error or "CNN inspection failed")
        return self.last_transition

    def _attribution_for_transition(
        self,
        transition: _PlaybackTransition,
    ) -> _PlaybackTransition:
        if transition.decision is None:
            return replace(
                transition,
                attribution=None,
                attribution_status="not_computed",
                attribution_mode=self.attribution_mode,
                attribution_generation=0,
                attribution_reason="no_policy_decision",
            )
        if not transition.before_frames:
            return replace(
                transition,
                attribution=None,
                attribution_status="not_computed",
                attribution_mode=self.attribution_mode,
                attribution_generation=0,
                attribution_reason="no_image_observation",
            )
        self.attribution_generation += 1
        generation = self.attribution_generation
        try:
            if self.attributor is None:
                self.attributor = self._attributor_factory(self.model)
            heatmap = self.attributor.attribute(
                self.attribution_mode,
                transition.model_obs,
                transition.decision.raw_action,
            )
        except Exception as exc:
            message = f"{self.attribution_mode} attribution failed: {exc}"
            self.attribution_status = "error"
            self.attribution_error = message
            return replace(
                transition,
                attribution=None,
                attribution_status="error",
                attribution_mode=self.attribution_mode,
                attribution_generation=generation,
                attribution_reason=message,
            )
        self.attribution_last_computed_sequence = transition.sequence
        return replace(
            transition,
            attribution=np.asarray(heatmap).copy(),
            attribution_status="available",
            attribution_mode=self.attribution_mode,
            attribution_generation=generation,
            attribution_reason=None,
        )

    def configure_attribution(
        self,
        mode: str,
        interval: int | None = None,
    ) -> _PlaybackTransition | None:
        normalized = str(mode).strip().casefold()
        if normalized == "none":
            self.attribution_mode = "none"
            self.attribution_status = "off"
            self.attribution_error = None
            if self.last_transition is not None:
                self.last_transition = replace(
                    self.last_transition,
                    attribution=None,
                    attribution_status="off",
                    attribution_mode="none",
                    attribution_generation=0,
                    attribution_reason="disabled",
                )
            return self.last_transition
        if normalized not in ATTRIBUTION_MODES:
            raise ValueError(f"unknown attribution mode {mode!r}")
        supported = tuple(self.attribution_capability.get("supported_modes") or ())
        if normalized not in supported:
            reason = self.attribution_capability.get("unavailable_reason")
            raise ValueError(str(reason or f"{normalized} attribution is unavailable"))
        if isinstance(interval, bool):
            raise ValueError("attribution interval must be a positive integer")
        if isinstance(interval, float) and not interval.is_integer():
            raise ValueError("attribution interval must be a positive integer")
        resolved_interval = (
            ATTRIBUTION_INTERVAL_DEFAULTS[normalized] if interval is None else int(interval)
        )
        if resolved_interval < 1:
            raise ValueError("attribution interval must be >= 1")

        self.attribution_mode = normalized
        self.attribution_interval = resolved_interval
        self.attribution_status = "active"
        self.attribution_error = None
        try:
            if self.attributor is None:
                self.attributor = self._attributor_factory(self.model)
        except Exception as exc:
            self.attribution_status = "error"
            self.attribution_error = f"{normalized} attribution failed to initialize: {exc}"
            raise AttributionError(self.attribution_error) from exc

        if self.last_transition is not None:
            self.last_transition = self._attribution_for_transition(self.last_transition)
            if self.attribution_status == "error":
                raise AttributionError(self.attribution_error or "attribution failed")
        return self.last_transition

    def step(
        self,
        *,
        deterministic: bool = False,
        action_selection_mode: str | None = None,
    ) -> _PlaybackTransition:
        if self.policy_runtime is None:
            raise RuntimeError(f"policy runtime is unavailable: {self._policy_runtime_error}")
        requested_mode = action_selection_mode
        if requested_mode is None:
            requested_mode = "deterministic" if deterministic else None
        batch = self.policy_runtime.decide(
            self.model_obs,
            action_selection_mode=requested_mode,
            include_diagnostics=bool(self.processing_features & {"policy", "raw"}),
            execution_context=(
                self.env.policy_execution_context(self.model)
                if callable(getattr(self.env, "policy_execution_context", None))
                else None
            ),
        )
        if len(batch.decisions) != 1:
            raise RuntimeError("interactive playback requires exactly one policy decision per step")
        decision = batch.decisions[0]
        return self._advance(
            decision=decision,
            executed_action=decision.executed_action,
            action_source="policy",
        )

    def manual_action(
        self,
        labels: set[str] | tuple[str, ...] | list[str],
    ) -> int | tuple[int, ...]:
        if isinstance(getattr(self, "action_contract", None), Mapping):
            return action_value_for_controls(self.action_contract, labels)
        requested = {str(label).strip().casefold() for label in labels if str(label).strip()}
        for index, meaning in enumerate(self.action_names):
            normalized = str(meaning).strip().casefold()
            parts = (
                set()
                if normalized in {"", "noop", "no_op", "none"}
                else set(
                    part for part in normalized.split("_") if part and not part.startswith("p1")
                )
            )
            if parts == requested:
                return index
        available = ", ".join(self.action_names) or "none"
        raise ValueError(
            f"no configured discrete action matches buttons {sorted(requested)!r}; "
            f"available actions: {available}"
        )

    def step_human(self, labels: set[str] | tuple[str, ...] | list[str]) -> _PlaybackTransition:
        action = self.manual_action(labels)
        self.interactive = True
        return self._advance(
            decision=None,
            executed_action=np.asarray(action),
            action_source="human",
        )

    @staticmethod
    def _frame_tuple(frames: deque[np.ndarray] | None) -> tuple[np.ndarray, ...]:
        if frames is None:
            return ()
        return tuple(np.asarray(frame).copy() for frame in frames)

    def _advance(
        self,
        *,
        decision: PolicyDecision | None,
        executed_action: object,
        action_source: str,
    ) -> _PlaybackTransition:
        processing = self.processing_features
        needs_raw = "raw" in processing
        needs_observation = bool(processing & {"observation", "attribution", "cnn-inspection"})
        model_obs = self.model_obs
        model_obs_snapshot = (
            deepcopy(model_obs)
            if needs_raw or "attribution" in processing or "cnn-inspection" in processing
            else None
        )
        pre_task = deepcopy(self.active_task) if needs_raw else None
        before_frame = (
            None if not needs_raw or self.current_frame is None else self.current_frame.copy()
        )
        before_frames = self._frame_tuple(self.frames) if needs_observation else ()

        batched_action = np.expand_dims(np.asarray(executed_action), axis=0)
        policy_obs, rewards, dones, infos = self.env.step(batched_action)
        diagnostics = self.env.take_step_diagnostics()
        records = drain_runtime_records(self.env)
        step_metrics = batch_metrics_for_lane(records, 0)
        info: dict[str, object] = {}
        if diagnostics is not None:
            info.update(diagnostics.provider_info)
            info.update(diagnostics.task_metrics)
        info.update(dict(infos[0]))
        info.update(step_metrics)

        reward = float(np.asarray(rewards)[0])
        done = bool(np.asarray(dones)[0])
        truncated = bool(info.get("TimeLimit.truncated", False))
        terminated = done and not truncated
        completed_records = episode_records(records)
        episode_result = None
        if completed_records:
            episode_result = episode_result_from_record(
                completed_records[0],
                semantics=environment_spec(
                    self.config.env_provider,
                    self.config.game,
                ).eval_semantics,
                terminal_info=info,
            )
            terminated = bool(episode_result["terminated"])
            truncated = bool(episode_result["truncated"])

        self.total_reward += reward
        self.max_x_pos = max(self.max_x_pos, int(info.get("max_x_pos", 0)))
        final_info = info
        if episode_result is not None:
            self.total_reward = float(episode_result["return"])
            self.max_x_pos = max(
                self.max_x_pos,
                int(episode_result.get("max_x_pos", 0)),
            )
            final_info = dict(episode_result.get("final_info", {}))
            completed = bool(episode_result.get("level_complete", False))
        else:
            completed = is_level_complete(final_info)
        boundary = playback_should_end_episode(terminated, truncated, completed)

        next_conditioning_info = dict(info.get("reset_info", {})) if boundary else info
        self._update_conditioning(next_conditioning_info)
        next_task = deepcopy(self.active_task) if needs_raw else None
        self.policy_obs = policy_obs
        self.current_frame = optional_vector_env_frame(self.env)
        self.frames = optional_fast_env_frames(policy_obs)
        terminal_frame = (
            diagnostics.terminal_frame if boundary and diagnostics is not None else None
        )
        after_frame = terminal_frame if terminal_frame is not None else self.current_frame
        if boundary:
            after_frame_role = (
                "terminal_observation"
                if terminal_frame is not None
                else "next_episode_initial_observation"
            )
            after_policy_obs = info.get("terminal_observation", policy_obs)
        else:
            after_frame_role = "after_action_observation"
            after_policy_obs = policy_obs
        after_frames = optional_fast_env_frames(after_policy_obs)
        self.sequence += 1
        transition = _PlaybackTransition(
            sequence=self.sequence,
            episode=self.episode,
            step=self.step_index + 1,
            seed=self.active_seed,
            start_id=(diagnostics.start_id if diagnostics is not None else None),
            model_obs=model_obs_snapshot,
            decision=decision,
            action_source=action_source,
            executed_action=deepcopy(executed_action),
            diagnostics=diagnostics,
            info=dict(info),
            before_frame=before_frame,
            after_frame=None if after_frame is None else np.asarray(after_frame).copy(),
            before_frames=before_frames,
            after_frames=self._frame_tuple(after_frames),
            attribution=None,
            pre_task=pre_task,
            next_task=next_task,
            reward=reward,
            total_reward=self.total_reward,
            max_x_pos=self.max_x_pos,
            terminated=terminated,
            truncated=truncated,
            completed=completed,
            boundary=boundary,
            after_frame_role=after_frame_role,
            attribution_status=(
                "off"
                if self.attribution_mode == "none"
                else "error"
                if self.attribution_status == "error"
                else "not_computed"
            ),
            attribution_mode=self.attribution_mode,
            attribution_generation=(
                self.attribution_generation if self.attribution_status == "error" else 0
            ),
            attribution_reason=(
                "disabled"
                if self.attribution_mode == "none"
                else self.attribution_error
                if self.attribution_status == "error"
                else "no_policy_decision"
                if decision is None
                else "no_image_observation"
                if not before_frames
                else "cadence"
            ),
            cnn_inspection=None,
            cnn_status=(
                "off"
                if not self.cnn_enabled
                else "error"
                if self.cnn_status == "error"
                else "not_computed"
            ),
            cnn_layer_id=self.cnn_layer_id,
            cnn_generation=self.cnn_generation if self.cnn_status == "error" else 0,
            cnn_reason=(
                "disabled"
                if not self.cnn_enabled
                else self.cnn_error
                if self.cnn_status == "error"
                else "no_image_observation"
                if not before_frames
                else "cadence"
            ),
        )
        if (
            "attribution" in processing
            and self.attribution_status == "active"
            and decision is not None
            and before_frames
            and self.step_index % self.attribution_interval == 0
        ):
            transition = self._attribution_for_transition(transition)
        if (
            "cnn-inspection" in processing
            and self.cnn_status == "active"
            and before_frames
            and self.step_index % self.cnn_interval == 0
        ):
            transition = self._cnn_for_transition(transition)
        self.last_transition = transition
        if boundary:
            reset_policy_state(self.model, [True])
            self.episode += 1
            self.step_index = 0
            self.total_reward = 0.0
            self.max_x_pos = 0
            if diagnostics is not None:
                self.active_seed = diagnostics.next_episode_seed
        else:
            self.step_index += 1
        return transition
