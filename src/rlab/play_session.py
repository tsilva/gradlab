from __future__ import annotations

# ruff: noqa: E402

import argparse
import os
from collections import deque
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import torch
from rlab.action_contract import configured_action_meanings, configured_action_name
from rlab.batch_runtime import StepDiagnostics
from rlab.env import (
    info_value_from_state_name,
    state_name_candidates_from_level_id,
    task_conditioning,
    task_max_episode_steps,
    task_reward,
    task_termination,
)
from rlab.eval_metrics import (
    batch_metrics_for_lane,
    drain_runtime_records,
    episode_records,
    episode_result_from_record,
    is_level_complete,
)
from rlab.model_sources import (
    DEFAULT_PUBLIC_MODELS_BASE_URL,
    positional_model_source_arg,
)
from rlab.play_attribution import PolicyActionAttributor
from rlab.play_debug import (
    PolicyDecision,
    inspect_policy,
    sample_policy_decision,
)
from rlab.policy_observation import (
    model_observation,
    task_info_value_from_info,
    task_info_vars,
    task_state_names,
)
from rlab.policy_runtime import reset_policy_state
from rlab.play_termination import (
    configured_termination_ids,
    termination_condition_payload,
    with_enabled_termination_conditions,
)
from rlab.seeds import DEFAULT_EVAL_SEED, EVAL_SEED_START
from rlab.targets import target_for_game


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
ATTRIBUTION_MODES = ("none", "gradcam", "occlusion")


def _color(text: str, style: str) -> str:
    if os.environ.get("NO_COLOR") or os.environ.get("RLAB_NO_COLOR"):
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
        if "image" not in obs:
            raise ValueError(f"dict fast env obs is missing 'image'; keys={tuple(obs)}")
        obs = obs["image"]
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
    heatmap: np.ndarray | None = None,
    heatmap_opacity: float = 0.45,
) -> np.ndarray:
    if scale < 1:
        raise ValueError("obs viewer scale must be >= 1")
    scaled_heatmap = None
    if heatmap is not None:
        scaled_heatmap = np.asarray(heatmap, dtype=np.float32)
        if scaled_heatmap.ndim != 2:
            raise ValueError(f"attribution heatmap must be 2D, got shape {scaled_heatmap.shape}")
        if scale != 1:
            scaled_heatmap = np.repeat(np.repeat(scaled_heatmap, scale, axis=0), scale, axis=1)
        scaled_heatmap = np.clip(scaled_heatmap, 0.0, 1.0)
        heat_color = _heatmap_color(scaled_heatmap)
        alpha = (float(heatmap_opacity) * scaled_heatmap)[..., None]
    panels = []
    for frame in frames:
        gray = frame[..., 0]
        panel = np.repeat(gray[..., None], 3, axis=2)
        if scale != 1:
            panel = np.repeat(np.repeat(panel, scale, axis=0), scale, axis=1)
        if scaled_heatmap is not None:
            if scaled_heatmap.shape != panel.shape[:2]:
                raise ValueError(
                    "attribution heatmap shape does not match observation frame: "
                    f"{scaled_heatmap.shape} vs {panel.shape[:2]}"
                )
            panel = ((1.0 - alpha) * panel.astype(np.float32) + alpha * heat_color).astype(np.uint8)
        panels.append(panel)
    return np.concatenate(panels, axis=1)


def add_play_source_args(parser: argparse.ArgumentParser) -> None:
    def positional_play_source_arg(value: str) -> str:
        from rlab.play_catalog import is_wandb_url

        if is_wandb_url(value):
            return value
        return positional_model_source_arg(value)

    parser.add_argument(
        "artifact_ref",
        nargs="?",
        type=positional_play_source_arg,
        help=(
            "W&B project/run URL, immutable public checkpoint manifest, or Hugging Face model ref. "
            "Use --model for a local checkpoint or --run for an rlab public run."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Local rlab policy path. The artifact must have model.json and recipe.json sidecars.",
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
        default=Path("runs"),
        help="Local run root searched by --recipe; defaults to ./runs.",
    )
    parser.add_argument(
        "--run",
        help=(
            "Immutable rlab run ID. Resolves its public promoted checkpoint without "
            "W&B or private R2 credentials."
        ),
    )
    parser.add_argument(
        "--public-models-base-url",
        default=DEFAULT_PUBLIC_MODELS_BASE_URL,
        help="Public models bucket URL. Defaults to rlab's checked-in public endpoint.",
    )
    parser.add_argument(
        "--wandb-entity",
        help="W&B entity containing repository-declared runs. Defaults to WANDB_ENTITY.",
    )
    parser.add_argument(
        "--public-model-root",
        default="runs/public_models",
        help="Local cache for public run checkpoints.",
    )
    parser.set_defaults(
        hf_revision=None,
        hf_model_root="runs/hf_models",
    )


def positive_int_arg(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def nonnegative_int_arg(value: str) -> int:
    parsed = int(value)
    if parsed < 0 or parsed > 65535:
        raise argparse.ArgumentTypeError("must be in [0, 65535]")
    return parsed


def attribution_opacity_arg(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be in [0, 1]")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlab play",
        description=(
            "Browse W&B runs and public checkpoints, then inspect a policy in the "
            "interactive web player"
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
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
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
        "--no-progress",
        action="store_true",
        help="Disable model-download and player-startup progress bars.",
    )
    parser.add_argument(
        "--attribution",
        choices=ATTRIBUTION_MODES,
        default="none",
        help=(
            "Overlay policy-input attribution in the observation panel. "
            "Grad-CAM is fast; occlusion is slower but perturbation-based."
        ),
    )
    parser.add_argument(
        "--attribution-interval",
        type=positive_int_arg,
        default=None,
        help=(
            "Compute attribution every N policy steps. Defaults to 1 for Grad-CAM "
            "and 8 for occlusion."
        ),
    )
    parser.add_argument(
        "--attribution-opacity",
        type=attribution_opacity_arg,
        default=0.45,
        help="Heatmap opacity for --attribution overlays, in [0, 1].",
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
            "interface=web "
            f"respect_task_termination={getattr(args, 'respect_task_termination', False)}",
            "green",
        ),
        _summary_line(
            "◎",
            "attribution",
            f"mode={getattr(args, 'attribution', 'none')} "
            f"interval={getattr(args, 'attribution_interval', None) or '-'} "
            f"opacity={getattr(args, 'attribution_opacity', 0.45):.2f}",
            "magenta",
        ),
        _summary_line(
            "▤",
            "preprocessing",
            f"frame_skip={policy_config.frame_skip} max_pool={policy_config.max_pool_frames} "
            f"sticky={policy_config.sticky_action_prob} "
            f"obs={policy_config.observation_size} crop={_format_sequence(policy_config.obs_crop)} "
            f"crop_mode={policy_config.obs_crop_mode} crop_fill={policy_config.obs_crop_fill} "
            f"crop_top={policy_config.hud_crop_top} "
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
        attributor: PolicyActionAttributor | None,
        attribution_mode: str,
        attribution_interval: int,
        attribution_opacity: float,
        env_factory=None,
        termination_base_config=None,
        termination_source: str = "training",
    ):
        self.model = model
        self.env = env
        self.config = config
        self.initial_seed = initial_seed
        self.attributor = attributor
        self.attribution_mode = attribution_mode
        self.attribution_interval = attribution_interval
        self.attribution_opacity = attribution_opacity
        self.env_factory = env_factory
        self.termination_base_config = termination_base_config or config
        self.termination_source = termination_source
        self.info_vars = task_info_vars(config)
        self.conditioning_enabled = bool(task_conditioning(config).get("enabled"))
        self.configured_task_states = task_state_names(config) if self.conditioning_enabled else ()
        try:
            self.action_names = configured_action_meanings(config)
        except ValueError:
            self.action_names = ()
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
            from rlab.policy_runtime import bind_policy_action_space

            bind_policy_action_space(self.model, next_env.action_space)
            self.env = next_env
            self.config = next_config
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
        candidate = info.get("start_id") or info.get("start_state") or info.get("state")
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

    def reset_episode(self, seed: int | None = None) -> None:
        """Abandon the active trajectory and return to step zero."""

        if self.step_index > 0:
            self.episode += 1
        active_seed = self.active_seed if seed is None else seed
        self.restart(active_seed, reset_episode_index=False)

    def step(self, *, deterministic: bool = False) -> _PlaybackTransition:
        decision = (
            inspect_policy(self.model, self.model_obs)
            if deterministic
            else sample_policy_decision(self.model, self.model_obs)
        )
        return self._advance(
            decision=decision,
            executed_action=decision.executed_action,
            action_source="policy",
        )

    def manual_action(self, labels: set[str] | tuple[str, ...] | list[str]) -> int:
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
        model_obs = self.model_obs
        model_obs_snapshot = deepcopy(model_obs)
        pre_task = deepcopy(self.active_task)
        before_frame = None if self.current_frame is None else self.current_frame.copy()
        before_frames = self._frame_tuple(self.frames)
        heatmap = None
        if (
            decision is not None
            and self.attributor is not None
            and self.frames is not None
            and self.step_index % self.attribution_interval == 0
        ):
            heatmap = self.attributor.attribute(
                self.attribution_mode,
                model_obs,
                decision.raw_action,
            )

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
                semantics=target_for_game(self.config.game).eval_semantics,
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
        next_task = deepcopy(self.active_task)
        self.policy_obs = policy_obs
        self.current_frame = optional_vector_env_frame(self.env)
        self.frames = optional_fast_env_frames(policy_obs)
        terminal_frame = (
            diagnostics.terminal_frame
            if boundary and diagnostics is not None
            else None
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
            attribution=None if heatmap is None else np.asarray(heatmap).copy(),
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
        )
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

