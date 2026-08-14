"""Strict Turbo Vector API v2 adapter for selected Gymnasium environments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
import hashlib
import json
from types import MappingProxyType
from typing import Any

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, AutoresetMode
import numpy as np

from gradlab.turbo_api import CAPABILITY_KEYS, TURBO_API_VERSION


@dataclass(frozen=True)
class GymnasiumEnvContract:
    env_id: str
    entry_point: str
    observation_shape: tuple[int, ...]
    observation_dtype: str
    action_meanings: tuple[str, ...]
    action_controls: tuple[tuple[str, ...], ...]
    max_episode_steps: int
    reward_threshold: float
    render_fps: int

    def __post_init__(self) -> None:
        if len(self.action_meanings) != len(self.action_controls):
            raise ValueError(f"{self.env_id} action meanings and controls must align")

    @property
    def action_count(self) -> int:
        return len(self.action_meanings)

    @property
    def action_table_hash(self) -> str:
        button_indices = {"left": 0, "right": 1}
        masks = [
            sum(1 << button_indices[label] for label in controls)
            for controls in self.action_controls
        ]
        payload = json.dumps(masks, ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("ascii")).hexdigest()


GYMNASIUM_ENV_CONTRACTS: Mapping[str, GymnasiumEnvContract] = MappingProxyType(
    {
        "CartPole-v1": GymnasiumEnvContract(
            env_id="CartPole-v1",
            entry_point="gymnasium.envs.classic_control.cartpole:CartPoleEnv",
            observation_shape=(4,),
            observation_dtype="float32",
            action_meanings=("push_left", "push_right"),
            action_controls=(("left",), ("right",)),
            max_episode_steps=500,
            reward_threshold=475.0,
            render_fps=50,
        ),
        "MountainCar-v0": GymnasiumEnvContract(
            env_id="MountainCar-v0",
            entry_point="gymnasium.envs.classic_control.mountain_car:MountainCarEnv",
            observation_shape=(2,),
            observation_dtype="float32",
            action_meanings=("accelerate_left", "coast", "accelerate_right"),
            action_controls=(("left",), (), ("right",)),
            max_episode_steps=200,
            reward_threshold=-110.0,
            render_fps=30,
        ),
        "Acrobot-v1": GymnasiumEnvContract(
            env_id="Acrobot-v1",
            entry_point="gymnasium.envs.classic_control.acrobot:AcrobotEnv",
            observation_shape=(6,),
            observation_dtype="float32",
            action_meanings=("torque_negative", "torque_zero", "torque_positive"),
            action_controls=(("left",), (), ("right",)),
            max_episode_steps=500,
            reward_threshold=-100.0,
            render_fps=15,
        ),
    }
)

GYMNASIUM_ENV_IDS = tuple(GYMNASIUM_ENV_CONTRACTS)


def _make_scalar_env(env_id: str):
    """Create one scalar lane from a spawn-pickleable module-level callable."""

    return gym.make(env_id, render_mode="rgb_array")


def _disabled_autoreset(value: object) -> bool:
    name = getattr(value, "name", value)
    return str(name).split(".")[-1].strip().casefold() == "disabled"


def _validate_registered_contract(contract: GymnasiumEnvContract) -> None:
    spec = gym.spec(contract.env_id)
    if spec.entry_point != contract.entry_point:
        raise RuntimeError(
            f"{contract.env_id} entry point drifted: {spec.entry_point!r}; "
            f"expected {contract.entry_point!r}"
        )
    if dict(spec.kwargs) != {}:
        raise RuntimeError(f"{contract.env_id} must retain empty registered kwargs")
    if spec.max_episode_steps != contract.max_episode_steps:
        raise RuntimeError(
            f"{contract.env_id} horizon drifted: {spec.max_episode_steps!r}; "
            f"expected {contract.max_episode_steps}"
        )
    if spec.reward_threshold != contract.reward_threshold:
        raise RuntimeError(
            f"{contract.env_id} reward threshold drifted: {spec.reward_threshold!r}; "
            f"expected {contract.reward_threshold}"
        )


def _capabilities() -> Mapping[str, Any]:
    values: dict[str, Any] = {
        "supported_action_modes": ("discrete",),
        "supported_observation_layouts": ("flat",),
        "supported_observation_color_modes": (),
        "supported_resize_algorithms": (),
        "supported_crop_modes": (),
        "supported_observation_copy_modes": ("owned",),
        "supported_transition_transports": ("numpy",),
        "supports_async_step": True,
        "supports_branching": False,
        "supports_device_api": False,
        "supports_emulator_ram": False,
        "supports_enemy_variants": False,
        "supports_fire_reset": False,
        "supports_info_frame_stack": False,
        "supports_live_snapshots": False,
        "supports_maxpool_last_two": False,
        "supports_noop_reset": False,
        "supports_per_lane_rgb": True,
        "supports_reward_clipping": False,
        "supports_snapshot_codec": False,
        "supports_state_catalog": False,
        "supports_sticky_action_prob": False,
        "supports_surface_variants": False,
    }
    return MappingProxyType({name: values[name] for name in CAPABILITY_KEYS})


class GymnasiumTurboVecEnv:
    """Expose strict Turbo v2 semantics over Gymnasium's process vectorizer."""

    def __init__(
        self,
        game: str,
        num_envs: int,
        *,
        autoreset_mode: object,
        vectorization_mode: str,
        multiprocessing_context: str,
        shared_memory: bool,
        copy: bool,
        daemon: bool,
        observation_mode: str,
        render_mode: str,
    ):
        try:
            contract = GYMNASIUM_ENV_CONTRACTS[str(game)]
        except KeyError as exc:
            known = ", ".join(GYMNASIUM_ENV_IDS)
            raise ValueError(f"unsupported Gymnasium environment {game!r}; expected {known}") from exc
        if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs < 1:
            raise ValueError("Gymnasium num_envs must be a positive integer")
        required = {
            "autoreset_mode": _disabled_autoreset(autoreset_mode),
            "vectorization_mode": vectorization_mode == "async",
            "multiprocessing_context": multiprocessing_context == "spawn",
            "shared_memory": shared_memory is True,
            "copy": copy is True,
            "daemon": daemon is True,
            "observation_mode": observation_mode == "same",
            "render_mode": render_mode == "rgb_array",
        }
        invalid = sorted(name for name, valid in required.items() if not valid)
        if invalid:
            raise ValueError(
                "Gymnasium adapter requires fixed execution value(s): " + ", ".join(invalid)
            )
        _validate_registered_contract(contract)

        self.contract = contract
        self.num_envs = num_envs
        self.num_threads = num_envs
        self.autoreset_mode = AutoresetMode.DISABLED
        self.frame_skip = 1
        self.frame_stack = 1
        self.obs_layout = "flat"
        self.obs_copy = "owned"
        self.render_mode = "rgb_array"
        self.transport = "numpy"
        self.closed = False
        self.buttons = ("left", "right")
        self.action_mode = "discrete"
        self.action_preset = None
        self.action_table = contract.action_controls
        self.action_meanings = contract.action_meanings
        self.action_table_hash = contract.action_table_hash
        self.capabilities = _capabilities()
        self.observation_ownership = "owned"
        self.observation_buffer_depth = None
        self.state_catalog: tuple[str, ...] = ()
        self.signal_schema: Mapping[str, Any] = MappingProxyType({})
        self.supports_live_snapshots = False
        self.live_snapshots_deterministic = False
        self.metadata = {
            "turbo_api_version": TURBO_API_VERSION,
            "transition_transport": "numpy",
            "autoreset_mode": AutoresetMode.DISABLED,
            "render_modes": ("rgb_array",),
            "render_fps": contract.render_fps,
            "vectorization_mode": "async",
            "multiprocessing_context": "spawn",
        }
        active = np.full(num_envs, -1, dtype=np.int32)
        active.flags.writeable = False
        self._active_state_indices = active
        self._pending_reset = np.zeros(num_envs, dtype=np.bool_)
        self._async_pending = False
        self._observations: np.ndarray | None = None
        self._env: AsyncVectorEnv | None = None

        try:
            self._env = AsyncVectorEnv(
                tuple(partial(_make_scalar_env, contract.env_id) for _ in range(num_envs)),
                shared_memory=True,
                copy=True,
                context="spawn",
                daemon=True,
                observation_mode="same",
                autoreset_mode=AutoresetMode.DISABLED,
            )
            self.single_observation_space = self._env.single_observation_space
            self.single_action_space = self._env.single_action_space
            self.observation_space = self._env.observation_space
            self.action_space = self._env.action_space
            self._validate_spaces()
        except BaseException:
            if self._env is not None:
                self._env.close(terminate=True)
            self.closed = True
            raise

    def _validate_spaces(self) -> None:
        observation_space = self.single_observation_space
        if not isinstance(observation_space, gym.spaces.Box):
            raise TypeError(f"{self.contract.env_id} must expose a Box observation space")
        if observation_space.shape != self.contract.observation_shape:
            raise TypeError(
                f"{self.contract.env_id} observation shape {observation_space.shape} does not "
                f"match {self.contract.observation_shape}"
            )
        if observation_space.dtype != np.dtype(self.contract.observation_dtype):
            raise TypeError(
                f"{self.contract.env_id} observation dtype {observation_space.dtype} does not "
                f"match {self.contract.observation_dtype}"
            )
        action_space = self.single_action_space
        if not isinstance(action_space, gym.spaces.Discrete):
            raise TypeError(f"{self.contract.env_id} must expose a Discrete action space")
        if int(action_space.start) != 0 or int(action_space.n) != self.contract.action_count:
            raise TypeError(
                f"{self.contract.env_id} action space {action_space} does not match "
                f"zero-based Discrete({self.contract.action_count})"
            )

    def _require_open(self) -> AsyncVectorEnv:
        if self.closed or self._env is None:
            raise RuntimeError("Gymnasium vector environment is closed")
        return self._env

    def _require_idle(self) -> AsyncVectorEnv:
        env = self._require_open()
        if self._async_pending:
            raise RuntimeError("Gymnasium vector environment has an in-flight async step")
        return env

    def active_state_indices(self) -> np.ndarray:
        self._require_open()
        return self._active_state_indices

    def _reset_mask(self, options: Mapping[str, Any] | None) -> np.ndarray:
        reset_options = dict(options or {})
        unexpected = sorted(set(reset_options) - {"reset_mask"})
        if unexpected:
            raise ValueError(f"Gymnasium reset has unsupported options: {unexpected}")
        raw_mask = reset_options.get("reset_mask")
        if raw_mask is None:
            mask = np.ones(self.num_envs, dtype=np.bool_)
        else:
            mask = np.asarray(raw_mask)
            if mask.dtype != np.dtype(np.bool_):
                raise TypeError("reset_mask must have boolean dtype")
            if mask.shape != (self.num_envs,):
                raise ValueError(f"reset_mask must have shape ({self.num_envs},)")
            mask = mask.copy()
        if not np.any(mask):
            raise ValueError("reset_mask must select at least one lane")
        if self._observations is None and not np.all(mask):
            raise ValueError("the first Gymnasium reset must select every lane")
        return mask

    def _reset_seeds(self, seed: object, mask: np.ndarray) -> list[int | None]:
        if isinstance(seed, np.ndarray):
            values: Sequence[object] = seed.tolist()
        elif isinstance(seed, list | tuple):
            values = seed
        else:
            raise TypeError("Gymnasium reset seed must contain one lane seed per environment")
        if len(values) != self.num_envs:
            raise ValueError(f"Gymnasium reset seed must contain {self.num_envs} values")
        normalized: list[int | None] = []
        for lane, value in enumerate(values):
            if not bool(mask[lane]):
                if value is not None:
                    raise ValueError("unselected reset lanes must use seed=None")
                normalized.append(None)
                continue
            if not isinstance(value, int | np.integer) or isinstance(value, bool):
                raise TypeError(f"selected reset lane {lane} requires an integer seed")
            normalized.append(int(value))
        return normalized

    @staticmethod
    def _require_empty_infos(infos: object, *, operation: str) -> None:
        if not isinstance(infos, Mapping):
            raise TypeError(f"Gymnasium {operation} infos must be a columnar mapping")
        if infos:
            raise RuntimeError(
                f"Gymnasium {operation} emitted undeclared info keys: {sorted(infos)}"
            )

    def reset(self, *, seed=None, options=None):
        env = self._require_idle()
        mask = self._reset_mask(options)
        seeds = self._reset_seeds(seed, mask)
        observations, infos = env.reset(seed=seeds, options={"reset_mask": mask})
        self._require_empty_infos(infos, operation="reset")
        owned = np.asarray(observations).copy()
        if owned.shape[:1] != (self.num_envs,):
            raise ValueError("Gymnasium reset observations must contain every lane")
        if self._observations is not None:
            owned[~mask] = self._observations[~mask]
        self._observations = owned.copy()
        self._pending_reset[mask] = False
        return owned, {
            "state_index": np.full(self.num_envs, -1, dtype=np.int32),
            "_state_index": mask.copy(),
            "start_source": np.zeros(self.num_envs, dtype=np.int8),
            "_start_source": mask.copy(),
            "noop_reset_count": np.zeros(self.num_envs, dtype=np.int64),
            "_noop_reset_count": mask.copy(),
        }

    def _actions(self, actions: object) -> np.ndarray:
        values = np.asarray(actions)
        if values.shape != (self.num_envs,):
            raise ValueError(f"actions must have shape ({self.num_envs},)")
        if not np.issubdtype(values.dtype, np.integer):
            raise TypeError("Gymnasium discrete actions must have integer dtype")
        normalized = values.astype(np.int64, copy=True)
        if np.any(normalized < 0) or np.any(normalized >= self.contract.action_count):
            raise ValueError("Gymnasium action is outside the registered Discrete space")
        return normalized

    def step_async(self, actions: object) -> None:
        env = self._require_idle()
        if np.any(self._pending_reset):
            lanes = np.flatnonzero(self._pending_reset).tolist()
            raise RuntimeError(f"Gymnasium done lanes must be explicitly reset: {lanes}")
        env.step_async(self._actions(actions))
        self._async_pending = True

    def step_wait(self, timeout: int | float | None = None):
        env = self._require_open()
        if not self._async_pending:
            raise RuntimeError("Gymnasium step_wait requires an in-flight async step")
        try:
            observations, rewards, terminated, truncated, infos = env.step_wait(timeout=timeout)
        finally:
            self._async_pending = False
        self._require_empty_infos(infos, operation="step")
        owned_observations = np.asarray(observations).copy()
        owned_rewards = np.asarray(rewards).copy()
        owned_terminated = np.asarray(terminated, dtype=np.bool_).copy()
        owned_truncated = np.asarray(truncated, dtype=np.bool_).copy()
        expected = (self.num_envs,)
        if owned_observations.shape[:1] != expected:
            raise ValueError("Gymnasium step observations must contain every lane")
        for name, values in (
            ("rewards", owned_rewards),
            ("terminated", owned_terminated),
            ("truncated", owned_truncated),
        ):
            if values.shape != expected:
                raise ValueError(f"Gymnasium step {name} must have shape {expected}")
        self._observations = owned_observations.copy()
        self._pending_reset |= owned_terminated | owned_truncated
        return (
            owned_observations,
            owned_rewards,
            owned_terminated,
            owned_truncated,
            {},
        )

    def step(self, actions: object):
        self.step_async(actions)
        return self.step_wait()

    def get_images(self) -> list[np.ndarray]:
        env = self._require_idle()
        if self._observations is None:
            raise RuntimeError("Gymnasium rendering requires reset first")
        frames = env.call("render")
        if len(frames) != self.num_envs:
            raise ValueError("Gymnasium render must return one frame per lane")
        result: list[np.ndarray] = []
        for lane, frame in enumerate(frames):
            image = np.asarray(frame)
            if image.dtype != np.dtype(np.uint8) or image.ndim != 3 or image.shape[-1] != 3:
                raise TypeError(f"Gymnasium lane {lane} did not render an HWC uint8 RGB frame")
            result.append(image.copy())
        return result

    def render_lane(self, lane: int) -> np.ndarray:
        if not isinstance(lane, int) or isinstance(lane, bool) or not 0 <= lane < self.num_envs:
            raise IndexError(f"Gymnasium render lane must be in [0, {self.num_envs})")
        return self.get_images()[lane]

    def render(self):
        return self.get_images()

    def close(self) -> None:
        if self.closed:
            return
        env = self._env
        self._env = None
        self.closed = True
        if env is not None:
            env.close(terminate=self._async_pending)
        self._async_pending = False


__all__ = [
    "GYMNASIUM_ENV_CONTRACTS",
    "GYMNASIUM_ENV_IDS",
    "GymnasiumEnvContract",
    "GymnasiumTurboVecEnv",
]
