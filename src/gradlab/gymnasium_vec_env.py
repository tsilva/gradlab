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
class GymnasiumSignalContract:
    name: str
    dtype: str
    shape: tuple[int, ...] = ()
    available_on_reset: bool = True
    available_on_step: bool = True

    @property
    def schema(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "dtype": self.dtype,
                "shape": self.shape,
                "available_on_reset": self.available_on_reset,
                "available_on_step": self.available_on_step,
            }
        )


@dataclass(frozen=True)
class GymnasiumEnvContract:
    env_id: str
    entry_point: str
    registered_kwargs: tuple[tuple[str, Any], ...]
    observation_kind: str
    observation_shape: tuple[int, ...]
    observation_dtype: str
    observation_categories: tuple[int, ...]
    buttons: tuple[str, ...]
    action_meanings: tuple[str, ...]
    action_controls: tuple[tuple[str, ...], ...]
    registered_max_episode_steps: int | None
    goal_max_episode_steps: int
    reward_threshold: float | None
    render_fps: int
    signals: tuple[GymnasiumSignalContract, ...] = ()

    def __post_init__(self) -> None:
        if len(self.action_meanings) != len(self.action_controls):
            raise ValueError(f"{self.env_id} action meanings and controls must align")
        if self.observation_kind not in {"box", "discrete", "multi_discrete"}:
            raise ValueError(f"{self.env_id} has unsupported observation kind")
        if len(set(self.buttons)) != len(self.buttons):
            raise ValueError(f"{self.env_id} button labels must be unique")
        unknown_controls = {
            control for controls in self.action_controls for control in controls
        } - set(self.buttons)
        if unknown_controls:
            raise ValueError(f"{self.env_id} has unknown controls: {sorted(unknown_controls)}")
        if len({signal.name for signal in self.signals}) != len(self.signals):
            raise ValueError(f"{self.env_id} signal names must be unique")

    @property
    def action_count(self) -> int:
        return len(self.action_meanings)

    @property
    def max_episode_steps(self) -> int:
        """Return the finite GradLab task horizon for this environment."""

        return self.goal_max_episode_steps

    @property
    def action_table_hash(self) -> str:
        button_indices = {label: index for index, label in enumerate(self.buttons)}
        masks = [
            sum(1 << button_indices[label] for label in controls)
            for controls in self.action_controls
        ]
        payload = json.dumps(masks, ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("ascii")).hexdigest()

    @property
    def signal_schema(self) -> Mapping[str, Any]:
        return MappingProxyType({signal.name: signal.schema for signal in self.signals})


GYMNASIUM_ENV_CONTRACTS: Mapping[str, GymnasiumEnvContract] = MappingProxyType(
    {
        "CartPole-v1": GymnasiumEnvContract(
            env_id="CartPole-v1",
            entry_point="gymnasium.envs.classic_control.cartpole:CartPoleEnv",
            registered_kwargs=(),
            observation_kind="box",
            observation_shape=(4,),
            observation_dtype="float32",
            observation_categories=(),
            buttons=("left", "right"),
            action_meanings=("push_left", "push_right"),
            action_controls=(("left",), ("right",)),
            registered_max_episode_steps=500,
            goal_max_episode_steps=500,
            reward_threshold=475.0,
            render_fps=50,
        ),
        "MountainCar-v0": GymnasiumEnvContract(
            env_id="MountainCar-v0",
            entry_point="gymnasium.envs.classic_control.mountain_car:MountainCarEnv",
            registered_kwargs=(),
            observation_kind="box",
            observation_shape=(2,),
            observation_dtype="float32",
            observation_categories=(),
            buttons=("left", "right"),
            action_meanings=("accelerate_left", "coast", "accelerate_right"),
            action_controls=(("left",), (), ("right",)),
            registered_max_episode_steps=200,
            goal_max_episode_steps=200,
            reward_threshold=-110.0,
            render_fps=30,
        ),
        "Acrobot-v1": GymnasiumEnvContract(
            env_id="Acrobot-v1",
            entry_point="gymnasium.envs.classic_control.acrobot:AcrobotEnv",
            registered_kwargs=(),
            observation_kind="box",
            observation_shape=(6,),
            observation_dtype="float32",
            observation_categories=(),
            buttons=("left", "right"),
            action_meanings=("torque_negative", "torque_zero", "torque_positive"),
            action_controls=(("left",), (), ("right",)),
            registered_max_episode_steps=500,
            goal_max_episode_steps=500,
            reward_threshold=-100.0,
            render_fps=15,
        ),
        "LunarLander-v3": GymnasiumEnvContract(
            env_id="LunarLander-v3",
            entry_point="gymnasium.envs.box2d.lunar_lander:LunarLander",
            registered_kwargs=(),
            observation_kind="box",
            observation_shape=(8,),
            observation_dtype="float32",
            observation_categories=(),
            buttons=("left_engine", "main_engine", "right_engine"),
            action_meanings=(
                "noop",
                "fire_left_orientation_engine",
                "fire_main_engine",
                "fire_right_orientation_engine",
            ),
            action_controls=((), ("left_engine",), ("main_engine",), ("right_engine",)),
            registered_max_episode_steps=1000,
            goal_max_episode_steps=1000,
            reward_threshold=200.0,
            render_fps=50,
        ),
        "FrozenLake-v1": GymnasiumEnvContract(
            env_id="FrozenLake-v1",
            entry_point="gymnasium.envs.toy_text.frozen_lake:FrozenLakeEnv",
            registered_kwargs=(("map_name", "4x4"),),
            observation_kind="discrete",
            observation_shape=(),
            observation_dtype="int64",
            observation_categories=(16,),
            buttons=("left", "down", "right", "up"),
            action_meanings=("move_left", "move_down", "move_right", "move_up"),
            action_controls=(("left",), ("down",), ("right",), ("up",)),
            registered_max_episode_steps=100,
            goal_max_episode_steps=100,
            reward_threshold=0.7,
            render_fps=4,
            signals=(GymnasiumSignalContract("prob", "float64"),),
        ),
        "FrozenLake8x8-v1": GymnasiumEnvContract(
            env_id="FrozenLake8x8-v1",
            entry_point="gymnasium.envs.toy_text.frozen_lake:FrozenLakeEnv",
            registered_kwargs=(("map_name", "8x8"),),
            observation_kind="discrete",
            observation_shape=(),
            observation_dtype="int64",
            observation_categories=(64,),
            buttons=("left", "down", "right", "up"),
            action_meanings=("move_left", "move_down", "move_right", "move_up"),
            action_controls=(("left",), ("down",), ("right",), ("up",)),
            registered_max_episode_steps=200,
            goal_max_episode_steps=200,
            reward_threshold=0.85,
            render_fps=4,
            signals=(GymnasiumSignalContract("prob", "float64"),),
        ),
        "CliffWalking-v1": GymnasiumEnvContract(
            env_id="CliffWalking-v1",
            entry_point="gymnasium.envs.toy_text.cliffwalking:CliffWalkingEnv",
            registered_kwargs=(),
            observation_kind="discrete",
            observation_shape=(),
            observation_dtype="int64",
            observation_categories=(48,),
            buttons=("up", "right", "down", "left"),
            action_meanings=("move_up", "move_right", "move_down", "move_left"),
            action_controls=(("up",), ("right",), ("down",), ("left",)),
            registered_max_episode_steps=None,
            goal_max_episode_steps=200,
            reward_threshold=None,
            render_fps=4,
            signals=(GymnasiumSignalContract("prob", "float64"),),
        ),
        "CliffWalkingSlippery-v1": GymnasiumEnvContract(
            env_id="CliffWalkingSlippery-v1",
            entry_point="gymnasium.envs.toy_text.cliffwalking:CliffWalkingEnv",
            registered_kwargs=(("is_slippery", True),),
            observation_kind="discrete",
            observation_shape=(),
            observation_dtype="int64",
            observation_categories=(48,),
            buttons=("up", "right", "down", "left"),
            action_meanings=("move_up", "move_right", "move_down", "move_left"),
            action_controls=(("up",), ("right",), ("down",), ("left",)),
            registered_max_episode_steps=None,
            goal_max_episode_steps=200,
            reward_threshold=None,
            render_fps=4,
            signals=(GymnasiumSignalContract("prob", "float64"),),
        ),
        "Taxi-v3": GymnasiumEnvContract(
            env_id="Taxi-v3",
            entry_point="gymnasium.envs.toy_text.taxi:TaxiEnv",
            registered_kwargs=(),
            observation_kind="discrete",
            observation_shape=(),
            observation_dtype="int64",
            observation_categories=(500,),
            buttons=("south", "north", "east", "west", "pickup", "dropoff"),
            action_meanings=(
                "move_south",
                "move_north",
                "move_east",
                "move_west",
                "pickup",
                "dropoff",
            ),
            action_controls=(
                ("south",),
                ("north",),
                ("east",),
                ("west",),
                ("pickup",),
                ("dropoff",),
            ),
            registered_max_episode_steps=200,
            goal_max_episode_steps=200,
            reward_threshold=8.0,
            render_fps=4,
            signals=(
                GymnasiumSignalContract("prob", "float64"),
                GymnasiumSignalContract("action_mask", "int8", (6,)),
            ),
        ),
        "Blackjack-v1": GymnasiumEnvContract(
            env_id="Blackjack-v1",
            entry_point="gymnasium.envs.toy_text.blackjack:BlackjackEnv",
            registered_kwargs=(("sab", True), ("natural", False)),
            observation_kind="multi_discrete",
            observation_shape=(3,),
            observation_dtype="int64",
            observation_categories=(32, 11, 2),
            buttons=("stick", "hit"),
            action_meanings=("stick", "hit"),
            action_controls=(("stick",), ("hit",)),
            registered_max_episode_steps=None,
            goal_max_episode_steps=100,
            reward_threshold=None,
            render_fps=4,
        ),
    }
)

GYMNASIUM_ENV_IDS = tuple(GYMNASIUM_ENV_CONTRACTS)


class _TupleObservationAdapter(gym.ObservationWrapper):
    """Apply the fixed Blackjack tuple-to-MultiDiscrete observation contract."""

    def __init__(self, env: gym.Env, categories: tuple[int, ...]):
        super().__init__(env)
        native = env.observation_space
        if not isinstance(native, gym.spaces.Tuple) or len(native.spaces) != len(categories):
            raise TypeError("structured Gymnasium observation must be the declared tuple")
        for index, (space, count) in enumerate(zip(native.spaces, categories, strict=True)):
            if not isinstance(space, gym.spaces.Discrete) or space.start != 0 or space.n != count:
                raise TypeError(
                    f"structured Gymnasium observation element {index} must be Discrete({count})"
                )
        self.observation_space = gym.spaces.MultiDiscrete(
            np.asarray(categories, dtype=np.int64),
            dtype=np.int64,
        )

    def observation(self, observation: object) -> np.ndarray:
        encoded = np.asarray(observation, dtype=np.int64)
        if encoded.shape != self.observation_space.shape or not self.observation_space.contains(
            encoded
        ):
            raise ValueError("structured Gymnasium observation violates its declared encoding")
        return encoded.copy()


def _make_scalar_env(env_id: str, observation_kind: str, categories: tuple[int, ...]):
    """Create one scalar lane from a spawn-pickleable module-level callable."""

    env = gym.make(env_id, render_mode="rgb_array")
    if observation_kind == "multi_discrete":
        return _TupleObservationAdapter(env, categories)
    return env


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
    expected_kwargs = dict(contract.registered_kwargs)
    if dict(spec.kwargs) != expected_kwargs:
        raise RuntimeError(
            f"{contract.env_id} registered kwargs drifted: {dict(spec.kwargs)!r}; "
            f"expected {expected_kwargs!r}"
        )
    if spec.max_episode_steps != contract.registered_max_episode_steps:
        raise RuntimeError(
            f"{contract.env_id} horizon drifted: {spec.max_episode_steps!r}; "
            f"expected {contract.registered_max_episode_steps!r}"
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
        self.buttons = contract.buttons
        self.action_mode = "discrete"
        self.action_preset = None
        self.action_table = contract.action_controls
        self.action_meanings = contract.action_meanings
        self.action_table_hash = contract.action_table_hash
        self.capabilities = _capabilities()
        self.observation_ownership = "owned"
        self.observation_buffer_depth = None
        self.state_catalog: tuple[str, ...] = ()
        self.signal_schema = contract.signal_schema
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
                tuple(
                    partial(
                        _make_scalar_env,
                        contract.env_id,
                        contract.observation_kind,
                        contract.observation_categories,
                    )
                    for _ in range(num_envs)
                ),
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
        contract = self.contract
        if contract.observation_kind == "box":
            valid_kind = isinstance(observation_space, gym.spaces.Box)
        elif contract.observation_kind == "discrete":
            valid_kind = (
                isinstance(observation_space, gym.spaces.Discrete)
                and int(observation_space.start) == 0
                and int(observation_space.n) == contract.observation_categories[0]
            )
        else:
            valid_kind = (
                isinstance(observation_space, gym.spaces.MultiDiscrete)
                and bool(np.all(observation_space.start == 0))
                and tuple(int(value) for value in observation_space.nvec)
                == contract.observation_categories
            )
        if not valid_kind:
            raise TypeError(
                f"{contract.env_id} observation space {observation_space} does not match "
                f"the declared {contract.observation_kind} contract"
            )
        if observation_space.shape != contract.observation_shape:
            raise TypeError(
                f"{contract.env_id} observation shape {observation_space.shape} does not "
                f"match {contract.observation_shape}"
            )
        if observation_space.dtype != np.dtype(contract.observation_dtype):
            raise TypeError(
                f"{contract.env_id} observation dtype {observation_space.dtype} does not "
                f"match {contract.observation_dtype}"
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

    def _normalize_infos(
        self,
        infos: object,
        *,
        operation: str,
        expected_mask: np.ndarray,
    ) -> dict[str, np.ndarray]:
        if not isinstance(infos, Mapping):
            raise TypeError(f"Gymnasium {operation} infos must be a columnar mapping")
        expected_keys = {
            key
            for signal in self.contract.signals
            for key in (signal.name, f"_{signal.name}")
        }
        if set(infos) != expected_keys:
            raise RuntimeError(
                f"Gymnasium {operation} info contract drifted: expected "
                f"{sorted(expected_keys)}, got {sorted(infos)}"
            )
        normalized: dict[str, np.ndarray] = {}
        for signal in self.contract.signals:
            available = (
                signal.available_on_reset if operation == "reset" else signal.available_on_step
            )
            signal_mask = np.asarray(infos[f"_{signal.name}"])
            if signal_mask.dtype != np.dtype(np.bool_) or signal_mask.shape != (
                self.num_envs,
            ):
                raise TypeError(
                    f"Gymnasium {operation} info mask _{signal.name} must be boolean "
                    f"shape ({self.num_envs},)"
                )
            required_mask = expected_mask if available else np.zeros_like(expected_mask)
            if not np.array_equal(signal_mask, required_mask):
                raise RuntimeError(
                    f"Gymnasium {operation} info mask _{signal.name} disagrees with "
                    "the declared availability"
                )
            raw_values = np.asarray(infos[signal.name])
            expected_shape = (self.num_envs, *signal.shape)
            if raw_values.shape != expected_shape:
                raise TypeError(
                    f"Gymnasium {operation} info {signal.name} must have shape "
                    f"{expected_shape}, got {raw_values.shape}"
                )
            dtype = np.dtype(signal.dtype)
            if signal.name == "prob":
                if not np.issubdtype(raw_values.dtype, np.number):
                    raise TypeError("Gymnasium prob info must be numeric")
                values = raw_values.astype(dtype, copy=True)
                if not np.all(np.isfinite(values[signal_mask])):
                    raise ValueError("Gymnasium prob info must be finite")
            else:
                if raw_values.dtype != dtype:
                    raise TypeError(
                        f"Gymnasium {operation} info {signal.name} dtype "
                        f"{raw_values.dtype} does not match {dtype}"
                    )
                values = raw_values.copy()
            normalized[signal.name] = values
            normalized[f"_{signal.name}"] = signal_mask.copy()
        return normalized

    def reset(self, *, seed=None, options=None):
        env = self._require_idle()
        mask = self._reset_mask(options)
        seeds = self._reset_seeds(seed, mask)
        observations, infos = env.reset(seed=seeds, options={"reset_mask": mask})
        normalized_infos = self._normalize_infos(
            infos,
            operation="reset",
            expected_mask=mask,
        )
        owned = np.asarray(observations).copy()
        if owned.shape[:1] != (self.num_envs,):
            raise ValueError("Gymnasium reset observations must contain every lane")
        if self._observations is not None:
            owned[~mask] = self._observations[~mask]
        self._observations = owned.copy()
        self._pending_reset[mask] = False
        normalized_infos.update({
            "state_index": np.full(self.num_envs, -1, dtype=np.int32),
            "_state_index": mask.copy(),
            "start_source": np.zeros(self.num_envs, dtype=np.int8),
            "_start_source": mask.copy(),
            "noop_reset_count": np.zeros(self.num_envs, dtype=np.int64),
            "_noop_reset_count": mask.copy(),
        })
        return owned, normalized_infos

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
        normalized_infos = self._normalize_infos(
            infos,
            operation="step",
            expected_mask=np.ones(self.num_envs, dtype=np.bool_),
        )
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
            normalized_infos,
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
    "GymnasiumSignalContract",
    "GymnasiumTurboVecEnv",
]
