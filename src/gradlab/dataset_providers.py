"""Thin single-lane dataset adapter over gradlab's shared provider runtime."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from typing import Any

import gymnasium as gym
import numpy as np

from gradlab.action_contract import declared_action_contract, provider_buttons
from gradlab.batch_runtime import ProviderDescriptor
from gradlab.env import EnvConfig, make_native_provider, resolve_env_config
from gradlab.env_config import env_config_from_mapping
from gradlab.env_registry import (
    STABLE_RETRO_TURBO_PROVIDER,
    SUPERMARIOBROS_NES_TURBO_PROVIDER,
    resolve_env_provider,
)
from gradlab.json_utils import canonical_json_bytes, json_value
from gradlab.rom_assets import (
    DEFAULT_LOCAL_ROM_CACHE,
    portable_rom_asset_identity,
    rom_asset_manifest_for_game,
)
from gradlab.rom_runtime import bind_cached_rom


PROVIDER_CONTRACT_VERSION = 2
SUPPORTED_PROVIDER_IDS = frozenset(
    {
        STABLE_RETRO_TURBO_PROVIDER.provider_id,
        SUPERMARIOBROS_NES_TURBO_PROVIDER.provider_id,
    }
)
_CONFIG_FIELDS = frozenset(field.name for field in fields(EnvConfig))
_MANAGED_CONFIG_FIELDS = frozenset({"env_provider", "game", "states", "state_probs"})
_MANAGED_ENV_ARGS = frozenset(
    {"autoreset_mode", "game", "num_envs", "num_threads", "render_mode", "rom_path"}
)


def _lane_info(infos: Any) -> dict[str, Any]:
    if not isinstance(infos, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key, value in infos.items():
        if str(key).startswith("_"):
            continue
        mask = infos.get(f"_{key}")
        if mask is not None and not bool(np.asarray(mask).reshape(-1)[0]):
            continue
        if isinstance(value, np.ndarray) and value.shape[:1] == (1,):
            value = value[0]
        elif (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes, bytearray, memoryview))
            and len(value) == 1
        ):
            value = value[0]
        result[str(key)] = json_value(value)
    return result


class SingleLaneEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, vector_env: Any) -> None:
        if int(vector_env.num_envs) != 1:
            raise ValueError("dataset providers require exactly one environment lane")
        self.vector_env = vector_env
        self.action_space = vector_env.single_action_space
        self.observation_space = vector_env.single_observation_space
        self.render_mode = "rgb_array"
        self._needs_reset = True

    def reset(self, *, seed: int | None = None, options: Mapping[str, Any] | None = None):
        super().reset(seed=seed)
        observations, infos = self.vector_env.reset(seed=seed, options=options)
        self._needs_reset = False
        return observations[0], _lane_info(infos)

    def step(self, action: Any):
        if self._needs_reset:
            raise RuntimeError("reset() is required after a terminal step")
        if isinstance(self.action_space, gym.spaces.Discrete):
            scalar = int(np.asarray(action).reshape(-1)[0])
            if not self.action_space.contains(scalar):
                raise ValueError(f"action {scalar!r} is not in {self.action_space}")
            batched = np.asarray([scalar], dtype=self.action_space.dtype)
        else:
            scalar = np.asarray(action, dtype=self.action_space.dtype)
            if not self.action_space.contains(scalar):
                raise ValueError(f"action {action!r} is not in {self.action_space}")
            batched = scalar[np.newaxis, ...]
        observations, rewards, terminated, truncated, infos = self.vector_env.step(batched)
        is_terminated = bool(terminated[0])
        is_truncated = bool(truncated[0])
        self._needs_reset = is_terminated or is_truncated
        return (
            observations[0],
            float(rewards[0]),
            is_terminated,
            is_truncated,
            _lane_info(infos),
        )

    def render(self):
        get_images = getattr(self.vector_env, "get_images", None)
        if callable(get_images):
            images = get_images()
            if isinstance(images, Sequence) and len(images) == 1:
                return images[0]
        frame = self.vector_env.render()
        if isinstance(frame, Sequence) and not isinstance(
            frame, (str, bytes, bytearray, memoryview, np.ndarray)
        ):
            return frame[0] if len(frame) == 1 else frame
        array = np.asarray(frame)
        return array[0] if array.ndim >= 4 and array.shape[0] == 1 else frame

    def close(self) -> None:
        self.vector_env.close()


def _validate_declared_config(config: Mapping[str, Any]) -> None:
    unknown = sorted(set(config) - _CONFIG_FIELDS)
    if unknown:
        raise ValueError(f"unknown environment config key(s): {', '.join(unknown)}")
    managed = sorted(_MANAGED_CONFIG_FIELDS.intersection(config))
    if managed:
        raise ValueError(f"managed environment config key(s): {', '.join(managed)}")
    env_args = config.get("env_args", {})
    if not isinstance(env_args, Mapping):
        raise ValueError("env config env_args must be an object")
    managed_args = sorted(_MANAGED_ENV_ARGS.intersection(env_args))
    if managed_args:
        raise ValueError(
            "dataset runtime owns env_args key(s): " + ", ".join(managed_args)
        )
    state = config.get("state")
    if state is not None and not isinstance(state, str):
        raise ValueError("dataset recording accepts only a default or named scalar state")


def validate_provider_request(config: Mapping[str, Any]) -> None:
    """Validate the provider-independent recording boundary before construction."""

    if not isinstance(config, Mapping):
        raise ValueError("env config must be an object")
    _validate_declared_config(config)


def _resolved_config(
    provider_id: str,
    environment_id: str,
    declared_config: Mapping[str, Any],
) -> EnvConfig:
    validate_provider_request(declared_config)
    if provider_id not in SUPPORTED_PROVIDER_IDS:
        known = ", ".join(sorted(SUPPORTED_PROVIDER_IDS))
        raise ValueError(f"dataset recording supports providers: {known}")
    return resolve_env_config(
        env_config_from_mapping(
            {
                "env_provider": provider_id,
                "game": environment_id,
                **dict(declared_config),
            }
        )
    )


def _control_action_table(
    config: EnvConfig,
) -> tuple[tuple[str, ...], ...] | None:
    declared = declared_action_contract(config)
    table = declared.get("table") if isinstance(declared, Mapping) else None
    if not isinstance(table, list | tuple):
        return None
    normalized: list[tuple[str, ...]] = []
    for entry in table:
        if (
            not isinstance(entry, list | tuple)
            or any(not isinstance(label, str) for label in entry)
        ):
            return None
        normalized.append(tuple(label.upper() for label in entry))
    return tuple(normalized)


class ProviderSession:
    def __init__(
        self,
        *,
        config: EnvConfig,
        vector_env: Any,
        descriptor: ProviderDescriptor,
        assets: Mapping[str, Any],
    ) -> None:
        provider = resolve_env_provider(config.env_provider)
        self.provider_id = provider.provider_id
        self.environment_id = config.game
        self.effective_config = json_value(asdict(config))
        self.env = SingleLaneEnv(vector_env)
        system = str(
            getattr(vector_env, "system", None)
            or ("Nes" if provider == SUPERMARIOBROS_NES_TURBO_PROVIDER else config.game)
        )
        self.control_profile = f"stable_retro.{system}"
        self.fps = max(60.0 / max(int(config.frame_skip), 1), 1.0)
        self._buttons = tuple(
            str(button).upper()
            for button in provider_buttons(config.env_provider, config.game)
            if button is not None
        )
        self._control_actions = _control_action_table(config)
        self._descriptor = descriptor
        self.provenance = {
            "distribution": provider.distribution_name,
            "version": importlib.metadata.version(provider.distribution_name),
            "assets": json_value(assets),
        }

    def policy_observation(self, observation: Any) -> Any:
        return observation

    def recording_observation(self, observation: Any) -> Any:
        frame = self.env.render()
        return observation if frame is None else frame

    def adapt_policy_action(self, action: Any) -> Any:
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            return int(np.asarray(action).reshape(-1)[0])
        return action

    def validate_policy(self, policy: Any) -> None:
        policy_action = getattr(policy, "action_space", None)
        if policy_action is not None:
            expected_n = getattr(self.env.action_space, "n", None)
            actual_n = getattr(policy_action, "n", None)
            if expected_n is not None and actual_n != expected_n:
                raise ValueError("policy action space does not match the provider")
            if expected_n is None and getattr(policy_action, "shape", None) != getattr(
                self.env.action_space, "shape", None
            ):
                raise ValueError("policy action shape does not match the provider")
        policy_observation = getattr(policy, "observation_space", None)
        if policy_observation is not None and getattr(
            policy_observation, "shape", None
        ) != getattr(self.env.observation_space, "shape", None):
            raise ValueError("policy observation space does not match the provider")

    def action_from_labels(self, labels: Sequence[str]) -> Any:
        requested = {str(label).upper() for label in labels}
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            if self._control_actions is not None:
                for index, action_labels in enumerate(self._control_actions):
                    if set(action_labels) == requested:
                        return index
            meanings = self._descriptor.action_meanings or ()
            action_buttons = getattr(self.env.vector_env, "ACTION_BUTTONS", {})
            for index, meaning in enumerate(meanings):
                actual = {
                    str(label).upper()
                    for label in action_buttons.get(str(meaning), ())
                }
                if actual == requested:
                    return index
            raise ValueError(f"no configured action matches controls {sorted(requested)!r}")
        if not isinstance(self.env.action_space, gym.spaces.MultiBinary):
            raise ValueError("named controls require MultiBinary or named Discrete actions")
        action = np.zeros(self.env.action_space.n, dtype=self.env.action_space.dtype)
        for label in requested:
            try:
                action[self._buttons.index(label)] = 1
            except ValueError as exc:
                raise ValueError(f"control label {label!r} is unavailable") from exc
        return action


def create_provider_session(
    provider_id: str,
    environment_id: str,
    config: Mapping[str, Any],
) -> ProviderSession:
    resolved = _resolved_config(provider_id, environment_id, config)
    manifest = rom_asset_manifest_for_game(resolved.game)
    binding = bind_cached_rom(manifest, cache_root=DEFAULT_LOCAL_ROM_CACHE)
    vector_env, descriptor = make_native_provider(
        resolved,
        1,
        rom_binding=binding,
    )
    try:
        return ProviderSession(
            config=resolved,
            vector_env=vector_env,
            descriptor=descriptor,
            assets={"rom": portable_rom_asset_identity(manifest)},
        )
    except BaseException:
        vector_env.close()
        raise


def space_contract(space: Any) -> dict[str, Any]:
    document: dict[str, Any] = {"type": type(space).__name__, "repr": str(space)}
    for name in ("shape", "dtype", "n", "start"):
        value = getattr(space, name, None)
        if value is None:
            continue
        if name == "shape":
            document[name] = [int(item) for item in value]
        elif name in {"n", "start"}:
            document[name] = int(value)
        else:
            document[name] = str(value)
    for name in ("low", "high"):
        value = getattr(space, name, None)
        if value is not None:
            document[name] = json_value(value)
    nvec = getattr(space, "nvec", None)
    if nvec is not None:
        document["nvec"] = json_value(nvec)
    return document


@dataclass(frozen=True)
class EnvironmentArtifact:
    contract_id: str
    document: Mapping[str, Any]


def build_environment_artifact(
    *,
    provider_id: str,
    environment_id: str,
    declared_config: Mapping[str, Any],
    session: ProviderSession,
) -> EnvironmentArtifact:
    document = {
        "document_type": "gymrec.environment",
        "format_version": 1,
        "provider_id": provider_id,
        "provider_contract_version": PROVIDER_CONTRACT_VERSION,
        "environment_id": environment_id,
        "declared_config": copy.deepcopy(dict(declared_config)),
        "effective_config": copy.deepcopy(session.effective_config),
        "provenance": copy.deepcopy(session.provenance),
        "action_space": space_contract(session.env.action_space),
        "observation_space": space_contract(session.env.observation_space),
        "control_profile": session.control_profile,
        "fps": float(session.fps),
    }
    contract_id = hashlib.sha256(canonical_json_bytes(document)).hexdigest()
    return EnvironmentArtifact(contract_id, document)
