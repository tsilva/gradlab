from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnv

from gradlab.batch_runtime import EpisodeRecord
from gradlab.task_kernels import Outcome


@dataclass
class GraDoomDeviceBatchStep:
    observations: Any
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    final_observations: Any
    transition_info: Mapping[str, Any]
    reset_info: Mapping[str, Any] | None
    diagnostics: None = None
    curriculum_cell_ids: None = None
    curriculum_generations: None = None
    curriculum_episode_indices: None = None
    curriculum_feedback_dones: None = None
    control_boundaries: None = None


@dataclass(frozen=True)
class _DeviceContextField:
    name: str
    source_indices: tuple[int, ...]
    encoding: str
    scale: torch.Tensor | None
    offset: torch.Tensor | None
    low: torch.Tensor | None
    high: torch.Tensor | None
    clip: bool
    categories: torch.Tensor | None


class _DeviceContextEncoder:
    def __init__(self, env: Any, kernel: Any, device: torch.device) -> None:
        signal_indices = {name: index for index, name in enumerate(env.device_signal_names)}
        fields: list[_DeviceContextField] = []
        for field in getattr(kernel, "fields", ()):
            try:
                indices = tuple(signal_indices[name] for name in field.source_names)
            except KeyError as exc:
                raise ValueError(
                    f"GraDOOM device context references unavailable signal {exc.args[0]!r}"
                ) from exc
            if field.history is not None:
                raise ValueError("GraDOOM device context histories are not implemented")
            if field.encoding == "continuous":
                fields.append(
                    _DeviceContextField(
                        name=field.name,
                        source_indices=indices,
                        encoding=field.encoding,
                        scale=torch.as_tensor(field.scale, device=device),
                        offset=torch.as_tensor(field.offset, device=device),
                        low=torch.as_tensor(field.low, device=device),
                        high=torch.as_tensor(field.high, device=device),
                        clip=field.clip,
                        categories=None,
                    )
                )
                continue
            if any(not isinstance(value, int) for value in field.categories):
                raise ValueError("GraDOOM device categorical contexts must contain integer values")
            fields.append(
                _DeviceContextField(
                    name=field.name,
                    source_indices=indices,
                    encoding=field.encoding,
                    scale=None,
                    offset=None,
                    low=None,
                    high=None,
                    clip=False,
                    categories=torch.as_tensor(field.categories, device=device),
                )
            )
        self.fields = tuple(fields)

    def encode(self, observations: torch.Tensor, signals: torch.Tensor) -> Any:
        if not self.fields:
            return observations
        result: dict[str, torch.Tensor] = {"observation": observations}
        for field in self.fields:
            raw = signals[:, field.source_indices]
            if field.encoding == "continuous":
                assert field.scale is not None
                assert field.offset is not None
                encoded = raw * field.scale + field.offset
                if field.clip:
                    assert field.low is not None
                    assert field.high is not None
                    encoded = torch.maximum(torch.minimum(encoded, field.high), field.low)
                result[f"context/{field.name}"] = encoded
                continue
            assert field.categories is not None
            if raw.shape[1] != 1:
                raise ValueError("GraDOOM categorical contexts must be scalar")
            matches = raw[:, 0, None] == field.categories[None, :]
            result[f"context/{field.name}"] = torch.argmax(matches.to(torch.int64), dim=1)
        return result


class GraDoomDeviceRuntime:
    """Minimal all-device rollout lifecycle for the certified Deathmatch profile."""

    device_resident = True
    archive_curriculum = None

    def __init__(
        self,
        env: Any,
        descriptor: Any,
        kernel: Any,
        *,
        action_contract: Mapping[str, Any],
        run_seed: int,
    ) -> None:
        self.provider = env
        self.descriptor = descriptor
        self.kernel = kernel
        self.num_envs = int(env.num_envs)
        self.device = env.device
        self.observation_space = kernel.observation_space
        self.action_space = kernel.action_space
        self.action_contract = MappingProxyType(dict(action_contract))
        self.global_lane_ids = tuple(range(self.num_envs))
        self.capture_step_diagnostics = False
        self.state_archive = None
        self.reset_infos = [{} for _ in range(self.num_envs)]
        self._run_seed = int(run_seed)
        self._lane = torch.arange(self.num_envs, device=self.device, dtype=torch.int64)
        self._episode_index = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)
        self._episode_returns = torch.zeros(self.num_envs, device=self.device)
        self._episode_lengths = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.int32,
        )
        self._encoder = _DeviceContextEncoder(env, kernel, self.device)
        signal_indices = {name: index for index, name in enumerate(env.device_signal_names)}
        self._killcount_signal_index = signal_indices["killcount"]
        self._record_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._calls_total = 0
        self._closed = False

    def _seeds(self) -> torch.Tensor:
        return torch.bitwise_and(
            self._run_seed + self._lane * 0x85EBCA6B + self._episode_index * 0x9E3779B1,
            (1 << 32) - 1,
        )

    def reset(self, *, seed: int | None = None, options_by_lane: Any = None) -> Any:
        if options_by_lane not in (None, (), []):
            if any(options_by_lane):
                raise ValueError("GraDOOM device runtime does not support per-lane reset options")
        if seed is not None:
            self._run_seed = int(seed)
        self._episode_index.zero_()
        self._episode_returns.zero_()
        self._episode_lengths.zero_()
        self._record_batches.clear()
        mask = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        observations, signals = self.provider.reset_device(mask, self._seeds())
        return self._encoder.encode(observations, signals)

    def step(self, actions: torch.Tensor) -> GraDoomDeviceBatchStep:
        if self._closed:
            raise RuntimeError("GraDOOM device runtime is closed")
        action_indices = actions.to(device=self.device, dtype=torch.int64).reshape(self.num_envs)
        next_episode_index = self._episode_index + 1
        reset_seeds = torch.bitwise_and(
            self._run_seed + self._lane * 0x85EBCA6B + next_episode_index * 0x9E3779B1,
            (1 << 32) - 1,
        )
        transition = self.provider.step_and_reset_device(action_indices, reset_seeds)
        done = transition.terminated | transition.truncated
        self._episode_returns.add_(transition.rewards)
        self._episode_lengths.add_(1)
        self._record_batches.append(
            (
                torch.stack(
                    (
                        done,
                        transition.terminated,
                        transition.truncated,
                        self._episode_index,
                        self._episode_lengths,
                    ),
                    dim=1,
                ),
                torch.stack(
                    (
                        self._episode_returns,
                        transition.final_signals[:, self._killcount_signal_index],
                    ),
                    dim=1,
                ),
            )
        )
        self._episode_returns.masked_fill_(done, 0.0)
        self._episode_lengths.masked_fill_(done, 0)
        self._episode_index.add_(done.to(torch.int64))
        self._calls_total += 1
        return GraDoomDeviceBatchStep(
            observations=self._encoder.encode(transition.observations, transition.signals),
            rewards=transition.rewards,
            terminated=transition.terminated,
            truncated=transition.truncated,
            final_observations=self._encoder.encode(
                transition.final_observations, transition.final_signals
            ),
            transition_info={},
            reset_info=None,
        )

    def drain_records(self) -> list[Any]:
        if not self._record_batches:
            return []
        status = torch.stack([batch[0] for batch in self._record_batches]).to("cpu").numpy()
        values = torch.stack([batch[1] for batch in self._record_batches]).to("cpu").numpy()
        self._record_batches.clear()
        records: list[EpisodeRecord] = []
        for step, lane in np.argwhere(status[:, :, 0] != 0):
            terminated = bool(status[step, lane, 1])
            truncated = bool(status[step, lane, 2])
            outcome = Outcome.FAILURE if terminated else Outcome.SUCCESS
            event = "player_died" if terminated else "time_limit_reached"
            records.append(
                EpisodeRecord(
                    lane=int(self.global_lane_ids[lane]),
                    episode_index=int(status[step, lane, 3]),
                    start_id="default",
                    episode_return=float(values[step, lane, 0]),
                    episode_length=int(status[step, lane, 4]),
                    terminated=terminated,
                    truncated=truncated,
                    outcome=outcome,
                    events=(event,),
                    metrics={"kills": float(values[step, lane, 1])},
                    boundary_reason="terminated" if terminated else "truncated",
                    provider_start_id="default",
                )
            )
        return records

    def native_step_stats(self) -> dict[str, float | int]:
        return {"calls_total": self._calls_total, "seconds_total": 0.0}

    def close(self) -> None:
        if not self._closed:
            self.provider.close()
            self._closed = True


class GraDoomDeviceVecEnv(VecEnv):
    def __init__(self, runtime: GraDoomDeviceRuntime) -> None:
        self.runtime = runtime
        self.env = runtime.provider
        self._actions: torch.Tensor | None = None
        super().__init__(
            runtime.num_envs,
            runtime.observation_space,
            runtime.action_space,
        )

    def reset(self) -> Any:
        return self.runtime.reset()

    def step_async(self, actions: Any) -> None:
        self._actions = torch.as_tensor(actions, device=self.runtime.device)

    def step_wait(self):
        if self._actions is None:
            raise RuntimeError("step_wait called without step_async")
        step = self.runtime.step(self._actions)
        self._actions = None
        dones = step.terminated | step.truncated
        return step.observations, step.rewards, dones, [{} for _ in range(self.num_envs)]

    def close(self) -> None:
        self.runtime.close()

    def get_images(self) -> list[np.ndarray]:
        return self.env.get_images()

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        value = getattr(self.env, attr_name)
        return [value for _ in self._get_indices(indices)]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        for _index in self._get_indices(indices):
            setattr(self.env, attr_name, value)

    def env_method(self, method_name: str, *method_args: Any, indices=None, **method_kwargs: Any):
        method = getattr(self.env, method_name)
        return [method(*method_args, **method_kwargs) for _index in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class: type, indices=None) -> list[bool]:
        del wrapper_class
        return [False for _ in self._get_indices(indices)]


def make_gradoom_device_vec_env(
    config: Any,
    *,
    n_envs: int,
    seed: int,
    rom_binding: Any = None,
    state_archive: Mapping[str, Any] | None = None,
) -> GraDoomDeviceVecEnv:
    if state_archive is not None:
        raise ValueError("GraDOOM device runtime does not support state archives")
    from gradlab.action_contract import compile_runtime_action_contract
    from gradlab.env import (
        _bound_task_kernel,
        native_obs_crop,
        state_weight_mapping,
        task_action_codec,
        task_action_values,
        task_reward,
    )
    from gradlab.env_providers import (
        gradoom_vec_env_type,
        provider_descriptor,
        provider_native_vec_kwargs,
    )
    from gradlab.reward_programs import normalize_vizdoom_deathmatch_reward

    reward = normalize_vizdoom_deathmatch_reward(
        task_reward(config),
        label="task.reward",
        require_complete=True,
    )
    if reward["reward_mode"] != "native":
        raise ValueError("GraDOOM device runtime initially supports native Deathmatch reward only")
    runtime_rom_path = rom_binding.rom_path if rom_binding is not None else None
    kwargs = provider_native_vec_kwargs(
        config,
        n_envs=n_envs,
        native_obs_crop=native_obs_crop,
        state_weight_mapping=state_weight_mapping,
        runtime_rom_path=runtime_rom_path,
    )
    env_type = gradoom_vec_env_type()
    env = env_type(config.game, **kwargs)
    try:
        descriptor = provider_descriptor(
            config,
            env,
            state_weight_mapping=state_weight_mapping,
        )
        kernel = _bound_task_kernel(config, descriptor, n_envs)
        action_contract = compile_runtime_action_contract(
            config,
            descriptor,
            kernel.action_space,
            policy_action_values=task_action_values(config),
            policy_action_codec=task_action_codec(config),
        )
        runtime = GraDoomDeviceRuntime(
            env,
            descriptor,
            kernel,
            action_contract=action_contract,
            run_seed=seed,
        )
        return GraDoomDeviceVecEnv(runtime)
    except BaseException:
        env.close()
        raise


__all__ = [
    "GraDoomDeviceBatchStep",
    "GraDoomDeviceRuntime",
    "GraDoomDeviceVecEnv",
    "make_gradoom_device_vec_env",
]
