from __future__ import annotations

import os
import multiprocessing
import tempfile
import time
import traceback
from copy import deepcopy
from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np
import env_stableretro_turbo as retro

from gradlab import env_providers as provider_runtime
from gradlab.action_contract import compile_runtime_action_contract
from gradlab.action_overrides import with_conditional_action_overrides
from gradlab.batch_runtime import BatchRuntime, ProviderDescriptor
from gradlab.env_providers import (
    DEFAULT_RETRO_VEC_ENV as RetroVecEnv,
    ale_py_atari_vector_env_type as _ale_py_atari_vector_env_type,
    provider_descriptor,
    provider_native_vec_kwargs,
    super_mario_bros_nes_turbo_vec_env_type as _super_mario_bros_nes_turbo_vec_env_type,
)
from gradlab.env_registry import (
    ALE_PY_PROVIDER,
    STABLE_RETRO_TURBO_PROVIDER,
    env_supports_states,
    qualify_env_id,
    resolve_native_episode_horizon,
    resolve_env_provider,
    validate_provider_constructor_args,
    validate_provider_resolved_config,
)
from gradlab.environment_fields import (
    DEFAULT_OBS_RESIZE_ALGORITHM as DEFAULT_OBS_RESIZE_ALGORITHM,
    GAME as GAME,
    EnvConfig as EnvConfig,
)
from gradlab.local_paths import PORTABLE_DEFAULT_RUNS_DIR, configure_matplotlib_cache
from gradlab.env_identity import task_config_from_train_config, validate_task_config
from gradlab.env_registry import environment_spec
from gradlab.task_kernels import (
    CELL_NOVELTY_REWARD_KEY,
    EVENT_REWARDS_KEY,
    IdentityTaskDefinition,
    MarioTaskConfig,
    MarioTaskDefinition,
    with_cell_novelty,
    with_deathmatch_reward,
    with_episode_progress_metrics,
    with_event_rewards,
    with_reward_transform,
)
from gradlab.model_inputs import with_model_inputs
from gradlab.validation import (
    normalize_obs_crop as validate_obs_crop,
    normalize_obs_resize as validate_obs_resize,
)
from gradlab.rom_runtime import RomRuntimeBinding

configure_matplotlib_cache()


def validate_obs_crop_mode(value: str) -> str:
    if value not in {"remove", "mask"}:
        raise ValueError("obs_crop_mode must be 'remove' or 'mask'")
    return value


def validate_obs_crop_fill(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 255:
        raise ValueError("obs_crop_fill must be an integer in [0, 255]")
    return int(value)


def native_obs_crop(config: EnvConfig) -> tuple[int, int, int, int] | None:
    obs_crop = validate_obs_crop(config.obs_crop)
    return obs_crop if obs_crop is not None and any(obs_crop) else None


def _validate_sticky_action_prob(value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError("sticky_action_prob must be in [0, 1]")
    return float(value)


def resolve_env_config(config: EnvConfig) -> EnvConfig:
    if not config.game:
        raise ValueError("game is required; pass --game or set RETRO_GAME")
    qualify_env_id(config.env_provider, config.game)
    validate_provider_constructor_args(
        config.env_provider,
        config.env_args,
        label="env_args",
    )
    _validate_sticky_action_prob(config.sticky_action_prob)
    validate_obs_resize(config.obs_resize)
    validate_obs_crop_mode(config.obs_crop_mode)
    validate_obs_crop_fill(config.obs_crop_fill)
    spec = environment_spec(config.env_provider, config.game)
    updates: dict[str, Any] = {}
    if not config.state and spec.default_state:
        updates["state"] = spec.default_state
    if config.obs_crop is None and spec.default_obs_crop is not None:
        updates["obs_crop"] = spec.default_obs_crop
    config = replace(config, **updates) if updates else config
    validate_provider_resolved_config(
        config.env_provider,
        config,
        label="environment",
    )
    if config.task:
        validate_task_config(config.task)
        canonical_task = task_config_from_train_config(
            {
                "env_provider": config.env_provider,
                "game": config.game,
                "task": config.task,
            },
            task=config.task,
        )
    else:
        canonical_task = task_config_from_train_config(
            {"env_provider": config.env_provider, "game": config.game}
        )
    config = replace(config, task=canonical_task)
    return config


def _validate_state_names(game: str, states: tuple[str, ...]) -> None:
    if any(not state for state in states):
        raise ValueError("--states must not contain empty state names")
    valid_states = set(retro.data.list_states(game))
    unknown = [state for state in states if state not in valid_states]
    if unknown:
        valid_preview = ", ".join(sorted(valid_states)[:12])
        raise ValueError(
            "unknown stable-retro state(s) for "
            f"{game}: {', '.join(unknown)}. Known examples: {valid_preview}"
        )


def resolve_mixed_state_config(config: EnvConfig, n_envs: int) -> EnvConfig:
    config = resolve_env_config(config)
    if n_envs < 1:
        raise ValueError("n_envs must be >= 1")
    provider = resolve_env_provider(config.env_provider)
    if not env_supports_states(provider.provider_id, config.game) and (
        config.state or config.states or config.state_probs
    ):
        raise ValueError(
            f"environment provider {provider.provider_id!r} does not support "
            "state, states, or state_probs"
        )
    if not config.states:
        if config.state_probs:
            raise ValueError("--state-probs requires --states")
        return config
    if provider.provider_id == STABLE_RETRO_TURBO_PROVIDER.provider_id:
        _validate_state_names(config.game, config.states)
    if config.state_probs:
        if len(config.state_probs) != len(config.states):
            raise ValueError("--state-probs count must match --states count")
        probs = np.asarray(config.state_probs, dtype=np.float64)
        if not np.all(np.isfinite(probs)) or np.any(probs < 0.0) or probs.sum() <= 0.0:
            raise ValueError("--state-probs must be non-negative finite values with a positive sum")
        return config
    if len(config.states) != n_envs:
        raise ValueError(
            "--states without --state-probs must provide exactly one state per env slot: "
            f"got {len(config.states)} states for n_envs={n_envs}"
        )
    return config


def state_distribution_metadata(config: EnvConfig) -> list[dict[str, float | str]]:
    if not config.states:
        return []
    if config.state_probs:
        distribution: dict[str, float] = {}
        for state, prob in zip(config.states, config.state_probs, strict=True):
            distribution[state] = distribution.get(state, 0.0) + float(prob)
        total = sum(distribution.values())
        return [
            {"state": state, "probability": probability / total}
            for state, probability in distribution.items()
        ]
    probability = 1.0 / len(config.states)
    return [{"state": state, "probability": probability} for state in config.states]


def state_weight_mapping(config: EnvConfig) -> dict[str, float]:
    weights: dict[str, float] = {}
    for state, weight in zip(config.states, config.state_probs, strict=True):
        weights[state] = weights.get(state, 0.0) + float(weight)
    return weights


def state_name_candidates_from_level_id(level_id: str) -> tuple[str, ...]:
    candidates = [f"Level{level_id}"]
    parts = level_id.split("-", 1)
    if len(parts) == 2:
        try:
            candidates.append(f"Level{int(parts[0]) + 1}-{int(parts[1]) + 1}")
        except ValueError:
            pass
    return tuple(dict.fromkeys(candidates))


def info_value_from_state_name(
    state_name: str,
    info_vars: tuple[str, ...],
) -> tuple[int | str, ...] | None:
    if tuple(info_vars) == ("levelHi", "levelLo") and state_name.startswith("Level"):
        level = state_name.removeprefix("Level").split("-", 2)
        if len(level) >= 2:
            try:
                return (int(level[0]) - 1, int(level[1]) - 1)
            except ValueError:
                return None
    return None


def task_conditioning_info_values(config: EnvConfig) -> tuple[tuple[int | str, ...], ...]:
    conditioning = config.task.get("conditioning", {})
    if not isinstance(conditioning, Mapping) or not conditioning.get("enabled"):
        return ()
    configured = conditioning.get("values", ())
    if configured:
        return tuple(tuple(value) for value in configured)
    signal_name = conditioning.get("signal")
    signals = config.task.get("signals", {})
    source = signals.get(signal_name) if isinstance(signals, Mapping) else None
    info_vars = (source,) if isinstance(source, str) else tuple(source or ())
    values = []
    for state_name in dict.fromkeys(config.states or ((config.state,) if config.state else ())):
        value = info_value_from_state_name(state_name, info_vars)
        if value is not None:
            values.append(value)
    return tuple(values)


def task_action_set(config: EnvConfig) -> str:
    action = config.task.get("action", {})
    return str(action.get("set", "native")) if isinstance(action, Mapping) else "native"


def task_action_values(config: EnvConfig) -> tuple[Any, ...] | None:
    action = config.task.get("action", {})
    if not isinstance(action, Mapping):
        return None
    codec = action.get("codec")
    if not isinstance(codec, Mapping):
        return None
    values = codec.get("values")
    return tuple(values) if isinstance(values, list | tuple) else None


def task_action_codec(config: EnvConfig) -> Mapping[str, Any] | None:
    action = config.task.get("action", {})
    if not isinstance(action, Mapping):
        return None
    codec = action.get("codec")
    return codec if isinstance(codec, Mapping) else None


def task_termination(config: EnvConfig) -> Mapping[str, Any]:
    value = config.task.get("termination", {})
    return value if isinstance(value, Mapping) else {}


def task_reward(config: EnvConfig) -> Mapping[str, Any]:
    value = config.task.get("reward", {})
    return value if isinstance(value, Mapping) else {}


def task_max_episode_steps(config: EnvConfig) -> int:
    return int(task_termination(config).get("max_episode_steps", 0))


def task_conditioning(config: EnvConfig) -> Mapping[str, Any]:
    value = config.task.get("conditioning", {})
    return value if isinstance(value, Mapping) else {}


def with_task_termination(config: EnvConfig, **updates: Any) -> EnvConfig:
    task = deepcopy(config.task)
    termination = dict(task.get("termination", {}))
    termination.update(updates)
    task["termination"] = termination
    return replace(config, task=task)


def make_provider_vec_env(config: EnvConfig, *, native_kwargs: Mapping[str, Any]):
    return provider_runtime.make_provider_vec_env(
        config,
        native_kwargs=native_kwargs,
        retro_vec_env_type=RetroVecEnv,
        super_mario_vec_env_type=_super_mario_bros_nes_turbo_vec_env_type,
        ale_py_vec_env_type=_ale_py_atari_vector_env_type,
    )


def _provider_descriptor(config: EnvConfig, native_env: Any) -> ProviderDescriptor:
    return provider_descriptor(
        config,
        native_env,
        state_weight_mapping=state_weight_mapping,
    )


def make_native_provider(
    config: EnvConfig,
    n_envs: int,
    *,
    rom_binding: RomRuntimeBinding | None = None,
    native_kwargs_overrides: Mapping[str, Any] | None = None,
) -> tuple[Any, ProviderDescriptor]:
    """Construct and describe one provider, closing it if description fails."""

    provider = resolve_env_provider(config.env_provider)
    if provider.requires_external_rom_asset and rom_binding is None:
        raise FileNotFoundError(f"{provider.provider_id} requires a verified runtime ROM binding")

    native_kwargs = provider_native_vec_kwargs(
        config,
        n_envs=n_envs,
        native_obs_crop=native_obs_crop,
        state_weight_mapping=state_weight_mapping,
        runtime_rom_path=rom_binding.rom_path if rom_binding is not None else None,
    )
    if native_kwargs_overrides is not None:
        native_kwargs.update(native_kwargs_overrides)
    native_env = make_provider_vec_env(config, native_kwargs=native_kwargs)
    try:
        descriptor = _provider_descriptor(config, native_env)
    except BaseException:
        native_env.close()
        raise
    return native_env, descriptor


def bind_native_provider(
    config: EnvConfig,
    *,
    n_envs: int,
    seed: int,
    native_env: Any,
    descriptor: ProviderDescriptor,
    episode_progress_fields: Sequence[str] = (),
    global_lane_ids: tuple[int, ...] | None = None,
    capture_step_diagnostics: bool = False,
    state_archive: Mapping[str, Any] | None = None,
    state_archive_root: str | os.PathLike[str] | None = None,
) -> BatchRuntime:
    """Transfer a constructed provider into the task runtime or close it on failure."""

    runtime: BatchRuntime | None = None
    try:
        kernel = _bound_task_kernel(
            config,
            descriptor,
            n_envs,
            episode_progress_fields=episode_progress_fields,
        )
        action_values = task_action_values(config)
        action_contract = compile_runtime_action_contract(
            config,
            descriptor,
            kernel.action_space,
            policy_action_values=action_values,
            policy_action_codec=task_action_codec(config),
        )
        kernel = with_conditional_action_overrides(
            kernel,
            descriptor,
            config.task.get("signals", {}),
            action_contract,
        )
        runtime = BatchRuntime(
            native_env,
            descriptor,
            kernel,
            action_contract=action_contract,
            run_seed=seed,
            global_lane_ids=global_lane_ids,
            capture_step_diagnostics=capture_step_diagnostics,
            state_archive=state_archive,
            state_archive_root=state_archive_root,
        )
        return runtime
    except BaseException:
        if runtime is None:
            native_env.close()
        else:
            runtime.close()
        raise


def _bound_task_kernel(
    config: EnvConfig,
    descriptor: ProviderDescriptor,
    n_envs: int,
    *,
    episode_progress_fields: Sequence[str] = (),
):
    native_horizon = resolve_native_episode_horizon(
        {
            "env_provider": config.env_provider,
            "env_args": config.env_args,
            "frame_skip": config.frame_skip,
        }
    )
    task_id = config.task.get("id")
    if task_id == "mario":
        kernel = MarioTaskDefinition(MarioTaskConfig.from_env_config(config)).bind(
            descriptor,
            n_envs,
        )
        kernel = with_reward_transform(kernel, task_reward(config))
        kernel = with_episode_progress_metrics(
            kernel,
            descriptor,
            config.task.get("signals", {}),
            episode_progress_fields,
        )
        return with_model_inputs(
            kernel,
            descriptor,
            config.task,
            native_episode_horizon=native_horizon,
        )
    if task_id != "identity":
        raise ValueError(f"unknown task kernel {task_id!r}")
    reward = task_reward(config)
    action_codec = task_action_codec(config)
    if task_action_set(config) != "native" and action_codec is None:
        raise ValueError(
            "generic native-vector tasks require native actions or a task action codec"
        )
    reward_mode = reward.get("reward_mode")
    if reward_mode not in {"native", "sample-factory-v0"}:
        raise ValueError(
            "generic native-vector tasks require native or Sample Factory Deathmatch rewards"
        )
    if task_conditioning(config).get("enabled"):
        raise ValueError("generic native-vector tasks do not support task conditioning")
    # Stable Retro applies obs_crop natively. Only ale-py needs the task kernel
    # to mask its already-resized observations.
    observation_mask = (
        native_obs_crop(config) if config.env_provider == ALE_PY_PROVIDER.provider_id else None
    )
    source_shape = (210, 160) if observation_mask is not None else None
    kernel = IdentityTaskDefinition(
        observation_mask=observation_mask,
        observation_mask_fill=config.obs_crop_fill,
        observation_source_shape=source_shape,
        max_episode_steps=task_max_episode_steps(config),
        action_codec=action_codec,
        signals=config.task.get("signals", {}),
        events=config.task.get("events", {}),
        termination=task_termination(config),
    ).bind(descriptor, n_envs)
    if reward_mode == "sample-factory-v0":
        kernel = with_deathmatch_reward(
            kernel,
            descriptor,
            config.task.get("signals", {}),
            reward,
        )
    kernel = with_cell_novelty(kernel, reward.get(CELL_NOVELTY_REWARD_KEY))
    kernel = with_event_rewards(kernel, reward.get(EVENT_REWARDS_KEY))
    kernel = with_reward_transform(kernel, reward)
    kernel = with_episode_progress_metrics(
        kernel,
        descriptor,
        config.task.get("signals", {}),
        episode_progress_fields,
    )
    return with_model_inputs(
        kernel,
        descriptor,
        config.task,
        native_episode_horizon=native_horizon,
    )


def make_vec_envs(
    config: EnvConfig,
    n_envs: int,
    seed: int,
    *,
    episode_progress_fields: Sequence[str] = (),
    capture_step_diagnostics: bool = False,
    rom_binding: RomRuntimeBinding | None = None,
    state_archive: Mapping[str, Any] | None = None,
    state_archive_root: str | os.PathLike[str] | None = None,
    native_kwargs_overrides: Mapping[str, Any] | None = None,
) -> Any:
    from gradlab.training.sb3_vec_env import GradLabVecEnv

    runtime = make_training_batch_runtime(
        config,
        n_envs,
        seed,
        episode_progress_fields=episode_progress_fields,
        capture_step_diagnostics=capture_step_diagnostics,
        rom_binding=rom_binding,
        state_archive=state_archive,
        state_archive_root=state_archive_root,
        native_kwargs_overrides=native_kwargs_overrides,
    )
    vec_env = GradLabVecEnv(runtime)
    vec_env.seed(seed)
    return vec_env


def make_training_batch_runtime(
    config: EnvConfig,
    n_envs: int,
    seed: int,
    *,
    episode_progress_fields: Sequence[str] = (),
    global_lane_ids: tuple[int, ...] | None = None,
    capture_step_diagnostics: bool = False,
    rom_binding: RomRuntimeBinding | None = None,
    state_archive: Mapping[str, Any] | None = None,
    state_archive_root: str | os.PathLike[str] | None = None,
    native_kwargs_overrides: Mapping[str, Any] | None = None,
) -> BatchRuntime:
    os.environ.setdefault("STABLE_RETRO_DISABLE_AUDIO", "1")
    if state_archive is not None:
        from gradlab.state_archive import normalize_state_archive_config

        normalized_archive = normalize_state_archive_config(
            state_archive,
            n_envs=n_envs,
        )
        assert normalized_archive is not None
        cell = normalized_archive["recorder"].get("cell")
        sources = {
            str(dimension["source"])
            for dimension in (cell or {}).get("dimensions", ())
            if isinstance(dimension, Mapping) and "source" in dimension
        }
        if sources:
            env_args = dict(config.env_args)
            configured_filter = env_args.get("info_filter")
            configured_keys: set[str] = set()
            if isinstance(configured_filter, Mapping):
                if str(configured_filter.get("mode", "all")) != "all":
                    raise ValueError("state archive cell sources require info_filter mode='all'")
                keys = configured_filter.get("keys")
                if keys is not None:
                    if isinstance(keys, str | bytes) or not isinstance(
                        keys,
                        Sequence,
                    ):
                        raise ValueError("info_filter.keys must be a sequence")
                    configured_keys.update(str(key) for key in keys)
            elif configured_filter is not None and str(configured_filter) != "all":
                raise ValueError("state archive cell sources require info_filter='all'")
            task = config.task if isinstance(config.task, Mapping) else {}
            signals = task.get("signals")
            if isinstance(signals, Mapping):
                for source in signals.values():
                    configured_keys.update(
                        (str(source),)
                        if isinstance(source, str)
                        else (str(name) for name in source)
                    )
            if task.get("id") == "mario":
                configured_keys.add("time")
            env_args["info_filter"] = {
                "mode": "all",
                "keys": tuple(sorted(configured_keys | sources)),
            }
            config = replace(config, env_args=env_args)
    config = resolve_mixed_state_config(config, n_envs=n_envs)
    native_env, descriptor = make_native_provider(
        config,
        n_envs,
        rom_binding=rom_binding,
        native_kwargs_overrides=native_kwargs_overrides,
    )
    return bind_native_provider(
        config,
        n_envs=n_envs,
        seed=seed,
        native_env=native_env,
        descriptor=descriptor,
        episode_progress_fields=episode_progress_fields,
        global_lane_ids=global_lane_ids,
        capture_step_diagnostics=capture_step_diagnostics,
        state_archive=state_archive,
        state_archive_root=state_archive_root,
    )


def _state_archive_preflight_lane_count(value: Mapping[str, Any], configured_n_envs: int) -> int:
    from gradlab.state_archive import normalize_state_archive_config

    if configured_n_envs < 2:
        raise ValueError("state archive preflight requires at least two configured lanes")
    for lanes in range(2, min(configured_n_envs, 32) + 1):
        try:
            normalize_state_archive_config(value, n_envs=lanes)
        except ValueError as exc:
            if "resolves to" not in str(exc):
                raise
        else:
            return lanes
    return configured_n_envs


def _state_archive_preflight_child(
    connection: Any,
    config: EnvConfig,
    configured_n_envs: int,
    seed: int,
    rom_binding: RomRuntimeBinding | None,
    state_archive: Mapping[str, Any],
) -> None:
    runtime: BatchRuntime | None = None
    try:
        preflight_lanes = _state_archive_preflight_lane_count(
            state_archive,
            configured_n_envs,
        )
        with tempfile.TemporaryDirectory(prefix="gradlab-state-archive-preflight-") as root:
            try:
                runtime = make_training_batch_runtime(
                    config,
                    preflight_lanes,
                    seed,
                    rom_binding=rom_binding,
                    state_archive=state_archive,
                    state_archive_root=root,
                )
                if runtime.state_archive is None:
                    raise RuntimeError("state archive preflight runtime is disabled")
                payload = runtime.preflight_state_archive_round_trip(seed=seed)
                runtime.close()
                runtime = None
                connection.send(("ok", payload))
            finally:
                if runtime is not None:
                    runtime.close()
                    runtime = None
    except BaseException as exc:
        connection.send(
            (
                "error",
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(limit=20),
                },
            )
        )
    finally:
        if runtime is not None:
            runtime.close()
        connection.close()


def preflight_state_archive_provider(
    *,
    config: EnvConfig,
    n_envs: int,
    seed: int,
    rom_binding: RomRuntimeBinding | None,
    state_archive: Mapping[str, Any] | None,
    timeout_seconds: float = 60.0,
) -> dict[str, Any] | None:
    """Run the state-archive codec probe in an isolated, disposable process."""

    if state_archive is None:
        return None
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_state_archive_preflight_child,
        args=(sender, config, int(n_envs), int(seed), rom_binding, dict(state_archive)),
        name="gradlab-state-archive-preflight",
    )
    started_at = time.perf_counter()
    process.start()
    sender.close()
    try:
        if not receiver.poll(timeout_seconds):
            process.terminate()
            process.join(timeout=10.0)
            raise TimeoutError(
                f"state archive provider preflight exceeded {timeout_seconds:g} seconds"
            )
        status, payload = receiver.recv()
    finally:
        receiver.close()
    process.join(timeout=10.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=10.0)
        raise RuntimeError("state archive provider preflight child did not exit")
    if status != "ok":
        raise RuntimeError(
            "state archive provider preflight failed: "
            f"{payload['type']}: {payload['message']}\n{payload['traceback']}"
        )
    if process.exitcode != 0:
        raise RuntimeError(f"state archive provider preflight exited with code {process.exitcode}")
    return {
        **dict(payload),
        "elapsed_seconds": time.perf_counter() - started_at,
        "isolation": "spawned_process",
    }


def make_training_vec_env(
    config: EnvConfig,
    n_envs: int,
    seed: int,
    *,
    episode_progress_fields: Sequence[str] = (),
    rom_binding: RomRuntimeBinding | None = None,
    state_archive: Mapping[str, Any] | None = None,
    state_archive_root: str | os.PathLike[str] | None = None,
    native_kwargs_overrides: Mapping[str, Any] | None = None,
) -> Any:
    return make_vec_envs(
        config=config,
        n_envs=n_envs,
        seed=seed,
        episode_progress_fields=episode_progress_fields,
        rom_binding=rom_binding,
        state_archive=state_archive,
        state_archive_root=state_archive_root,
        native_kwargs_overrides=native_kwargs_overrides,
    )


def make_eval_vec_env(
    config: EnvConfig,
    n_envs: int,
    seed: int,
    *,
    capture_step_diagnostics: bool = False,
    rom_binding: RomRuntimeBinding | None = None,
    state_archive: Mapping[str, Any] | None = None,
    state_archive_root: str | os.PathLike[str] | None = None,
    native_kwargs_overrides: Mapping[str, Any] | None = None,
) -> Any:
    return make_vec_envs(
        config=resolve_env_config(config),
        n_envs=n_envs,
        seed=seed,
        capture_step_diagnostics=capture_step_diagnostics,
        rom_binding=rom_binding,
        state_archive=state_archive,
        state_archive_root=state_archive_root,
        native_kwargs_overrides=native_kwargs_overrides,
    )


def assert_provider_runtime_available(
    config: EnvConfig,
    *,
    rom_binding: RomRuntimeBinding | None = None,
) -> None:
    provider = resolve_env_provider(config.env_provider)
    if provider.requires_external_rom_asset:
        if rom_binding is None:
            raise FileNotFoundError(f"{config.game} requires a verified external ROM asset binding")
        if rom_binding.manifest.get("game") != config.game:
            raise ValueError("runtime ROM binding game mismatch")
    elif provider.provider_id == ALE_PY_PROVIDER.provider_id:
        from ale_py import roms

        if roms.get_rom_path(config.game) is None:
            raise FileNotFoundError(
                f"{config.game} is not available to ale-py. "
                "Install an ALE ROM package or import ROMs with ale-import-roms."
            )


def default_run_dir(
    run_name: str,
    runs_dir: str = PORTABLE_DEFAULT_RUNS_DIR,
) -> str:
    return os.path.join(os.path.expanduser(runs_dir), run_name)
