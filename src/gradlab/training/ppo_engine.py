from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.utils import get_schedule_fn

from gradlab.batch_runtime import BatchMetricRecord, CurriculumStepAttribution
from gradlab.callbacks import (
    ARCHIVE_CURRICULUM_METRIC_MAP,
    RewardStatsAccumulator,
    policy_entropy_bounds,
)
from gradlab.metric_names import (
    TRAIN_PPO_APPROX_KL,
    TRAIN_PPO_CLIP_FRACTION,
    TRAIN_PPO_EXPLAINED_VARIANCE,
    TRAIN_PPO_LEARNING_RATE,
    TRAIN_PPO_POLICY_ENTROPY,
    TRAIN_PPO_POLICY_ENTROPY_BOUND_LOWER,
    TRAIN_PPO_POLICY_ENTROPY_BOUND_UPPER,
    TRAIN_PPO_VALUE_LOSS,
    TRAIN_THROUGHPUT_BETWEEN_ROLLOUTS_SECONDS,
    TRAIN_THROUGHPUT_ENV_STEP_FPS,
    TRAIN_THROUGHPUT_ENV_STEP_SECONDS,
    TRAIN_THROUGHPUT_LOOP_FPS,
    TRAIN_THROUGHPUT_ROLLOUT_OVERHEAD_SECONDS,
    TRAIN_THROUGHPUT_ROLLOUT_SECONDS,
    stat_metric,
    train_algorithm_metric,
    validate_metric_payload,
)
from gradlab.ppo import GradLabPPO
from gradlab.training.sb3_on_policy import (
    active_reward_components,
    active_reward_signals,
    checkpoint_prefix,
    checkpoint_save_frequency,
    policy_kwargs_from_config,
    policy_type_for_config,
    save_model_bundle,
    validate_action_space,
    validate_resumed_policy_model,
)
from gradlab.training_backend import BackendContext
from gradlab.training_lifecycle import ProgressField, TrainingExecutionMode, TrainingResult


ObservationTree = torch.Tensor | dict[str, "ObservationTree"]


def _tree_map(value: Any, function: Callable[[Any], Any]) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _tree_map(item, function) for key, item in value.items()}
    return function(value)


def _allocate_observations(observations: Any, n_steps: int, device: torch.device) -> Any:
    def allocate(value: Any) -> torch.Tensor:
        array = np.asarray(value)
        return torch.empty(
            (n_steps, *array.shape),
            dtype=torch.from_numpy(np.empty((), dtype=array.dtype)).dtype,
            device=device,
        )

    return _tree_map(observations, allocate)


def _copy_observation_slot(destination: Any, observations: Any, step: int) -> None:
    if isinstance(destination, Mapping):
        if not isinstance(observations, Mapping) or destination.keys() != observations.keys():
            raise ValueError("rollout observation structure changed")
        for key in destination:
            _copy_observation_slot(destination[key], observations[key], step)
        return
    destination[step].copy_(torch.as_tensor(observations, device=destination.device))


def _observation_tensor(observations: Any, device: torch.device) -> Any:
    return _tree_map(observations, lambda value: torch.as_tensor(value, device=device))


def _take_observation_lanes(observations: Any, lanes: np.ndarray) -> Any:
    return _tree_map(observations, lambda value: np.asarray(value)[lanes])


def _flatten_observations(observations: Any, *, env_major: bool = False) -> Any:
    if isinstance(observations, Mapping):
        return {
            key: _flatten_observations(value, env_major=env_major)
            for key, value in observations.items()
        }
    if env_major:
        return observations.transpose(0, 1).flatten(0, 1)
    return observations.flatten(0, 1)


def _flatten_rollout_tensor(value: torch.Tensor, *, env_major: bool) -> torch.Tensor:
    if env_major:
        return value.transpose(0, 1).flatten(0, 1)
    return value.flatten(0, 1)


def _index_observations(observations: Any, indices: torch.Tensor) -> Any:
    if isinstance(observations, Mapping):
        return {key: _index_observations(value, indices) for key, value in observations.items()}
    return observations.index_select(0, indices)


def _tree_nbytes(observations: Any) -> int:
    if isinstance(observations, Mapping):
        return sum(_tree_nbytes(value) for value in observations.values())
    return int(np.asarray(observations).nbytes)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@dataclass
class TensorRolloutBuffer:
    observations: Any
    actions: torch.Tensor
    rewards: torch.Tensor
    episode_starts: torch.Tensor
    values: torch.Tensor
    log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    position: int = 0

    @classmethod
    def allocate(
        cls,
        observations: Any,
        *,
        action_space: spaces.Space,
        n_steps: int,
        n_envs: int,
        device: torch.device,
    ) -> TensorRolloutBuffer:
        action_shape = (get_action_dim(action_space),)
        if isinstance(action_space, spaces.Discrete):
            action_dtype = torch.int64
        elif isinstance(action_space, spaces.MultiDiscrete):
            action_dtype = torch.int64
        elif isinstance(action_space, spaces.MultiBinary):
            action_dtype = torch.float32
        elif isinstance(action_space, spaces.Box):
            action_dtype = torch.float32
        else:
            raise TypeError(f"unsupported PPO action space {type(action_space).__name__}")
        batch_shape = (n_steps, n_envs)
        return cls(
            observations=_allocate_observations(observations, n_steps, device),
            actions=torch.empty((*batch_shape, *action_shape), dtype=action_dtype, device=device),
            rewards=torch.empty(batch_shape, dtype=torch.float32, device=device),
            episode_starts=torch.empty(batch_shape, dtype=torch.bool, device=device),
            values=torch.empty(batch_shape, dtype=torch.float32, device=device),
            log_probs=torch.empty(batch_shape, dtype=torch.float32, device=device),
            advantages=torch.empty(batch_shape, dtype=torch.float32, device=device),
            returns=torch.empty(batch_shape, dtype=torch.float32, device=device),
        )

    @property
    def n_steps(self) -> int:
        return int(self.rewards.shape[0])

    @property
    def n_envs(self) -> int:
        return int(self.rewards.shape[1])

    @property
    def size(self) -> int:
        return self.n_steps * self.n_envs

    def add(
        self,
        observations: Any,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        episode_starts: torch.Tensor,
        values: torch.Tensor,
        log_probs: torch.Tensor,
    ) -> None:
        if self.position >= self.n_steps:
            raise RuntimeError("rollout buffer overflow")
        _copy_observation_slot(self.observations, observations, self.position)
        self.actions[self.position].copy_(actions.reshape_as(self.actions[self.position]))
        self.rewards[self.position].copy_(rewards)
        self.episode_starts[self.position].copy_(episode_starts)
        self.values[self.position].copy_(values.flatten().float())
        self.log_probs[self.position].copy_(log_probs.flatten().float())
        self.position += 1

    def finish(
        self,
        *,
        last_values: torch.Tensor,
        dones: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        if self.position != self.n_steps:
            raise RuntimeError("cannot finish an incomplete rollout")
        last_gae = torch.zeros(self.n_envs, dtype=torch.float32, device=self.rewards.device)
        for step in range(self.n_steps - 1, -1, -1):
            if step == self.n_steps - 1:
                next_non_terminal = ~dones
                next_values = last_values.flatten().float()
            else:
                next_non_terminal = ~self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = (
                self.rewards[step]
                + float(gamma) * next_values * next_non_terminal.float()
                - self.values[step]
            )
            last_gae = (
                delta
                + float(gamma)
                * float(gae_lambda)
                * next_non_terminal.float()
                * last_gae
            )
            self.advantages[step].copy_(last_gae)
        self.returns.copy_(self.advantages + self.values)


class _CompiledPolicyCalls:
    def __init__(
        self,
        policy: Any,
        device: torch.device,
        *,
        compile_policy: bool = True,
    ) -> None:
        self.device = device
        if device.type == "cuda" and compile_policy:
            self.forward = torch.compile(policy.forward, dynamic=False, fullgraph=False)
            self.evaluate_actions = torch.compile(
                policy.evaluate_actions,
                dynamic=False,
                fullgraph=False,
            )
            self.predict_values = torch.compile(
                policy.predict_values,
                dynamic=True,
                fullgraph=False,
            )
        else:
            self.forward = policy.forward
            self.evaluate_actions = policy.evaluate_actions
            self.predict_values = policy.predict_values


class _Precision:
    def __init__(self, name: str, device: torch.device) -> None:
        self.name = name
        self.device = device
        if name != "fp32" and device.type != "cuda":
            raise ValueError(f"{name} precision requires a CUDA device")
        self.dtype = {
            "fp32": torch.float32,
            "amp-fp16": torch.float16,
            "amp-bf16": torch.bfloat16,
        }[name]
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=name == "amp-fp16",
        )

    def autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=self.name != "fp32",
        )


@dataclass(frozen=True)
class _CompletedRollout:
    step: int
    steps: int
    started_at: float
    ended_at: float
    rollout_seconds: float
    env_step_seconds: float | None


class _ThroughputTracker:
    def __init__(self, context: BackendContext, runtime: Any, device: torch.device) -> None:
        self.context = context
        self.runtime = runtime
        self.device = device
        self.completed: _CompletedRollout | None = None
        self.started_at = 0.0
        self.started_step = 0
        self.native_start: Mapping[str, float | int] | None = None

    def begin(self, step: int) -> None:
        _synchronize(self.device)
        now = time.perf_counter()
        if self.completed is not None:
            self._publish(self.completed, next_start=now)
            self.completed = None
        self.started_at = now
        self.started_step = int(step)
        self.native_start = self.runtime.native_step_stats()

    def end(self, step: int) -> None:
        _synchronize(self.device)
        now = time.perf_counter()
        native_end = self.runtime.native_step_stats()
        native_seconds: float | None = None
        if self.native_start is not None:
            calls = int(native_end["calls_total"]) - int(self.native_start["calls_total"])
            elapsed = float(native_end["seconds_total"]) - float(
                self.native_start["seconds_total"]
            )
            if calls > 0 and elapsed > 0.0:
                native_seconds = elapsed
        self.completed = _CompletedRollout(
            step=int(step),
            steps=int(step) - self.started_step,
            started_at=self.started_at,
            ended_at=now,
            rollout_seconds=now - self.started_at,
            env_step_seconds=native_seconds,
        )

    def flush(self) -> None:
        if self.completed is None:
            return
        _synchronize(self.device)
        self._publish(self.completed, next_start=time.perf_counter())
        self.completed = None

    def _publish(self, rollout: _CompletedRollout, *, next_start: float) -> None:
        loop_seconds = next_start - rollout.started_at
        between_seconds = next_start - rollout.ended_at
        if rollout.steps <= 0 or loop_seconds <= 0.0 or between_seconds < 0.0:
            return
        payload: dict[str, float] = {
            TRAIN_THROUGHPUT_LOOP_FPS: rollout.steps / loop_seconds,
            TRAIN_THROUGHPUT_ROLLOUT_SECONDS: rollout.rollout_seconds,
            TRAIN_THROUGHPUT_BETWEEN_ROLLOUTS_SECONDS: between_seconds,
        }
        if rollout.env_step_seconds is not None:
            payload.update(
                {
                    TRAIN_THROUGHPUT_ENV_STEP_FPS: rollout.steps / rollout.env_step_seconds,
                    TRAIN_THROUGHPUT_ENV_STEP_SECONDS: rollout.env_step_seconds,
                    TRAIN_THROUGHPUT_ROLLOUT_OVERHEAD_SECONDS: max(
                        rollout.rollout_seconds - rollout.env_step_seconds,
                        0.0,
                    ),
                }
            )
        validate_metric_payload(payload)
        self.context.session.metric_sink.publish(payload, step=rollout.step)


class _CurriculumFeedback:
    _METRIC_MAP = ARCHIVE_CURRICULUM_METRIC_MAP

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.enabled = runtime.archive_curriculum is not None
        self.steps: list[CurriculumStepAttribution] = []
        self.fragments: dict[tuple[int, int, int, str], list[float | int]] = {}

    def begin(self) -> None:
        self.steps = []
        if self.enabled:
            self.runtime.curriculum_begin_rollout()

    def capture(self, step: Any) -> None:
        if not self.enabled:
            return
        self.steps.append(
            CurriculumStepAttribution(
                curriculum_cell_ids=np.asarray(step.curriculum_cell_ids, dtype=object).copy(),
                curriculum_generations=np.asarray(
                    step.curriculum_generations, dtype=np.int64
                ).copy(),
                curriculum_episode_indices=np.asarray(
                    step.curriculum_episode_indices, dtype=np.int64
                ).copy(),
                curriculum_feedback_dones=np.asarray(
                    step.curriculum_feedback_dones, dtype=np.bool_
                ).copy(),
                control_boundaries=np.asarray(step.control_boundaries, dtype=np.bool_).copy(),
            )
        )

    def complete(self, raw_advantages: torch.Tensor) -> dict[str, float]:
        if not self.enabled:
            return {}
        advantages = raw_advantages.detach().float().cpu().numpy()
        if advantages.shape[0] != len(self.steps):
            raise RuntimeError("state archive attribution does not align with rollout")
        completed: list[tuple[tuple[int, int, int, str], float]] = []
        for step_index, attribution in enumerate(self.steps):
            for lane in range(advantages.shape[1]):
                cell_id = attribution.curriculum_cell_ids[lane]
                if cell_id is None:
                    continue
                key = (
                    int(attribution.curriculum_generations[lane]),
                    lane,
                    int(attribution.curriculum_episode_indices[lane]),
                    str(cell_id),
                )
                fragment = self.fragments.setdefault(key, [0.0, 0])
                value = float(abs(advantages[step_index, lane]))
                if not math.isfinite(value):
                    raise RuntimeError("state archive received non-finite raw GAE")
                fragment[0] = float(fragment[0]) + value
                fragment[1] = int(fragment[1]) + 1
                if bool(attribution.curriculum_feedback_dones[lane]):
                    total, count = self.fragments.pop(key)
                    completed.append((key, float(total) / int(count)))
        for key, value_error in sorted(completed, key=lambda item: item[0]):
            self.runtime.submit_curriculum_feedback(key[3], value_error)
        internal = self.runtime.curriculum_complete_rollout()
        self.steps = []
        return {
            metric_name: float(internal.get(internal_name, 0.0))
            for internal_name, metric_name in self._METRIC_MAP.items()
        }


def _validate_grouped_context(config: Any, backend_config: Mapping[str, Any]) -> tuple[str, str | None]:
    from gradlab.model_inputs import model_input_fields
    from gradlab.task_advantage import resolve_advantage_normalization_mode

    mode, context = resolve_advantage_normalization_mode(backend_config)
    if mode == "grouped":
        fields = model_input_fields(config.task)
        field = fields.get(str(context))
        if field is None:
            raise ValueError(
                "grouped advantage normalization references undeclared context "
                f"{context!r}"
            )
        if field["encoding"]["kind"] != "categorical":
            raise ValueError(
                "grouped advantage normalization requires categorical context, got "
                f"{context!r}"
            )
    return mode, context


def _configure_optimizer_for_device(
    optimizer: Any,
    device: torch.device,
    *,
    fused: bool = True,
) -> None:
    use_fused = device.type == "cuda" and fused
    optimizer.defaults["fused"] = use_fused
    optimizer.defaults["capturable"] = False
    optimizer.defaults["foreach"] = False if use_fused else None
    for group in optimizer.param_groups:
        group["fused"] = use_fused
        group["capturable"] = False
        group["foreach"] = False if use_fused else None


@dataclass(frozen=True)
class _ExecutionProfile:
    compile_policy: bool
    fused_optimizer: bool
    torch_permutation: bool


_EXECUTION_PROFILES = {
    "sb3-parity": _ExecutionProfile(False, False, False),
    "compiled-parity": _ExecutionProfile(True, False, False),
    "compiled-fused-parity": _ExecutionProfile(True, True, False),
    "max-throughput": _ExecutionProfile(True, True, True),
}


def _make_model(
    context: BackendContext,
    env: Any,
    config: Any,
    device_name: str,
    *,
    fused_optimizer: bool = True,
) -> tuple[GradLabPPO, str, str | None]:
    from gradlab.policy_models import load_pinned_remote_policy_model
    from gradlab.schedules import apply_resume_hyperparameters, learning_rate_schedule

    common_config = context.train_config
    backend_config = context.backend_config
    normalization_mode, advantage_context = _validate_grouped_context(config, backend_config)
    if backend_config["resume"]:
        model = load_pinned_remote_policy_model(
            backend_config["resume"],
            download_root=context.run_dir / ".resume-source",
            approval_hash=backend_config["resume_approval_hash"],
            manifest=backend_config["resume_manifest"],
            expected_algorithm_id="ppo",
            env=env,
            tensorboard_log=str(context.run_dir),
            device=device_name,
            ppo_model_class=GradLabPPO,
        )
        validate_resumed_policy_model(model, common_config)
        loaded_context = str(getattr(model, "advantage_context", "") or "")
        if normalization_mode == "grouped" and loaded_context != advantage_context:
            raise ValueError(
                "resume artifact grouped advantage context does not match the recipe"
            )
        if normalization_mode != "grouped" and loaded_context:
            raise ValueError(
                "resume artifact uses grouped advantage normalization but the recipe does not"
            )
        apply_resume_hyperparameters(model, common_config, backend_config)
        model.gamma = float(backend_config["gamma"])
        model.gae_lambda = float(backend_config["gae_lambda"])
        model.clip_range_vf = (
            None
            if backend_config["clip_range_vf"] is None
            else get_schedule_fn(backend_config["clip_range_vf"])
        )
    else:
        model = GradLabPPO(
            policy_type_for_config(env.observation_space, common_config),
            env,
            learning_rate=learning_rate_schedule(common_config, backend_config),
            n_steps=backend_config["n_steps"],
            batch_size=backend_config["batch_size"],
            n_epochs=backend_config["n_epochs"],
            gamma=backend_config["gamma"],
            gae_lambda=backend_config["gae_lambda"],
            ent_coef=backend_config["ent_coef"],
            vf_coef=backend_config["vf_coef"],
            clip_range=backend_config["clip_range"],
            clip_range_vf=backend_config["clip_range_vf"],
            normalize_advantage=normalization_mode == "global",
            target_kl=backend_config["target_kl"],
            policy_kwargs=policy_kwargs_from_config(
                backend_config,
                common_config=common_config,
                optimizer_eps=backend_config["adam_eps"],
            ),
            tensorboard_log=str(context.run_dir),
            device=device_name,
            verbose=0,
        )
        model.rollout_buffer = None
    model.n_steps = int(backend_config["n_steps"])
    model.batch_size = int(backend_config["batch_size"])
    model.n_epochs = int(backend_config["n_epochs"])
    model.normalize_advantage = normalization_mode == "global"
    model.advantage_context = advantage_context or ""
    _configure_optimizer_for_device(
        model.policy.optimizer,
        torch.device(device_name),
        fused=fused_optimizer,
    )
    return model, normalization_mode, advantage_context


def _preflight_cuda_memory(
    observations: Any,
    *,
    model: GradLabPPO,
    n_steps: int,
    n_envs: int,
    action_space: spaces.Space,
    device: torch.device,
) -> None:
    if device.type != "cuda":
        return
    observation_bytes = _tree_nbytes(observations) * n_steps
    action_width = get_action_dim(action_space)
    action_bytes = 8 if isinstance(
        action_space,
        spaces.Discrete | spaces.MultiDiscrete,
    ) else 4
    scalar_bytes = 6 * 4 + 1
    rollout_bytes = observation_bytes + n_steps * n_envs * (
        action_width * action_bytes + scalar_bytes
    )
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.policy.parameters()
    )
    # Gradients plus Adam's two moment tensors are allocated lazily. Reserve
    # another parameter copy and 512 MiB for compiled graphs/activations.
    estimate = rollout_bytes + 4 * parameter_bytes + 512 * 1024**2
    free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
    required = math.ceil(estimate * 1.25)
    if required > int(free_bytes * 0.8):
        raise MemoryError(
            "gradlab.ppo rollout buffer preflight failed: "
            f"estimated_required={required} available={free_bytes}; reduce n_steps or n_envs"
        )


def _normalize_grouped_advantages(
    buffer: TensorRolloutBuffer,
    context: str,
) -> None:
    if not isinstance(buffer.observations, Mapping):
        raise ValueError("grouped advantage normalization requires dict observations")
    key = f"context/{context}"
    if key not in buffer.observations:
        raise ValueError(
            f"grouped advantage normalization requires observations with a {key!r} key"
        )
    task_ids = buffer.observations[key]
    if task_ids.shape == (*buffer.advantages.shape, 1):
        task_ids = task_ids[..., 0]
    if task_ids.shape != buffer.advantages.shape:
        raise ValueError("categorical context shape must match advantages")
    if task_ids.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError(f"grouped context {context!r} must contain integer category indices")
    for task_id in torch.unique(task_ids):
        if int(task_id.item()) < 0:
            raise ValueError(f"grouped context {context!r} contains a negative category index")
        mask = task_ids == task_id
        values = buffer.advantages[mask]
        if values.numel() > 1:
            buffer.advantages[mask] = (
                values - values.mean()
            ) / (values.std(correction=0) + 1e-8)


def _rollout_diagnostics(buffer: TensorRolloutBuffer, action_space: spaces.Space) -> dict[str, float]:
    payload: dict[str, float] = {}
    for suffix, values in (
        ("rollout/value_prediction", buffer.values),
        ("rollout/advantage", buffer.advantages),
    ):
        finite = values[torch.isfinite(values)].float()
        if finite.numel() == 0:
            continue
        prefix = train_algorithm_metric("ppo", suffix)
        payload.update(
            {
                stat_metric(prefix, "mean"): float(finite.mean().item()),
                stat_metric(prefix, "std"): float(finite.std(correction=0).item()),
                stat_metric(prefix, "min"): float(finite.min().item()),
                stat_metric(prefix, "max"): float(finite.max().item()),
            }
        )
    if isinstance(action_space, spaces.Discrete):
        counts = torch.bincount(buffer.actions.flatten().long())
        if counts.numel() > 0:
            payload[train_algorithm_metric("ppo", "policy/dominant_action_rate")] = float(
                counts.max().item() / buffer.actions.numel()
            )
    bounds = policy_entropy_bounds(action_space)
    if bounds is not None:
        payload[TRAIN_PPO_POLICY_ENTROPY_BOUND_LOWER] = bounds[0]
        payload[TRAIN_PPO_POLICY_ENTROPY_BOUND_UPPER] = bounds[1]
    return payload


def _ppo_update(
    model: GradLabPPO,
    buffer: TensorRolloutBuffer,
    *,
    calls: _CompiledPolicyCalls,
    precision: _Precision,
    progress_remaining: float,
    normalization_mode: str,
    advantage_context: str | None,
    ent_coef: float,
    torch_permutation: bool = True,
) -> dict[str, float]:
    model.policy.set_training_mode(True)
    learning_rate = float(model.lr_schedule(progress_remaining))
    for group in model.policy.optimizer.param_groups:
        group["lr"] = learning_rate
    clip_range = float(model.clip_range(progress_remaining))
    clip_range_vf = (
        None
        if model.clip_range_vf is None
        else float(model.clip_range_vf(progress_remaining))
    )
    if normalization_mode == "grouped":
        assert advantage_context is not None
        _normalize_grouped_advantages(buffer, advantage_context)

    env_major = not torch_permutation
    flat_observations = _flatten_observations(
        buffer.observations,
        env_major=env_major,
    )
    flat_actions = _flatten_rollout_tensor(buffer.actions, env_major=env_major)
    flat_values = _flatten_rollout_tensor(buffer.values, env_major=env_major).flatten()
    flat_log_probs = _flatten_rollout_tensor(buffer.log_probs, env_major=env_major).flatten()
    flat_advantages = _flatten_rollout_tensor(buffer.advantages, env_major=env_major).flatten()
    flat_returns = _flatten_rollout_tensor(buffer.returns, env_major=env_major).flatten()
    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropy_losses: list[float] = []
    clip_fractions: list[float] = []
    last_epoch_kls: list[float] = []
    continue_training = True
    final_loss = torch.zeros((), device=buffer.rewards.device)

    for _epoch in range(int(model.n_epochs)):
        last_epoch_kls = []
        if torch_permutation:
            indices = torch.randperm(buffer.size, device=buffer.rewards.device)
        else:
            indices = torch.as_tensor(
                np.random.permutation(buffer.size),
                dtype=torch.int64,
                device=buffer.rewards.device,
            )
        for start in range(0, buffer.size, int(model.batch_size)):
            batch_indices = indices[start : start + int(model.batch_size)]
            observations = _index_observations(flat_observations, batch_indices)
            actions = flat_actions.index_select(0, batch_indices)
            if isinstance(model.action_space, spaces.Discrete):
                actions = actions.long().flatten()
            old_values = flat_values.index_select(0, batch_indices)
            old_log_probs = flat_log_probs.index_select(0, batch_indices)
            advantages = flat_advantages.index_select(0, batch_indices)
            returns = flat_returns.index_select(0, batch_indices)
            if normalization_mode == "global" and advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

            with precision.autocast():
                values, log_prob, entropy = calls.evaluate_actions(observations, actions)
                values = values.flatten()
                ratio = torch.exp(log_prob - old_log_probs)
                policy_loss = -torch.min(
                    advantages * ratio,
                    advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range),
                ).mean()
                if clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = old_values + torch.clamp(
                        values - old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                value_loss = F.mse_loss(returns, values_pred)
                entropy_loss = (
                    log_prob.mean() if entropy is None else -entropy.mean()
                )
                final_loss = (
                    policy_loss
                    + float(ent_coef) * entropy_loss
                    + float(model.vf_coef) * value_loss
                )

            with torch.no_grad():
                log_ratio = log_prob - old_log_probs
                approx_kl = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean()
                kl_value = float(approx_kl.float().item())
                last_epoch_kls.append(kl_value)
                clip_fractions.append(
                    float(((ratio - 1.0).abs() > clip_range).float().mean().item())
                )
            policy_losses.append(float(policy_loss.float().item()))
            value_losses.append(float(value_loss.float().item()))
            entropy_losses.append(float(entropy_loss.float().item()))
            if model.target_kl is not None and kl_value > 1.5 * float(model.target_kl):
                continue_training = False
                break

            model.policy.optimizer.zero_grad(set_to_none=True)
            if precision.scaler.is_enabled():
                precision.scaler.scale(final_loss).backward()
                precision.scaler.unscale_(model.policy.optimizer)
                torch.nn.utils.clip_grad_norm_(model.policy.parameters(), model.max_grad_norm)
                precision.scaler.step(model.policy.optimizer)
                precision.scaler.update()
            else:
                final_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.policy.parameters(), model.max_grad_norm)
                model.policy.optimizer.step()
        model._n_updates += 1
        if not continue_training:
            break

    returns_variance = torch.var(flat_returns, correction=0)
    explained_variance = (
        float("nan")
        if float(returns_variance.item()) == 0.0
        else float(
            (
                1.0
                - torch.var(flat_returns - flat_values, correction=0) / returns_variance
            ).item()
        )
    )
    payload: dict[str, float] = {
        TRAIN_PPO_APPROX_KL: float(np.mean(last_epoch_kls)) if last_epoch_kls else 0.0,
        TRAIN_PPO_CLIP_FRACTION: float(np.mean(clip_fractions)) if clip_fractions else 0.0,
        TRAIN_PPO_VALUE_LOSS: float(np.mean(value_losses)) if value_losses else 0.0,
        TRAIN_PPO_LEARNING_RATE: learning_rate,
        TRAIN_PPO_POLICY_ENTROPY: -float(np.mean(entropy_losses)) if entropy_losses else 0.0,
        train_algorithm_metric("ppo", "update/policy_gradient_loss"): (
            float(np.mean(policy_losses)) if policy_losses else 0.0
        ),
        train_algorithm_metric("ppo", "hyperparameter/entropy_coefficient"): float(ent_coef),
    }
    if math.isfinite(explained_variance):
        payload[TRAIN_PPO_EXPLAINED_VARIANCE] = explained_variance
    if hasattr(model.policy, "log_std"):
        payload[train_algorithm_metric("ppo", "policy/distribution_std")] = float(
            torch.exp(model.policy.log_std).mean().detach().item()
        )
    validate_metric_payload(payload)
    return payload


def _environment_actions(
    actions: torch.Tensor,
    action_space: spaces.Space,
    policy: Any,
) -> np.ndarray:
    values = actions.detach().cpu().numpy()
    if isinstance(action_space, spaces.Box):
        if policy.squash_output:
            values = policy.unscale_action(values)
        else:
            values = np.clip(values, action_space.low, action_space.high)
    return values


def _entropy_coefficient(config: Mapping[str, Any], step: int, total: int) -> float:
    final = config["ent_coef_final"]
    if final is None:
        return float(config["ent_coef"])
    duration = int(config["ent_coef_schedule_timesteps"] or total)
    progress = min(max(int(step) / duration, 0.0), 1.0)
    return float(config["ent_coef"]) + (float(final) - float(config["ent_coef"])) * progress


def run_gradlab_ppo(
    context: BackendContext,
    *,
    progress_fields: Sequence[ProgressField] = (),
) -> TrainingResult:
    from stable_baselines3.common.utils import set_random_seed

    from gradlab.device import resolve_sb3_device
    from gradlab.env import make_training_vec_env, preflight_state_archive_provider
    from gradlab.file_utils import file_sha256
    from gradlab.policy_bundle import write_canonical_json
    from gradlab.training.sb3_helpers import GracefulStopHelper

    common_config = context.train_config
    backend_config = context.backend_config
    config = context.environment
    n_envs = int(common_config["resolved_n_envs"])
    preflight = preflight_state_archive_provider(
        config=config,
        n_envs=n_envs,
        seed=int(common_config["seed"]),
        rom_binding=getattr(context, "rom_binding", None),
        state_archive=common_config.get("state_archive"),
    )
    if preflight is not None:
        path = context.run_dir / "state_archive_preflight.json"
        write_canonical_json(path, preflight)
        common_config["state_archive_preflight_sha256"] = file_sha256(path)
        context.session.event(
            "state archive provider preflight passed: "
            f"provider={preflight['provider_id']} codec={preflight['codec_id']} "
            f"lanes={preflight['preflight_lanes']}"
        )
    env = make_training_vec_env(
        config=config,
        n_envs=n_envs,
        seed=int(common_config["seed"]),
        rom_binding=getattr(context, "rom_binding", None),
        state_archive=common_config.get("state_archive"),
        state_archive_root=context.run_dir / "state-archive",
    )
    runtime = env.runtime
    try:
        set_random_seed(int(common_config["seed"]))
        validate_action_space(env.action_space, algorithm_id="ppo")
        device_name = resolve_sb3_device(str(backend_config["device"]))
        device = torch.device(device_name)
        context.session.event(f"using torch device: {device}")
        execution_profile_name = str(backend_config["execution_profile"])
        execution_profile = _EXECUTION_PROFILES[execution_profile_name]
        context.session.event(
            "gradlab.ppo execution profile: "
            f"{execution_profile_name} "
            f"compile={execution_profile.compile_policy} "
            f"fused_optimizer={execution_profile.fused_optimizer} "
            f"torch_permutation={execution_profile.torch_permutation}"
        )
        model, normalization_mode, advantage_context = _make_model(
            context,
            env,
            config,
            device_name,
            fused_optimizer=execution_profile.fused_optimizer,
        )
        rollout_quantum = n_envs * int(backend_config["n_steps"])
        budget = context.session.configure_budget(
            requested_limit=int(common_config["timesteps"]),
            step_quantum=rollout_quantum,
            initial_step=int(model.num_timesteps),
            progress_fields=progress_fields,
        )
        model._total_timesteps = int(budget.execution_total)
        graceful_stop = GracefulStopHelper(
            context.stop_flag,
            marker_path=context.run_dir / "learner_stop_observed.json",
            event=context.session.event,
        )
        if (
            common_config["checkpoint_eval_backend"] == "none"
            and context.session.execution_policy.mode != TrainingExecutionMode.LOCAL_DEMO
        ):
            context.session.event(
                "checkpoint evaluation disabled; this run cannot establish promotion or acceptance"
            )
        elif common_config["checkpoint_eval_backend"] != "none":
            context.session.event(
                "training-loop eval disabled; async checkpoint eval handles promotion metrics"
            )

        observations = runtime.reset(seed=int(common_config["seed"]))
        _preflight_cuda_memory(
            observations,
            model=model,
            n_steps=int(backend_config["n_steps"]),
            n_envs=n_envs,
            action_space=env.action_space,
            device=device,
        )
        buffer = TensorRolloutBuffer.allocate(
            observations,
            action_space=env.action_space,
            n_steps=int(backend_config["n_steps"]),
            n_envs=n_envs,
            device=device,
        )
        calls = _CompiledPolicyCalls(
            model.policy,
            device,
            compile_policy=execution_profile.compile_policy,
        )
        precision = _Precision(str(backend_config["precision"]), device)
        reward_stats = RewardStatsAccumulator(
            active_components=active_reward_components(config.task),
            active_signals=active_reward_signals(config.task),
        )
        curriculum = _CurriculumFeedback(runtime)
        throughput = _ThroughputTracker(context, runtime, device)
        episode_starts = torch.ones(n_envs, dtype=torch.bool, device=device)
        dones = torch.zeros(n_envs, dtype=torch.bool, device=device)
        checkpoint_calls = checkpoint_save_frequency(
            int(common_config["checkpoint_freq"]),
            n_envs,
        )
        calls_since_start = 0
        rollout_count = 0
        context.mark_ready()

        while int(model.num_timesteps) < budget.execution_total:
            if context.stop_flag.requested:
                graceful_stop.acknowledge_safe_boundary(
                    num_timesteps=int(model.num_timesteps)
                )
                break
            buffer.position = 0
            curriculum.begin()
            model.policy.set_training_mode(False)
            throughput.begin(int(model.num_timesteps))
            for _step in range(int(backend_config["n_steps"])):
                obs_tensor = _observation_tensor(observations, device)
                with torch.no_grad(), precision.autocast():
                    actions, values, log_probs = calls.forward(obs_tensor)
                environment_actions = _environment_actions(
                    actions,
                    env.action_space,
                    model.policy,
                )
                batch_step = runtime.step(environment_actions)
                reward_tensor = torch.as_tensor(
                    batch_step.rewards,
                    device=device,
                    dtype=torch.float32,
                ).clone()
                truncated_lanes = np.flatnonzero(batch_step.truncated)
                if truncated_lanes.size:
                    if batch_step.final_observations is None:
                        raise RuntimeError("truncated batch is missing final observations")
                    terminal_observations = _take_observation_lanes(
                        batch_step.final_observations,
                        truncated_lanes,
                    )
                    terminal_tensor = _observation_tensor(terminal_observations, device)
                    with torch.no_grad(), precision.autocast():
                        terminal_values = calls.predict_values(terminal_tensor).flatten().float()
                    lane_tensor = torch.as_tensor(
                        truncated_lanes,
                        dtype=torch.int64,
                        device=device,
                    )
                    reward_tensor[lane_tensor] += float(model.gamma) * terminal_values
                buffer.add(
                    obs_tensor,
                    actions,
                    reward_tensor,
                    episode_starts,
                    values,
                    log_probs,
                )
                curriculum.capture(batch_step)
                observations = batch_step.observations
                done_array = np.logical_or(batch_step.terminated, batch_step.truncated)
                dones = torch.as_tensor(done_array, device=device).clone()
                episode_starts = dones
                model.num_timesteps += n_envs
                calls_since_start += 1
                records = runtime.drain_records()
                episodes: list[Any] = []
                for record in records:
                    if isinstance(record, BatchMetricRecord):
                        reward_stats.consume(
                            record.metrics,
                            reserve=rollout_quantum,
                        )
                    elif hasattr(record, "episode_return"):
                        episodes.append(record)
                context.session.advance(int(model.num_timesteps), episodes)
                context.session.observe_episode_completions(
                    step=int(model.num_timesteps),
                    records=episodes,
                )
                if (
                    checkpoint_calls is not None
                    and calls_since_start % checkpoint_calls == 0
                    and int(model.num_timesteps) < int(common_config["timesteps"])
                ):
                    step = int(model.num_timesteps)
                    save_model_bundle(
                        model=model,
                        context=context,
                        model_path=context.checkpoint_dir
                        / f"{checkpoint_prefix(config.game, algorithm_id='ppo')}_{step}_steps.zip",
                        kind="checkpoint",
                        step=step,
                    )

            next_observations = _observation_tensor(observations, device)
            with torch.no_grad(), precision.autocast():
                last_values = calls.predict_values(next_observations)
            buffer.finish(
                last_values=last_values,
                dones=dones,
                gamma=float(model.gamma),
                gae_lambda=float(model.gae_lambda),
            )
            throughput.end(int(model.num_timesteps))
            raw_advantages = buffer.advantages.clone() if curriculum.enabled else buffer.advantages
            curriculum_metrics = curriculum.complete(raw_advantages)
            rollout_metrics = _rollout_diagnostics(buffer, env.action_space)
            progress_remaining = max(
                1.0 - int(model.num_timesteps) / max(int(common_config["timesteps"]), 1),
                0.0,
            )
            model._current_progress_remaining = progress_remaining
            ent_coef = _entropy_coefficient(
                backend_config,
                int(model.num_timesteps),
                int(common_config["timesteps"]),
            )
            model.ent_coef = ent_coef
            update_metrics = _ppo_update(
                model,
                buffer,
                calls=calls,
                precision=precision,
                progress_remaining=progress_remaining,
                normalization_mode=normalization_mode,
                advantage_context=advantage_context,
                ent_coef=ent_coef,
                torch_permutation=execution_profile.torch_permutation,
            )
            rollout_metrics.update(curriculum_metrics)
            rollout_metrics.update(reward_stats.flush())
            rollout_metrics.update(update_metrics)
            progress_metrics = {
                "train/approx_kl": update_metrics[TRAIN_PPO_APPROX_KL],
                "train/entropy_loss": -update_metrics[TRAIN_PPO_POLICY_ENTROPY],
            }
            if TRAIN_PPO_EXPLAINED_VARIANCE in update_metrics:
                progress_metrics["train/explained_variance"] = update_metrics[
                    TRAIN_PPO_EXPLAINED_VARIANCE
                ]
            context.session.advance(
                int(model.num_timesteps),
                progress_metrics=progress_metrics,
            )
            context.session.report(
                step=int(model.num_timesteps),
                metrics=rollout_metrics,
            )
            rollout_count += 1
            if rollout_count % 64 == 0 and context.wandb_enabled:
                context.metric_store.enqueue_event(
                    kind="histogram",
                    payload={
                        "histograms": {
                            train_algorithm_metric(
                                "ppo", "rollout/value_prediction/hist"
                            ): buffer.values.detach().float().cpu().flatten().tolist(),
                            train_algorithm_metric(
                                "ppo", "rollout/advantage/hist"
                            ): buffer.advantages.detach().float().cpu().flatten().tolist(),
                        }
                    },
                    step=int(model.num_timesteps),
                    source="train",
                )
            if context.stop_flag.requested:
                graceful_stop.acknowledge_safe_boundary(
                    num_timesteps=int(model.num_timesteps)
                )
                break

        throughput.flush()
        reason = context.session.terminal_reason()
        if (
            context.session.should_persist_interrupted_checkpoint(reason)
            and int(common_config["checkpoint_freq"]) > 0
        ):
            step = int(model.num_timesteps)
            save_model_bundle(
                model=model,
                context=context,
                model_path=context.checkpoint_dir
                / f"{checkpoint_prefix(config.game, algorithm_id='ppo')}"
                f"_interrupted_{step}_steps.zip",
                kind="interrupted",
                step=step,
            )
        terminal_kind = context.session.terminal_model_kind(reason)
        context.train_config["training_terminal"] = context.session.terminal_provenance(
            terminal_reason=reason,
            final_step=int(model.num_timesteps),
        )
        final_model_path = context.run_dir / "final_model.zip"
        save_model_bundle(
            model=model,
            context=context,
            model_path=final_model_path,
            kind=terminal_kind,
            step=int(model.num_timesteps),
            terminal=True,
        )
        context.session.event(f"saved {final_model_path}")
        return context.session.result(
            terminal_reason=reason,
            final_step=int(model.num_timesteps),
            model_kind=terminal_kind,
        )
    finally:
        env.close()
