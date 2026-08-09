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

from gradlab.action_codecs import LegalTupleMultiDiscrete
from gradlab.batch_runtime import BatchMetricRecord, CurriculumStepAttribution
from gradlab.callbacks import (
    ARCHIVE_CURRICULUM_METRIC_MAP,
    RewardStatsAccumulator,
)
from gradlab.metric_names import (
    TRAIN_PPO_APPROX_KL,
    TRAIN_PPO_CLIP_FRACTION,
    TRAIN_PPO_EXPLAINED_VARIANCE,
    TRAIN_PPO_LEARNING_RATE,
    TRAIN_PPO_POLICY_ENTROPY,
    TRAIN_PPO_VALUE_LOSS,
    stat_metric,
    train_algorithm_metric,
    validate_metric_payload,
)
from gradlab.ppo import GradLabPPO
from gradlab.training.sb3_on_policy import (
    active_reward_components,
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
from gradlab.training_metrics import throughput_delta_metrics


ObservationTree = torch.Tensor | dict[str, "ObservationTree"]


def _tree_map(value: Any, function: Callable[[Any], Any]) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _tree_map(item, function) for key, item in value.items()}
    return function(value)


def _allocate_observations(observations: Any, n_steps: int, device: torch.device) -> Any:
    def allocate(value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return torch.empty(
                (n_steps, *value.shape),
                dtype=value.dtype,
                device=device,
            )
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
    destination[step].copy_(torch.as_tensor(observations))


def _observation_slot(observations: Any, step: int) -> Any:
    if isinstance(observations, Mapping):
        return {key: _observation_slot(value, step) for key, value in observations.items()}
    return observations[step]


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
    if isinstance(observations, torch.Tensor):
        return observations.numel() * observations.element_size()
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
    final_observations: Any | None = None
    truncated: torch.Tensor | None = None
    position: int = 0
    _step_open: bool = False

    @classmethod
    def allocate(
        cls,
        observations: Any,
        *,
        action_space: spaces.Space,
        n_steps: int,
        n_envs: int,
        device: torch.device,
        store_final_observations: bool = False,
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
            final_observations=(
                _allocate_observations(observations, n_steps, device)
                if store_final_observations
                else None
            ),
            truncated=(
                torch.empty(batch_shape, dtype=torch.bool, device=device)
                if store_final_observations
                else None
            ),
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

    def reset(self) -> None:
        if self._step_open:
            raise RuntimeError("cannot reset a rollout buffer with an open step")
        self.position = 0

    def begin_step(
        self,
        observations: Any,
        episode_starts: torch.Tensor,
    ) -> Any:
        if self._step_open:
            raise RuntimeError("rollout buffer step is already open")
        if self.position >= self.n_steps:
            raise RuntimeError("rollout buffer overflow")
        _copy_observation_slot(self.observations, observations, self.position)
        self.episode_starts[self.position].copy_(episode_starts)
        self._step_open = True
        return _observation_slot(self.observations, self.position)

    def end_step(
        self,
        actions: torch.Tensor,
        rewards: Any,
        values: torch.Tensor,
        log_probs: torch.Tensor,
        final_observations: Any | None = None,
        truncated: Any | None = None,
    ) -> torch.Tensor:
        if not self._step_open:
            raise RuntimeError("rollout buffer step is not open")
        self.actions[self.position].copy_(actions.reshape_as(self.actions[self.position]))
        reward_slot = self.rewards[self.position]
        reward_slot.copy_(torch.as_tensor(rewards))
        self.values[self.position].copy_(values.flatten().float())
        self.log_probs[self.position].copy_(log_probs.flatten().float())
        if self.final_observations is not None:
            if final_observations is None or truncated is None or self.truncated is None:
                raise ValueError("device rollout requires final observations and truncation flags")
            _copy_observation_slot(self.final_observations, final_observations, self.position)
            self.truncated[self.position].copy_(truncated)
        self.position += 1
        self._step_open = False
        return reward_slot

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
                delta + float(gamma) * float(gae_lambda) * next_non_terminal.float() * last_gae
            )
            self.advantages[step].copy_(last_gae)
        self.returns.copy_(self.advantages + self.values)


def _bootstrap_device_time_limits(
    buffer: TensorRolloutBuffer,
    *,
    calls: _CompiledPolicyCalls,
    precision: _Precision,
    gamma: float,
) -> None:
    if buffer.final_observations is None or buffer.truncated is None:
        return
    flat_truncated = buffer.truncated.flatten()
    indices = torch.nonzero(flat_truncated, as_tuple=False).flatten()
    safe_indices = torch.cat((indices, torch.zeros(1, device=indices.device, dtype=torch.int64)))
    flat_final = _flatten_observations(buffer.final_observations)
    selected = _index_observations(flat_final, safe_indices)
    with torch.no_grad(), precision.autocast():
        selected_values = calls.predict_values(selected).flatten().float()
    bootstrap = torch.zeros_like(flat_truncated, dtype=torch.float32)
    bootstrap.index_copy_(0, indices, selected_values[:-1])
    buffer.rewards.add_(bootstrap.view_as(buffer.rewards) * float(gamma))


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
            elapsed = float(native_end["seconds_total"]) - float(self.native_start["seconds_total"])
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
        payload = throughput_delta_metrics(
            steps=rollout.steps,
            loop_seconds=loop_seconds,
            provider_step_seconds=rollout.env_step_seconds,
            rollout_seconds=rollout.rollout_seconds,
            between_rollouts_seconds=between_seconds,
        )
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


def _validate_grouped_context(
    config: Any, backend_config: Mapping[str, Any]
) -> tuple[str, str | None]:
    from gradlab.model_inputs import model_input_fields
    from gradlab.task_advantage import resolve_advantage_normalization_mode

    mode, context = resolve_advantage_normalization_mode(backend_config)
    if mode == "grouped":
        fields = model_input_fields(config.task)
        field = fields.get(str(context))
        if field is None:
            raise ValueError(
                f"grouped advantage normalization references undeclared context {context!r}"
            )
        if field["encoding"]["kind"] != "categorical":
            raise ValueError(
                f"grouped advantage normalization requires categorical context, got {context!r}"
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
            raise ValueError("resume artifact grouped advantage context does not match the recipe")
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
    store_final_observations: bool = False,
) -> None:
    if device.type != "cuda":
        return
    observation_bytes = (
        _tree_nbytes(observations) * n_steps * (2 if store_final_observations else 1)
    )
    action_width = get_action_dim(action_space)
    action_bytes = (
        8
        if isinstance(
            action_space,
            spaces.Discrete | spaces.MultiDiscrete,
        )
        else 4
    )
    scalar_bytes = 6 * 4 + 1
    rollout_bytes = observation_bytes + n_steps * n_envs * (
        action_width * action_bytes + scalar_bytes
    )
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.policy.parameters()
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
    if task_ids.dtype != torch.uint8 and bool(torch.any(task_ids < 0)):
        raise ValueError(f"grouped context {context!r} contains a negative category index")
    for task_id in torch.unique(task_ids):
        mask = task_ids == task_id
        values = buffer.advantages[mask]
        if values.numel() > 1:
            buffer.advantages[mask] = (values - values.mean()) / (values.std(correction=0) + 1e-8)


def _finite_mean_std(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = values.float()
    finite = torch.isfinite(values)
    count = finite.sum().float()
    safe_count = count.clamp_min(1.0)
    finite_values = torch.where(finite, values, torch.zeros_like(values))
    mean = finite_values.sum() / safe_count
    centered = torch.where(finite, values - mean, torch.zeros_like(values))
    std = torch.sqrt(centered.square().sum() / safe_count)
    nan = torch.full((), float("nan"), dtype=torch.float32, device=values.device)
    return torch.where(count > 0, mean, nan), torch.where(count > 0, std, nan)


def _dominant_action_rate(
    actions: torch.Tensor,
    action_space: spaces.Space,
) -> torch.Tensor | None:
    if isinstance(action_space, LegalTupleMultiDiscrete):
        axis_count = int(np.asarray(action_space.nvec).size)
        flattened = actions.reshape(-1, axis_count).long()
        nvec = tuple(int(value) for value in np.asarray(action_space.nvec).reshape(-1))
        encoded = flattened[:, 0].clone()
        multiplier = nvec[0]
        for axis in range(1, axis_count):
            encoded.add_(flattened[:, axis], alpha=multiplier)
            multiplier *= nvec[axis]
        counts = torch.bincount(encoded, minlength=multiplier)
        return counts.max().float() / max(int(flattened.shape[0]), 1)
    if not isinstance(action_space, spaces.Discrete):
        return None
    flattened = actions.reshape(-1).long()
    counts = torch.bincount(flattened, minlength=int(action_space.n))
    return counts.max().float() / max(int(flattened.numel()), 1)


def _rollout_diagnostics(
    buffer: TensorRolloutBuffer, action_space: spaces.Space
) -> tuple[dict[str, torch.Tensor], frozenset[str]]:
    payload: dict[str, torch.Tensor] = {}
    omit_if_nonfinite: set[str] = set()
    for suffix, values in (
        ("rollout/value/prediction", buffer.values),
        ("rollout/advantage", buffer.advantages),
    ):
        prefix = train_algorithm_metric("ppo", suffix)
        mean_name = stat_metric(prefix, "mean")
        std_name = stat_metric(prefix, "std")
        mean, std = _finite_mean_std(values)
        payload.update(
            {
                mean_name: mean,
                std_name: std,
            }
        )
        omit_if_nonfinite.update((mean_name, std_name))
    dominant_action_rate = _dominant_action_rate(buffer.actions.detach(), action_space)
    if dominant_action_rate is not None:
        payload[train_algorithm_metric("ppo", "policy/dominant/action/rate")] = dominant_action_rate
    return payload, frozenset(omit_if_nonfinite)


def _materialize_metrics(
    values: Mapping[str, float | torch.Tensor],
    *,
    omit_if_nonfinite: frozenset[str] = frozenset(),
) -> dict[str, float]:
    payload = {
        name: float(value) for name, value in values.items() if not isinstance(value, torch.Tensor)
    }
    tensor_items = [
        (name, value.detach().float().reshape(()))
        for name, value in values.items()
        if isinstance(value, torch.Tensor)
    ]
    if tensor_items:
        host_values = torch.stack([value for _name, value in tensor_items]).cpu().tolist()
        for (name, _value), host_value in zip(tensor_items, host_values, strict=True):
            materialized = float(host_value)
            if name in omit_if_nonfinite and not math.isfinite(materialized):
                continue
            payload[name] = materialized
    validate_metric_payload(payload)
    return payload


def _target_kl_exceeded(approx_kl: torch.Tensor, target_kl: float | None) -> bool:
    if target_kl is None:
        return False
    return float(approx_kl.detach().float().item()) > 1.5 * float(target_kl)


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
    extra_metric_tensors: Mapping[str, torch.Tensor] | None = None,
    omit_if_nonfinite: frozenset[str] = frozenset(),
) -> dict[str, float]:
    model.policy.set_training_mode(True)
    learning_rate = float(model.lr_schedule(progress_remaining))
    for group in model.policy.optimizer.param_groups:
        group["lr"] = learning_rate
    clip_range = float(model.clip_range(progress_remaining))
    clip_range_vf = (
        None if model.clip_range_vf is None else float(model.clip_range_vf(progress_remaining))
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
    metric_sums = torch.zeros(4, dtype=torch.float32, device=buffer.rewards.device)
    metric_count = 0
    last_epoch_kl_sum = torch.zeros((), dtype=torch.float32, device=buffer.rewards.device)
    last_epoch_kl_count = 0
    continue_training = True
    final_loss = torch.zeros((), device=buffer.rewards.device)

    for _epoch in range(int(model.n_epochs)):
        last_epoch_kl_sum.zero_()
        last_epoch_kl_count = 0
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
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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
                entropy_loss = log_prob.mean() if entropy is None else -entropy.mean()
                final_loss = (
                    policy_loss + float(ent_coef) * entropy_loss + float(model.vf_coef) * value_loss
                )

            with torch.no_grad():
                log_ratio = log_prob - old_log_probs
                approx_kl = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > clip_range).float().mean()
                last_epoch_kl_sum.add_(approx_kl.detach().float())
                last_epoch_kl_count += 1
                metric_sums.add_(
                    torch.stack(
                        (
                            policy_loss.detach().float(),
                            value_loss.detach().float(),
                            entropy_loss.detach().float(),
                            clip_fraction,
                        )
                    )
                )
                metric_count += 1
            if model.target_kl is not None and _target_kl_exceeded(
                approx_kl,
                model.target_kl,
            ):
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
    explained_variance = torch.where(
        returns_variance == 0.0,
        torch.full_like(returns_variance, float("nan")),
        1.0 - torch.var(flat_returns - flat_values, correction=0) / returns_variance,
    )
    denominator = max(metric_count, 1)
    metric_means = metric_sums / denominator
    payload: dict[str, float | torch.Tensor] = dict(extra_metric_tensors or {})
    payload.update(
        {
            TRAIN_PPO_APPROX_KL: last_epoch_kl_sum / max(last_epoch_kl_count, 1),
            TRAIN_PPO_CLIP_FRACTION: metric_means[3],
            TRAIN_PPO_VALUE_LOSS: metric_means[1],
            TRAIN_PPO_LEARNING_RATE: learning_rate,
            TRAIN_PPO_POLICY_ENTROPY: -metric_means[2],
            train_algorithm_metric("ppo", "update/policy_gradient_loss"): metric_means[0],
            TRAIN_PPO_EXPLAINED_VARIANCE: explained_variance,
        }
    )
    optional_metrics = set(omit_if_nonfinite)
    optional_metrics.add(TRAIN_PPO_EXPLAINED_VARIANCE)
    if hasattr(model.policy, "log_std"):
        payload[train_algorithm_metric("ppo", "policy/distribution/std")] = (
            torch.exp(model.policy.log_std).mean().detach()
        )
    return _materialize_metrics(
        payload,
        omit_if_nonfinite=frozenset(optional_metrics),
    )


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
    from gradlab.env_registry import GRADOOM_PROVIDER
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
    device_resident = config.env_provider == GRADOOM_PROVIDER.provider_id
    if device_resident:
        from gradlab.gradoom_device_runtime import make_gradoom_device_vec_env

        env = make_gradoom_device_vec_env(
            config,
            n_envs=n_envs,
            seed=int(common_config["seed"]),
            rom_binding=getattr(context, "rom_binding", None),
            state_archive=common_config.get("state_archive"),
        )
    else:
        env = make_training_vec_env(
            config=config,
            n_envs=n_envs,
            seed=int(common_config["seed"]),
            episode_progress_fields=tuple(common_config.get("episode_progress_fields", ())),
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
        if device_resident and device.type != "cuda":
            raise ValueError("GraDOOM device training requires backend device='cuda'")
        if device_resident and runtime.device != device:
            raise ValueError(
                f"GraDOOM simulator device {runtime.device} differs from learner device {device}"
            )
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
        from gradlab.action_contract import runtime_action_contract
        from gradlab.policy_runtime import bind_policy_action_space

        bind_policy_action_space(
            model,
            env.action_space,
            runtime_action_contract(env),
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
            store_final_observations=device_resident,
        )
        buffer = TensorRolloutBuffer.allocate(
            observations,
            action_space=env.action_space,
            n_steps=int(backend_config["n_steps"]),
            n_envs=n_envs,
            device=device,
            store_final_observations=device_resident,
        )
        calls = _CompiledPolicyCalls(
            model.policy,
            device,
            compile_policy=execution_profile.compile_policy,
        )
        precision = _Precision(str(backend_config["precision"]), device)
        reward_stats = RewardStatsAccumulator(
            active_components=active_reward_components(config.task),
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
        context.mark_ready()

        while int(model.num_timesteps) < budget.execution_total:
            if context.stop_flag.requested:
                graceful_stop.acknowledge_safe_boundary(num_timesteps=int(model.num_timesteps))
                break
            buffer.reset()
            curriculum.begin()
            model.policy.set_training_mode(False)
            throughput.begin(int(model.num_timesteps))
            for _step in range(int(backend_config["n_steps"])):
                obs_tensor = buffer.begin_step(observations, episode_starts)
                with torch.no_grad(), precision.autocast():
                    actions, values, log_probs = calls.forward(obs_tensor)
                environment_actions = (
                    actions
                    if device_resident
                    else _environment_actions(
                        actions,
                        env.action_space,
                        model.policy,
                    )
                )
                batch_step = runtime.step(environment_actions)
                truncated_lanes: np.ndarray | None = None
                terminal_values: torch.Tensor | None = None
                if not device_resident:
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
                            terminal_values = (
                                calls.predict_values(terminal_tensor).flatten().float()
                            )
                reward_slot = buffer.end_step(
                    actions,
                    batch_step.rewards,
                    values,
                    log_probs,
                    final_observations=(batch_step.final_observations if device_resident else None),
                    truncated=(batch_step.truncated if device_resident else None),
                )
                if truncated_lanes is not None and truncated_lanes.size:
                    assert terminal_values is not None
                    lane_tensor = torch.as_tensor(
                        truncated_lanes,
                        dtype=torch.int64,
                        device=device,
                    )
                    reward_slot[lane_tensor] += float(model.gamma) * terminal_values
                curriculum.capture(batch_step)
                observations = batch_step.observations
                if device_resident:
                    torch.logical_or(
                        batch_step.terminated,
                        batch_step.truncated,
                        out=dones,
                    )
                else:
                    done_array = np.logical_or(batch_step.terminated, batch_step.truncated)
                    dones.copy_(torch.as_tensor(done_array))
                episode_starts = dones
                model.num_timesteps += n_envs
                calls_since_start += 1
                records = [] if device_resident else runtime.drain_records()
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

            if device_resident:
                episodes = runtime.drain_records()
                context.session.advance(int(model.num_timesteps), episodes)
                context.session.observe_episode_completions(
                    step=int(model.num_timesteps),
                    records=episodes,
                )

            next_observations = _observation_tensor(observations, device)
            with torch.no_grad(), precision.autocast():
                last_values = calls.predict_values(next_observations)
            if device_resident:
                _bootstrap_device_time_limits(
                    buffer,
                    calls=calls,
                    precision=precision,
                    gamma=float(model.gamma),
                )
            buffer.finish(
                last_values=last_values,
                dones=dones,
                gamma=float(model.gamma),
                gae_lambda=float(model.gae_lambda),
            )
            throughput.end(int(model.num_timesteps))
            curriculum_metrics = curriculum.complete(buffer.advantages)
            rollout_metric_tensors, optional_rollout_metrics = _rollout_diagnostics(
                buffer,
                env.action_space,
            )
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
                extra_metric_tensors=rollout_metric_tensors,
                omit_if_nonfinite=optional_rollout_metrics,
            )
            rollout_metrics = dict(curriculum_metrics)
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
            if context.stop_flag.requested:
                graceful_stop.acknowledge_safe_boundary(num_timesteps=int(model.num_timesteps))
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
