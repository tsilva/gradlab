from __future__ import annotations

import math
import time
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter

from gradlab.action_codecs import LegalTupleMultiDiscrete
from gradlab.action_contract import runtime_action_contract
from gradlab.artifacts import install_model_bundle
from gradlab.early_stop import (
    MetricEarlyStopStateMachine,
    MetricSample,
)
from gradlab.env import EnvConfig
from gradlab.file_utils import atomic_write_json
from gradlab.metric_names import (
    canonical_training_scalars,
    TRAIN_ARTIFACT_SAVE_SECONDS,
    TRAIN_ARCHIVE_ADMISSION_ACCEPTED_COUNT,
    TRAIN_ARCHIVE_ADMISSION_CANDIDATE_COUNT,
    TRAIN_ARCHIVE_CAPTURE_CALL_COUNT,
    TRAIN_ARCHIVE_CAPTURE_SECONDS,
    TRAIN_ARCHIVE_CURRICULUM_CELL_COUNT,
    TRAIN_ARCHIVE_CURRICULUM_ENTRY_COUNT,
    TRAIN_ARCHIVE_EVICTED_COUNT,
    TRAIN_ARCHIVE_FEEDBACK_TRAJECTORY_COUNT,
    TRAIN_ARCHIVE_RESTORE_EPISODE_COUNT,
    TRAIN_ARCHIVE_RESTORE_FORCED_BOUNDARY_COUNT,
    TRAIN_ARCHIVE_RESTORE_SECONDS,
    TRAIN_ARCHIVE_SAMPLING_EFFECTIVE_CELL_COUNT,
    TRAIN_ARCHIVE_SAMPLING_PROBABILITY_MAX,
    TRAIN_ARCHIVE_TRANSITION_SHARE,
    TRAIN_REWARD_ROOT,
    stat_metric,
    train_algorithm_metric,
    train_early_stop_metric,
    train_reward_component_metric,
    validate_metric_name,
)
from gradlab.policy_execution import compile_policy_execution_contract
from gradlab.metric_store import MetricStore
from gradlab.state_archive import state_archive_artifact_summary
from gradlab.train_config import wandb_publication_enabled
from gradlab.training_lifecycle import LoggerMetricFrameSink
from gradlab.training_metrics import (
    EpisodeMetricsReducer,
    throughput_delta_metrics,
)


def task_metric_source(start_id: Any) -> Any:
    """Keep the same readable start identifier in training and evaluation."""
    return start_id


def policy_entropy_bounds(action_space: Any) -> tuple[float, float] | None:
    """Return finite theoretical entropy bounds for an SB3 discrete policy."""
    if isinstance(action_space, gym.spaces.Discrete):
        upper = math.log(int(action_space.n))
    elif isinstance(action_space, LegalTupleMultiDiscrete):
        upper = math.log(action_space.legal_tuple_count)
    elif isinstance(action_space, gym.spaces.MultiDiscrete):
        upper = math.fsum(
            math.log(int(cardinality)) for cardinality in np.asarray(action_space.nvec).reshape(-1)
        )
    elif isinstance(action_space, gym.spaces.MultiBinary):
        upper = math.prod(int(size) for size in action_space.shape) * math.log(2)
    else:
        return None
    return 0.0, float(upper)


def policy_discrete_action_indices(actions: Any, action_space: Any) -> np.ndarray:
    """Map rollout actions to scalar categories used by collapse diagnostics."""

    if actions is None:
        return np.array([], dtype=np.int64)
    if isinstance(action_space, LegalTupleMultiDiscrete):
        return action_space.legal_tuple_indices(actions).reshape(-1)
    if not isinstance(getattr(action_space, "n", None), Integral):
        return np.array([], dtype=np.int64)
    values = np.asarray(actions)
    if values.size == 0 or (values.ndim > 1 and values.shape[-1] != 1):
        return np.array([], dtype=np.int64)
    flattened = values.reshape(-1)
    if not np.issubdtype(flattened.dtype, np.number):
        return np.array([], dtype=np.int64)
    finite = flattened[np.isfinite(flattened)]
    integers = finite.astype(np.int64)
    if not np.allclose(finite, integers):
        return np.array([], dtype=np.int64)
    return integers


class CallbackHelper:
    """Plain lifecycle component driven by the single SB3 callback."""

    def __init__(self) -> None:
        self.model: Any = None
        self.locals: dict[str, Any] = {}
        self.globals: dict[str, Any] = {}
        self.num_timesteps = 0
        self.n_calls = 0

    @property
    def logger(self) -> Any:
        return self.model.logger

    def bind(self, callback: BaseCallback) -> None:
        self.model = callback.model
        self.locals = callback.locals
        self.globals = callback.globals
        self.num_timesteps = callback.num_timesteps
        self.n_calls = callback.n_calls


class LedgerCheckpointHelper(CallbackHelper):
    def __init__(
        self,
        *,
        train_config: Mapping[str, Any],
        config: EnvConfig,
        save_freq: int,
        save_path: str | Path,
        name_prefix: str,
        metric_store_path: Path | str,
        eval_required: bool = True,
        checkpoint_coordinator: Any | None = None,
    ) -> None:
        super().__init__()
        self.train_config = train_config
        self.config = config
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix
        self.metric_store = MetricStore(metric_store_path)
        self.eval_required = bool(eval_required)
        self.checkpoint_coordinator = checkpoint_coordinator

    def _init_callback(self) -> None:
        if self.checkpoint_coordinator is None or self.checkpoint_coordinator.persist_intermediate:
            self.save_path.mkdir(parents=True, exist_ok=True)
        self.metric_store.init()

    def _on_step(self) -> bool:
        if self.save_freq <= 0 or self.n_calls % self.save_freq != 0:
            return True
        training_cap = self.train_config.get("timesteps")
        if training_cap is not None and self.num_timesteps >= int(training_cap):
            # The learner writes the authoritative final checkpoint immediately
            # after learn() returns. Avoid an immutable periodic/final collision at
            # an exactly aligned training cap.
            return True
        self.save_checkpoint(self.num_timesteps, kind="checkpoint")
        return True

    def save_checkpoint(self, step: int, *, kind: str) -> Path | None:
        final_path = self.save_path / f"{self.name_prefix}_{step}_steps.zip"
        if self.checkpoint_coordinator is not None:
            return self.checkpoint_coordinator.save(
                kind=kind,
                step=step,
                model_path=final_path,
                save_bundle=lambda path, artifact_kind, artifact_step: install_model_bundle(
                    path,
                    save_checkpoint=lambda destination: self.model.save(str(destination)),
                    train_config=self.train_config,
                    config=self.config,
                    kind=artifact_kind,
                    checkpoint_step_value=artifact_step,
                    state_archive_summary=state_archive_artifact_summary(
                        getattr(self.model, "env", None)
                    ),
                    action_contract=runtime_action_contract(getattr(self.model, "env", None)),
                    policy_execution_contract=compile_policy_execution_contract(
                        self.model,
                        getattr(self.model, "env", None),
                    ),
                ),
            )
        started = time.perf_counter()
        final_path = install_model_bundle(
            final_path,
            save_checkpoint=lambda path: self.model.save(str(path)),
            train_config=self.train_config,
            config=self.config,
            kind=kind,
            checkpoint_step_value=step,
            state_archive_summary=state_archive_artifact_summary(getattr(self.model, "env", None)),
            action_contract=runtime_action_contract(getattr(self.model, "env", None)),
            policy_execution_contract=compile_policy_execution_contract(
                self.model,
                getattr(self.model, "env", None),
            ),
        )
        checkpoint_id = self.metric_store.record_checkpoint(
            run_name=str(self.train_config.get("run_name") or ""),
            kind=kind,
            step=step,
            path=final_path,
            sha256=None,
            eval_required=self.eval_required,
        )
        self.metric_store.append_metrics(
            {TRAIN_ARTIFACT_SAVE_SECONDS: time.perf_counter() - started},
            step=step,
            source=f"checkpoint-save:{kind}",
            publish=wandb_publication_enabled(self.train_config),
        )
        print(
            f"checkpoint ready: id={checkpoint_id} step={step} path={final_path}",
            flush=True,
        )
        return final_path


@dataclass(frozen=True)
class _CompletedRollout:
    step: int
    steps: int
    start_time: float
    end_time: float
    rollout_seconds: float
    env_step_seconds: float | None


class ThroughputHelper(CallbackHelper):
    """Publish one temporally aligned frame for each completed training iteration."""

    def __init__(
        self,
        clock: Callable[[], float] | None = None,
        *,
        metric_store_path: Path | str | None = None,
        wandb_enabled: bool = True,
    ):
        super().__init__()
        self.clock = clock or time.perf_counter
        self.metric_store = MetricStore(metric_store_path) if metric_store_path else None
        if self.metric_store is not None:
            self.metric_store.init()
        self.wandb_enabled = bool(wandb_enabled)
        self.rollout_start_time: float | None = None
        self.rollout_start_timesteps: int | None = None
        self.completed_rollout: _CompletedRollout | None = None
        self.native_step_stats_start: Mapping[str, float | int] | None = None

    @staticmethod
    def _native_step_stats_source(env: Any) -> Any | None:
        seen: set[int] = set()
        current = env
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            native_step_stats = getattr(current, "native_step_stats", None)
            if callable(native_step_stats):
                return current
            current = getattr(current, "venv", None) or getattr(current, "env", None)
        return None

    @classmethod
    def _native_step_stats(cls, env: Any) -> Mapping[str, float | int] | None:
        source = cls._native_step_stats_source(env)
        if source is None:
            return None
        stats = source.native_step_stats()
        return stats if isinstance(stats, Mapping) else None

    def _on_rollout_start(self) -> None:
        now = self.clock()
        if self.completed_rollout is not None:
            self._publish_completed_iteration(self.completed_rollout, next_start_time=now)
            self.completed_rollout = None

        self.rollout_start_time = now
        self.rollout_start_timesteps = self.num_timesteps
        self.native_step_stats_start = self._native_step_stats(getattr(self.model, "env", None))

    def _on_rollout_end(self) -> None:
        now = self.clock()
        if self.rollout_start_time is not None and self.rollout_start_timesteps is not None:
            elapsed = now - self.rollout_start_time
            steps = self.num_timesteps - self.rollout_start_timesteps
            if elapsed > 0 and steps > 0:
                self.completed_rollout = _CompletedRollout(
                    step=self.num_timesteps,
                    steps=steps,
                    start_time=self.rollout_start_time,
                    end_time=now,
                    rollout_seconds=elapsed,
                    env_step_seconds=self._native_step_seconds(),
                )

    def _on_training_end(self) -> None:
        if self.completed_rollout is not None:
            self._publish_completed_iteration(
                self.completed_rollout,
                next_start_time=self.clock(),
            )
            self.completed_rollout = None

    def _native_step_seconds(self) -> float | None:
        start = self.native_step_stats_start
        end = self._native_step_stats(getattr(self.model, "env", None))
        self.native_step_stats_start = None
        if start is None or end is None:
            return None
        native_seconds = float(end.get("seconds_total", 0.0)) - float(
            start.get("seconds_total", 0.0)
        )
        native_calls = int(end.get("calls_total", 0)) - int(start.get("calls_total", 0))
        if native_seconds <= 0 or native_calls <= 0:
            return None
        return native_seconds

    def _publish_completed_iteration(
        self,
        rollout: _CompletedRollout,
        *,
        next_start_time: float,
    ) -> None:
        between_seconds = next_start_time - rollout.end_time
        loop_seconds = next_start_time - rollout.start_time
        if between_seconds < 0 or loop_seconds <= 0:
            return
        payload = throughput_delta_metrics(
            steps=rollout.steps,
            loop_seconds=loop_seconds,
            provider_step_seconds=rollout.env_step_seconds,
            rollout_seconds=rollout.rollout_seconds,
            between_rollouts_seconds=between_seconds,
        )
        if self.metric_store is not None:
            self.metric_store.append_metrics(
                payload,
                step=rollout.step,
                source="train",
                publish=self.wandb_enabled,
            )
            return
        for name, value in payload.items():
            self.logger.record(name, value)


class MetricEarlyStopHelper(CallbackHelper):
    def __init__(
        self,
        *,
        decision_path: Path,
        config: Any,
        stop_flag: Any,
        metric_store_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.machine = MetricEarlyStopStateMachine(config, label="early_stop")
        self.watched_metrics = tuple(
            sorted({str(condition["metric"]) for condition in self.machine.conditions.values()})
        )
        self.decision_path = decision_path
        self.stop_flag = stop_flag
        self.triggered = False
        self.stop_requested = False
        self.metric_store = (
            MetricStore(metric_store_path, timeout=0.05) if metric_store_path else None
        )

    def _on_rollout_end(self) -> None:
        if self.stop_requested:
            return
        self.evaluate_now()

    def evaluate_now(self) -> bool:
        samples = {
            metric: sample
            for metric in self.watched_metrics
            if (sample := self.current_metric_sample(metric)) is not None
        }
        update = self.machine.update(samples)
        for condition_id, observation in update.observations.items():
            values = {
                train_early_stop_metric(
                    condition_id,
                    "patience/progress",
                ): observation.patience_progress,
            }
            if observation.target_progress is not None:
                values[train_early_stop_metric(condition_id, "target/progress")] = (
                    observation.target_progress
                )
            for name, value in values.items():
                self.logger.record(name, value)
        if update.stop_decision is None:
            return True
        self.triggered = True
        self.stop_requested = True
        atomic_write_json(self.decision_path, update.stop_decision)
        self.stop_flag.request(f"early_stop:{str(update.stop_decision['condition_id'])}")
        print(
            "early stop: "
            f"condition={update.stop_decision['condition_id']} "
            f"outcome={update.stop_decision['outcome']} "
            f"metric={update.stop_decision['metric']} "
            f"value={float(update.stop_decision['value']):.12g} "
            f"step={int(update.stop_decision['metric_step'])}",
            flush=True,
        )
        return False

    def current_metric_sample(self, metric_name: str) -> MetricSample | None:
        logger = getattr(self.model, "logger", None)
        for attr in ("name_to_value", "records"):
            values = getattr(logger, attr, None)
            if not isinstance(values, Mapping) or metric_name not in values:
                continue
            try:
                value = float(values[metric_name])
            except TypeError, ValueError:
                return None
            return (
                MetricSample(value=value, step=int(self.num_timesteps))
                if math.isfinite(value)
                else None
            )
        if self.metric_store is not None:
            try:
                sample = self.metric_store.latest_metric_sample(metric_name)
            except Exception as exc:
                print(
                    f"warning: metric store lookup failed for early stop metric "
                    f"{metric_name}: {exc}",
                    flush=True,
                )
                return None
            if sample is not None:
                value, step = sample
                return MetricSample(value=value, step=step)
        return None


class MetricStoreOutputFormat(KVWriter):
    """Persist the complete scalar payload received by SB3's logger dump."""

    def __init__(
        self,
        metric_store_path: Path | str,
        *,
        algorithm_id: str = "ppo",
        source: str = "train",
        wandb_enabled: bool = True,
        clock: Callable[[], float] | None = None,
    ) -> None:
        # Training metrics are durable evidence. Let SQLite's bounded busy
        # handler absorb brief publisher/coordinator transactions instead of
        # terminating a multi-hour run after 50 ms of contention.
        self.metric_store = MetricStore(metric_store_path)
        self.algorithm_id = algorithm_id
        self.source = source
        self.wandb_enabled = bool(wandb_enabled)
        self.clock = clock or time.perf_counter
        self.last_warning: float | None = None
        self.metric_store.init()

    def write(
        self,
        key_values: dict[str, Any],
        key_excluded: dict[str, tuple[str, ...]],
        step: int = 0,
    ) -> None:
        del key_excluded
        payload = {
            key: value
            for key, value in canonical_training_scalars(
                key_values,
                algorithm_id=self.algorithm_id,
            ).items()
            if math.isfinite(value)
        }
        if not payload:
            return
        try:
            self.metric_store.append_metrics(
                payload,
                step=step,
                source=self.source,
                publish=self.wandb_enabled,
            )
        except Exception as exc:
            now = self.clock()
            if self.last_warning is None or now - self.last_warning >= 60:
                print(f"metric store write failed: {exc}", flush=True)
                self.last_warning = now
            raise RuntimeError("durable metric frame write failed") from exc

    def close(self) -> None:
        """MetricStore owns no persistent connection."""


class MetricStoreLoggerHelper(CallbackHelper):
    """Install the durable metric writer at SB3's authoritative dump boundary."""

    def __init__(
        self,
        metric_store_path: Path | str,
        *,
        algorithm_id: str = "ppo",
        source: str = "train",
        wandb_enabled: bool = True,
        clock: Callable[[], float] | None = None,
    ) -> None:
        super().__init__()
        self.output_format = MetricStoreOutputFormat(
            metric_store_path,
            algorithm_id=algorithm_id,
            source=source,
            wandb_enabled=wandb_enabled,
            clock=clock,
        )

    def _on_training_start(self) -> None:
        output_formats = self.logger.output_formats
        if self.output_format not in output_formats:
            output_formats.append(self.output_format)

    def _on_training_end(self) -> None:
        pending = getattr(self.logger, "name_to_value", None)
        if isinstance(pending, Mapping) and pending:
            self.logger.dump(step=self.num_timesteps)


class RolloutDiagnosticsHelper(CallbackHelper):
    """Log compact PPO rollout and discrete-policy collapse diagnostics."""

    def __init__(
        self,
        algorithm_id: str = "ppo",
    ):
        super().__init__()
        self.algorithm_id = algorithm_id

    def _on_rollout_end(self) -> None:
        rollout_buffer = getattr(self.model, "rollout_buffer", None)
        if rollout_buffer is None:
            return

        value_predictions = self._finite_values(getattr(rollout_buffer, "values", None))
        advantages = self._finite_values(getattr(rollout_buffer, "advantages", None))
        discrete_actions = self._discrete_actions(
            getattr(rollout_buffer, "actions", None),
            getattr(self.model, "action_space", None),
        )
        self._record_stats(
            train_algorithm_metric(self.algorithm_id, "rollout/value/prediction"),
            value_predictions,
        )
        self._record_stats(
            train_algorithm_metric(self.algorithm_id, "rollout/advantage"),
            advantages,
        )
        if discrete_actions.size > 0:
            _actions, counts = np.unique(discrete_actions, return_counts=True)
            self.logger.record(
                train_algorithm_metric(self.algorithm_id, "policy/dominant/action/rate"),
                float(np.max(counts) / discrete_actions.size),
            )

    @staticmethod
    def _finite_values(values: Any) -> np.ndarray:
        if values is None:
            return np.array([], dtype=np.float64)
        flattened = np.asarray(values, dtype=np.float64).reshape(-1)
        return flattened[np.isfinite(flattened)]

    @staticmethod
    def _discrete_actions(actions: Any, action_space: Any) -> np.ndarray:
        return policy_discrete_action_indices(actions, action_space)

    def _record_stats(self, prefix: str, values: np.ndarray) -> None:
        if values.size == 0:
            return
        self.logger.record(stat_metric(prefix, "mean"), float(np.mean(values)))
        self.logger.record(stat_metric(prefix, "std"), float(np.std(values)))


ARCHIVE_CURRICULUM_METRIC_MAP = {
    "archive_cell_count": TRAIN_ARCHIVE_CURRICULUM_CELL_COUNT,
    "archive_entry_count": TRAIN_ARCHIVE_CURRICULUM_ENTRY_COUNT,
    "admission_candidate_count": TRAIN_ARCHIVE_ADMISSION_CANDIDATE_COUNT,
    "admission_accepted_count": TRAIN_ARCHIVE_ADMISSION_ACCEPTED_COUNT,
    "evicted_count": TRAIN_ARCHIVE_EVICTED_COUNT,
    "capture_call_count": TRAIN_ARCHIVE_CAPTURE_CALL_COUNT,
    "archive_reset_count": TRAIN_ARCHIVE_RESTORE_EPISODE_COUNT,
    "forced_boundary_count": TRAIN_ARCHIVE_RESTORE_FORCED_BOUNDARY_COUNT,
    "feedback_trajectory_count": TRAIN_ARCHIVE_FEEDBACK_TRAJECTORY_COUNT,
    "transition_share": TRAIN_ARCHIVE_TRANSITION_SHARE,
    "sampling_probability_max": TRAIN_ARCHIVE_SAMPLING_PROBABILITY_MAX,
    "sampling_effective_cell_count": TRAIN_ARCHIVE_SAMPLING_EFFECTIVE_CELL_COUNT,
    "capture_seconds": TRAIN_ARCHIVE_CAPTURE_SECONDS,
    "reset_seconds": TRAIN_ARCHIVE_RESTORE_SECONDS,
}


class ArchiveCurriculumFeedbackHelper(CallbackHelper):
    """Attribute raw rollout GAE to true archive-origin episodes."""

    _METRIC_MAP = ARCHIVE_CURRICULUM_METRIC_MAP

    def __init__(self) -> None:
        super().__init__()
        self._source: Any | None = None
        self._steps: list[Any] = []
        self._fragments: dict[tuple[int, int, int, str], list[float | int]] = {}

    def _find_source(self) -> Any:
        if self._source is not None:
            return self._source
        current = getattr(self.model, "env", None)
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if callable(getattr(current, "take_curriculum_step", None)):
                self._source = current
                return current
            current = getattr(current, "venv", None) or getattr(current, "env", None)
        raise RuntimeError("state archive requires GradLabVecEnv curriculum hooks")

    def _on_rollout_start(self) -> None:
        self._steps = []
        self._find_source().curriculum_begin_rollout()

    def _on_step(self) -> bool:
        step = self._find_source().take_curriculum_step()
        if step is None:
            raise RuntimeError("state archive step attribution is missing")
        self._steps.append(step)
        return True

    def _on_rollout_end(self) -> None:
        rollout_buffer = getattr(self.model, "rollout_buffer", None)
        if rollout_buffer is None:
            raise RuntimeError("state archive requires an on-policy rollout buffer")
        advantages = np.asarray(rollout_buffer.advantages, dtype=np.float64)
        if advantages.ndim != 2 or advantages.shape[0] != len(self._steps):
            raise RuntimeError("state archive attribution does not align with the rollout buffer")
        source = self._find_source()
        completed: list[tuple[tuple[int, int, int, str], float]] = []
        for step_index, step in enumerate(self._steps):
            cell_ids = np.asarray(step.curriculum_cell_ids, dtype=object)
            generations = np.asarray(step.curriculum_generations, dtype=np.int64)
            episode_indices = np.asarray(step.curriculum_episode_indices, dtype=np.int64)
            feedback_dones = np.asarray(step.curriculum_feedback_dones, dtype=np.bool_)
            if cell_ids.shape != (advantages.shape[1],):
                raise RuntimeError("state archive sidecar lane shape changed")
            for lane in range(advantages.shape[1]):
                cell_id = cell_ids[lane]
                if cell_id is None:
                    continue
                key = (
                    int(generations[lane]),
                    lane,
                    int(episode_indices[lane]),
                    str(cell_id),
                )
                fragment = self._fragments.setdefault(key, [0.0, 0])
                value = float(abs(advantages[step_index, lane]))
                if not math.isfinite(value):
                    raise RuntimeError("state archive received non-finite raw GAE")
                fragment[0] = float(fragment[0]) + value
                fragment[1] = int(fragment[1]) + 1
                if bool(feedback_dones[lane]):
                    total, count = self._fragments.pop(key)
                    completed.append((key, float(total) / int(count)))
        for key, value_error in sorted(completed, key=lambda item: item[0]):
            source.submit_curriculum_feedback(key[3], value_error)
        metrics = source.curriculum_complete_rollout()
        for internal_name, metric_name in self._METRIC_MAP.items():
            self.logger.record(metric_name, float(metrics.get(internal_name, 0.0)))
        self._steps = []

    def _on_training_end(self) -> None:
        self._steps = []
        self._fragments.clear()


class _BufferedStats:
    """Reusable contiguous storage for one rollout's vector batches."""

    __slots__ = ("buffer", "size")

    def __init__(self) -> None:
        self.buffer = np.empty(0, dtype=np.float64)
        self.size = 0

    def reset(self) -> None:
        self.size = 0

    def update(self, value: Any, *, reserve: int) -> None:
        values = np.asarray(value).reshape(-1)
        if values.size == 0:
            return
        end = self.size + values.size
        if end > self.buffer.size:
            capacity = max(end, reserve, max(64, self.buffer.size * 2))
            grown = np.empty(capacity, dtype=np.float64)
            grown[: self.size] = self.buffer[: self.size]
            self.buffer = grown
        self.buffer[self.size : end] = values
        self.size = end

    def flush(self) -> np.ndarray:
        values = self.buffer[: self.size]
        values = values[np.isfinite(values)]
        self.reset()
        return values


class RewardStatsAccumulator:
    component_info_keys = {
        "native": "native_reward_component",
        "cell_novelty": "cell_novelty_reward_component",
        "progress": "progress_reward_component",
        "score": "score_reward_component",
        "completion": "completion_reward_component",
        "death": "death_penalty_component",
        "time": "time_penalty_component",
        "kill": "kill_reward_component",
        "hit": "hit_reward_component",
        "damage": "damage_reward_component",
        "health": "health_reward_component",
        "armor": "armor_reward_component",
        "weapon": "weapon_reward_component",
        "ammo": "ammo_reward_component",
        "weapon_hold": "weapon_hold_reward_component",
    }

    def __init__(
        self,
        *,
        active_components: Sequence[str] = (),
    ) -> None:
        self.shaped = _BufferedStats()
        self.raw = _BufferedStats()
        self.active_components = tuple(
            component for component in active_components if component in self.component_info_keys
        )
        self.components = {component: _BufferedStats() for component in self.active_components}

    def consume(self, metrics: Mapping[str, Any], *, reserve: int) -> None:
        if (value := metrics.get("shaped_reward")) is not None:
            self.shaped.update(value, reserve=reserve)
        if (value := metrics.get("raw_reward")) is not None:
            self.raw.update(value, reserve=reserve)
        for component, accumulator in self.components.items():
            info_key = self.component_info_keys[component]
            value = metrics.get(info_key)
            if value is not None:
                accumulator.update(value, reserve=reserve)

    @staticmethod
    def _distribution(prefix: str, values: np.ndarray, stats: Sequence[str]) -> dict[str, float]:
        if values.size == 0:
            return {}
        calculations = {
            "mean": lambda: float(np.mean(values)),
            "std": lambda: float(np.std(values)),
            "nonzero_rate": lambda: float(np.mean(values != 0.0)),
        }
        return {
            (
                validate_metric_name(f"{prefix}/nonzero/rate")
                if stat == "nonzero_rate"
                else stat_metric(prefix, stat)
            ): calculations[stat]()
            for stat in stats
        }

    def flush(self) -> dict[str, float]:
        shaped = self.shaped.flush()
        raw = self.raw.flush()
        payload = self._distribution(
            f"{TRAIN_REWARD_ROOT}/shaped",
            shaped,
            ("mean", "std", "nonzero_rate"),
        )
        if raw.size > 0 and (shaped.size != raw.size or not np.array_equal(shaped, raw)):
            payload.update(self._distribution(f"{TRAIN_REWARD_ROOT}/raw", raw, ("mean", "std")))
        abs_sums: dict[str, float] = {}
        for component, accumulator in self.components.items():
            values = accumulator.flush()
            if values.size == 0:
                continue
            payload[train_reward_component_metric(component, "mean")] = float(np.mean(values))
            payload[train_reward_component_metric(component, "nonzero_rate")] = float(
                np.mean(values != 0.0)
            )
            abs_sums[component] = float(np.sum(np.abs(values)))
        total_abs_sum = sum(abs_sums.values())
        for component, abs_sum in abs_sums.items():
            payload[train_reward_component_metric(component, "share")] = (
                abs_sum / total_abs_sum if total_abs_sum > 0.0 else 0.0
            )
        return payload


class RuntimeMetricsHelper(CallbackHelper):
    """Reduce runtime records and publish one scalar payload per rollout."""

    def __init__(
        self,
        *,
        event_names: Sequence[str] = (),
        active_reward_components: Sequence[str] = (),
        configured_starts: Sequence[str] = (),
        progress_fields: Sequence[str] = (),
        track_success: bool = False,
        session: Any | None = None,
    ) -> None:
        super().__init__()
        self.session = session
        self.reward_stats = RewardStatsAccumulator(
            active_components=active_reward_components,
        )
        self.episode_metrics = EpisodeMetricsReducer(
            event_names=event_names,
            configured_starts=configured_starts,
            progress_fields=progress_fields,
            track_success=track_success,
        )
        self.pending_metrics: dict[str, int | float] = {}

    def _on_training_start(self) -> None:
        if self.session is not None:
            self.session.set_metric_sink(LoggerMetricFrameSink(self.logger))

    def _on_records(self, records: Iterable[Any]) -> bool:
        episode_records: list[Any] = []
        for record in records:
            if hasattr(record, "num_envs") and not hasattr(record, "lane"):
                num_envs = int(record.num_envs)
                rollout_steps = int(getattr(self.model, "n_steps", 1))
                self.reward_stats.consume(
                    getattr(record, "metrics", {}) or {},
                    reserve=num_envs * rollout_steps,
                )
                continue
            if hasattr(record, "episode_return"):
                episode_records.append(record)
        if self.session is not None:
            self.session.advance(self.num_timesteps, episode_records)
            self.session.observe_episode_completions(
                step=self.num_timesteps,
                records=episode_records,
            )
        else:
            self.pending_metrics.update(self.episode_metrics.consume(episode_records))
        return True

    def _on_rollout_end(self) -> None:
        reward_payload = self.reward_stats.flush()
        if self.session is not None:
            current = getattr(self.logger, "name_to_value", {})
            payload = dict(current) if isinstance(current, Mapping) else {}
            payload.update(reward_payload)
            self.session.report(step=self.num_timesteps, metrics=payload)
            return
        self.pending_metrics.update(reward_payload)
        if self.pending_metrics:
            for key, value in self.pending_metrics.items():
                self.logger.record(key, value)
        self.pending_metrics = {}

    def _on_training_end(self) -> None:
        if self.session is None:
            return
        current = getattr(self.logger, "name_to_value", {})
        payload = dict(current) if isinstance(current, Mapping) else {}
        self.session.report(step=self.num_timesteps, metrics=payload)


class GradLabCallback(BaseCallback):
    """The sole SB3 callback; delegates lifecycle work to plain components."""

    def __init__(self, components: Sequence[CallbackHelper]) -> None:
        super().__init__()
        self.components = tuple(components)
        self._record_source: Any | None = None
        self._record_source_searched = False
        hook_names = (
            "_init_callback",
            "_on_training_start",
            "_on_rollout_start",
            "_on_rollout_end",
            "_on_training_end",
        )
        self._hooks = {
            hook: tuple(
                (component, method)
                for component in self.components
                if callable(method := getattr(component, hook, None))
            )
            for hook in hook_names
        }
        self._step_operations = tuple(
            (
                component,
                record_hook if callable(record_hook) else getattr(component, "_on_step", None),
                callable(record_hook),
            )
            for component in self.components
            if callable(record_hook := getattr(component, "_on_records", None))
            or callable(getattr(component, "_on_step", None))
        )

    def _bind(self, component: CallbackHelper) -> None:
        component.bind(self)

    def _call(self, hook: str) -> bool:
        for component, method in self._hooks[hook]:
            self._bind(component)
            if method() is False:
                return False
        return True

    def _call_step(self) -> bool:
        records: Iterable[Any] | None = None
        for component, method, consumes_records in self._step_operations:
            self._bind(component)
            if consumes_records:
                if records is None:
                    source = self._find_record_source()
                    if source is None:
                        raise RuntimeError("GradLabCallback requires GradLabVecEnv.drain_records()")
                    records = tuple(source.drain_records())
                result = method(records)
            else:
                result = method()
            if result is False:
                return False
        return True

    def _find_record_source(self) -> Any | None:
        if self._record_source_searched:
            return self._record_source
        self._record_source_searched = True
        seen: set[int] = set()
        current = getattr(self.model, "env", None)
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if callable(getattr(current, "drain_records", None)):
                self._record_source = current
                break
            current = getattr(current, "venv", None) or getattr(current, "env", None)
        return self._record_source

    def _init_callback(self) -> None:
        self._call("_init_callback")

    def _on_training_start(self) -> None:
        self._call("_on_training_start")

    def _on_rollout_start(self) -> None:
        self._call("_on_rollout_start")

    def _on_step(self) -> bool:
        return self._call_step()

    def _on_rollout_end(self) -> None:
        self._call("_on_rollout_end")

    def _on_training_end(self) -> None:
        self._call("_on_training_end")
