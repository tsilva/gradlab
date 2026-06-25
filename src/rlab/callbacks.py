from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from rlab.artifacts import checkpoint_step, log_wandb_model_artifact
from rlab.env import EnvConfig
from rlab.metric_names import (
    GLOBAL_STEP,
    ROLLOUT_ADVANTAGE,
    ROLLOUT_ADVANTAGE_HIST,
    ROLLOUT_VALUE_PRED,
    ROLLOUT_VALUE_PRED_HIST,
    THROUGHPUT_LOOP_FPS,
    THROUGHPUT_ROLLOUT_FPS,
    TRAIN_DONE_ALL,
    TRAIN_DONE_MAX_STEPS,
    TRAIN_DONE_UNCLASSIFIED,
    TRAIN_REWARD_COMPONENT_ROOT,
    TRAIN_REWARD_SHARE_ROOT,
    stat_metric,
    train_done_value_metric,
    train_done_reason_metric,
)


class WandbCheckpointArtifactCallback(BaseCallback):
    def __init__(
        self,
        wandb_run,
        args: argparse.Namespace,
        config: EnvConfig,
        checkpoint_dir: str,
        scan_freq: int,
    ):
        super().__init__()
        self.wandb_run = wandb_run
        self.args = args
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.scan_freq = scan_freq
        self.logged_paths: set[Path] = set()

    def _on_step(self) -> bool:
        if self.scan_freq <= 1 or self.n_calls % self.scan_freq == 0:
            self.log_new_checkpoints()
        return True

    def log_new_checkpoints(self) -> None:
        for checkpoint_path in sorted(self.checkpoint_dir.glob("*.zip")):
            resolved_path = checkpoint_path.resolve()
            if resolved_path in self.logged_paths:
                continue
            step = checkpoint_step(checkpoint_path)
            aliases = ["latest"]
            if step is not None:
                aliases.append(f"step-{step}")
            log_wandb_model_artifact(
                self.wandb_run,
                self.args,
                self.config,
                checkpoint_path,
                kind="checkpoint",
                aliases=aliases,
            )
            self.logged_paths.add(resolved_path)


class ThroughputCallback(BaseCallback):
    """Log rollout-only and full-loop instantaneous throughput."""

    def __init__(self, clock: Callable[[], float] | None = None):
        super().__init__()
        self.clock = clock or time.perf_counter
        self.rollout_start_time: float | None = None
        self.rollout_start_timesteps: int | None = None
        self.previous_rollout_start_time: float | None = None
        self.previous_rollout_start_timesteps: int | None = None
        self.pending_fps_instant: float | None = None

    def _on_rollout_start(self) -> None:
        now = self.clock()
        if (
            self.previous_rollout_start_time is not None
            and self.previous_rollout_start_timesteps is not None
        ):
            elapsed = now - self.previous_rollout_start_time
            steps = self.num_timesteps - self.previous_rollout_start_timesteps
            if elapsed > 0 and steps > 0:
                self.pending_fps_instant = steps / elapsed

        self.rollout_start_time = now
        self.rollout_start_timesteps = self.num_timesteps
        self.previous_rollout_start_time = now
        self.previous_rollout_start_timesteps = self.num_timesteps

    def _on_rollout_end(self) -> None:
        now = self.clock()
        if self.rollout_start_time is not None and self.rollout_start_timesteps is not None:
            elapsed = now - self.rollout_start_time
            steps = self.num_timesteps - self.rollout_start_timesteps
            if elapsed > 0 and steps > 0:
                self.logger.record(THROUGHPUT_ROLLOUT_FPS, steps / elapsed)

        if self.pending_fps_instant is not None:
            self.logger.record(THROUGHPUT_LOOP_FPS, self.pending_fps_instant)
            self.pending_fps_instant = None

    def _on_step(self) -> bool:
        return True


class RolloutDiagnosticsCallback(BaseCallback):
    """Log rollout-buffer value and advantage distributions."""

    def __init__(self, wandb_run=None, log_histograms: bool = True):
        super().__init__()
        self.wandb_run = wandb_run
        self.log_histograms = log_histograms

    def _on_rollout_end(self) -> None:
        rollout_buffer = getattr(self.model, "rollout_buffer", None)
        if rollout_buffer is None:
            return

        value_predictions = self._finite_values(getattr(rollout_buffer, "values", None))
        advantages = self._finite_values(getattr(rollout_buffer, "advantages", None))
        self._record_stats(ROLLOUT_VALUE_PRED, value_predictions)
        self._record_stats(ROLLOUT_ADVANTAGE, advantages)
        self._log_wandb_histograms(value_predictions, advantages)

    def _on_step(self) -> bool:
        return True

    @staticmethod
    def _finite_values(values: Any) -> np.ndarray:
        if values is None:
            return np.array([], dtype=np.float64)
        flattened = np.asarray(values, dtype=np.float64).reshape(-1)
        return flattened[np.isfinite(flattened)]

    def _record_stats(self, prefix: str, values: np.ndarray) -> None:
        if values.size == 0:
            return
        self.logger.record(stat_metric(prefix, "mean"), float(np.mean(values)))
        self.logger.record(stat_metric(prefix, "std"), float(np.std(values)))
        self.logger.record(stat_metric(prefix, "min"), float(np.min(values)))
        self.logger.record(stat_metric(prefix, "max"), float(np.max(values)))
        self.logger.record(stat_metric(prefix, "abs_mean"), float(np.mean(np.abs(values))))

    def _log_wandb_histograms(self, value_predictions: np.ndarray, advantages: np.ndarray) -> None:
        if self.wandb_run is None or not self.log_histograms:
            return

        import wandb

        payload: dict[str, object] = {GLOBAL_STEP: self.num_timesteps}
        if value_predictions.size > 0:
            payload[ROLLOUT_VALUE_PRED_HIST] = wandb.Histogram(value_predictions)
        if advantages.size > 0:
            payload[ROLLOUT_ADVANTAGE_HIST] = wandb.Histogram(advantages)
        if len(payload) > 1:
            self.wandb_run.log(payload, step=self.num_timesteps)


class RewardComponentDiagnosticsCallback(BaseCallback):
    """Log per-rollout reward component distributions from env info dicts."""

    component_info_keys = {
        "shaped": "shaped_reward",
        "raw": "raw_reward",
        "native": "native_reward_component",
        "prog": "progress_component",
        "prog_x": "progress_reward_component",
        "score": "score_reward_component",
        "score_d": "score_delta",
        "done": "completion_reward_component",
        "death": "death_penalty_component",
        "time": "time_penalty_component",
    }
    reward_share_components = ("prog_x", "score", "death", "done", "time", "native")

    def __init__(self):
        super().__init__()
        self.component_values: dict[str, list[float]] = {
            component: [] for component in self.component_info_keys
        }

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            for component, info_key in self.component_info_keys.items():
                value = info.get(info_key)
                if value is None:
                    continue
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(numeric_value):
                    self.component_values[component].append(numeric_value)
        return True

    def _on_rollout_end(self) -> None:
        self.record_reward_shares()
        for component, values in self.component_values.items():
            if not values:
                continue
            array = np.asarray(values, dtype=np.float64)
            prefix = f"{TRAIN_REWARD_COMPONENT_ROOT}/{component}"
            self.logger.record(stat_metric(prefix, "mean"), float(np.mean(array)))
            self.logger.record(stat_metric(prefix, "std"), float(np.std(array)))
            self.logger.record(stat_metric(prefix, "min"), float(np.min(array)))
            self.logger.record(stat_metric(prefix, "max"), float(np.max(array)))
            self.logger.record(stat_metric(prefix, "abs_mean"), float(np.mean(np.abs(array))))
            self.logger.record(stat_metric(prefix, "nonzero_rate"), float(np.mean(array != 0.0)))
            values.clear()

    def record_reward_shares(self) -> None:
        abs_sums: dict[str, float] = {}
        for component in self.reward_share_components:
            values = self.component_values.get(component, ())
            abs_sums[component] = float(np.sum(np.abs(np.asarray(values, dtype=np.float64))))

        total_abs_sum = sum(abs_sums.values())
        for component, abs_sum in abs_sums.items():
            share = abs_sum / total_abs_sum if total_abs_sum > 0.0 else 0.0
            self.logger.record(f"{TRAIN_REWARD_SHARE_ROOT}/{component}", share)


class DoneCounterCallback(BaseCallback):
    ep_window_size = 100

    def __init__(self, wandb_run=None, default_state: str | None = None):
        super().__init__()
        self.wandb_run = wandb_run
        self.default_state = default_state
        self.done_count = 0
        self.reason_counts: dict[str, int] = {}
        self.detail_counts: dict[str, int] = {}
        self.detail_episode_window: deque[tuple[str, ...]] = deque(
            maxlen=self.ep_window_size,
        )
        self.detail_metrics_seen: set[str] = set()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for done, info in zip(dones, infos, strict=False):
            if not bool(done) or bool(info.get("global_reset", False)):
                continue

            reason_payloads = self.done_reason_payloads(info)
            if bool(info.get("TimeLimit.truncated", False)) and "max_steps" not in reason_payloads:
                reason_payloads["max_steps"] = {}
            if not reason_payloads:
                reason_payloads["unclassified"] = {}

            payload = self.record_done(reason_payloads)

            if self.wandb_run is not None:
                self.wandb_run.log(
                    {
                        GLOBAL_STEP: self.num_timesteps,
                        **payload,
                    },
                    step=self.num_timesteps,
                )

        return True

    @staticmethod
    def done_reason_payloads(info: dict[str, Any]) -> dict[str, Any]:
        done_on_info = info.get("done_on_info")
        if isinstance(done_on_info, dict):
            return {str(reason): payload for reason, payload in done_on_info.items() if str(reason)}
        if isinstance(done_on_info, (list, tuple, set)):
            return {str(reason): {} for reason in done_on_info if str(reason)}
        if isinstance(done_on_info, str) and done_on_info:
            return {done_on_info: {}}
        return {}

    def record_done(self, reason_payloads: dict[str, Any]) -> dict[str, int | float]:
        self.done_count += 1
        episode_detail_metrics: list[str] = []
        for reason, payload in reason_payloads.items():
            self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1
            for metric in self.done_detail_metrics(reason, payload):
                self.detail_counts[metric] = self.detail_counts.get(metric, 0) + 1
                episode_detail_metrics.append(metric)
        self.detail_episode_window.append(tuple(episode_detail_metrics))
        self.detail_metrics_seen.update(episode_detail_metrics)
        return self.record_metrics()

    @staticmethod
    def done_detail_metrics(reason: str, payload: Any) -> tuple[str, ...]:
        if not isinstance(payload, dict):
            return ()
        has_prev = "prev" in payload and payload["prev"] is not None
        if has_prev:
            return (train_done_value_metric(reason, "from", payload["prev"]),)
        return ()

    @staticmethod
    def done_ep_window_rate_metric(metric: str) -> str:
        return f"{metric}/ep_window/rate"

    def record_ep_window_rates(self) -> dict[str, float]:
        denominator = len(self.detail_episode_window)
        if denominator < self.ep_window_size:
            return {}
        counts = dict.fromkeys(self.detail_metrics_seen, 0)
        for episode_metrics in self.detail_episode_window:
            for metric in set(episode_metrics):
                counts[metric] = counts.get(metric, 0) + 1
        return {
            self.done_ep_window_rate_metric(metric): count / denominator
            for metric, count in sorted(counts.items())
        }

    def record_metrics(self) -> dict[str, int | float]:
        payload: dict[str, int | float] = {TRAIN_DONE_ALL: self.done_count}
        payload.update(
            {train_done_reason_metric(reason): count for reason, count in self.reason_counts.items()},
        )
        payload.update(self.detail_counts)
        payload.update(self.record_ep_window_rates())
        payload.setdefault(TRAIN_DONE_MAX_STEPS, self.reason_counts.get("max_steps", 0))
        payload.setdefault(TRAIN_DONE_UNCLASSIFIED, self.reason_counts.get("unclassified", 0))
        for key, value in payload.items():
            self.logger.record(key, value)
        return payload
