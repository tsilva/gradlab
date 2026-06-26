from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from rlab.artifacts import checkpoint_step, log_wandb_model_artifact
from rlab.env import DoneOnInfoRules, EnvConfig
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
    train_done_from_rate_metric,
    train_done_value_metric,
    train_done_reason_metric,
    train_event_reason_metric,
    train_event_value_metric,
    train_outcome_from_rate_metric,
    train_outcome_value_metric,
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

    def __init__(
        self,
        wandb_run=None,
        default_state: str | None = None,
        done_on_info: DoneOnInfoRules | None = None,
    ):
        super().__init__()
        self.wandb_run = wandb_run
        self.default_state = default_state
        self.done_on_info = dict(done_on_info or {})
        self.done_count = 0
        self.reason_counts: dict[str, int] = {}
        self.detail_counts: dict[str, int] = {}
        self.detail_episode_windows: dict[str, deque[bool]] = {}

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

            payload = self.record_done(reason_payloads, info)

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

    def record_done(
        self,
        reason_payloads: dict[str, Any],
        info: Mapping[str, Any] | None = None,
    ) -> dict[str, int | float]:
        self.done_count += 1
        info = info or {}
        episode_detail_metrics: set[str] = set()
        for reason, payload in reason_payloads.items():
            self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1
            for metric in self.done_detail_metrics(reason, payload):
                self.detail_counts[metric] = self.detail_counts.get(metric, 0) + 1
                episode_detail_metrics.add(metric)
        self.record_detail_episode_windows(reason_payloads, episode_detail_metrics, info)
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

    def record_detail_episode_windows(
        self,
        reason_payloads: dict[str, Any],
        fired_detail_metrics: set[str],
        info: Mapping[str, Any],
    ) -> None:
        source_reasons = set(self.done_on_info)
        source_reasons.update(reason_payloads)
        for reason in sorted(source_reasons):
            source_value = self.source_value_for_reason(reason, reason_payloads.get(reason), info)
            if source_value is None:
                continue
            metric = train_done_value_metric(reason, "from", source_value)
            window = self.detail_episode_windows.setdefault(
                metric,
                deque(maxlen=self.ep_window_size),
            )
            window.append(metric in fired_detail_metrics)

    def source_value_for_reason(
        self,
        reason: str,
        payload: Any,
        info: Mapping[str, Any],
    ) -> Any | None:
        if isinstance(payload, dict) and "prev" in payload and payload["prev"] is not None:
            return payload["prev"]
        keys = self.source_keys_for_reason(reason, payload)
        if keys is None:
            return None
        return self.info_value_for_keys(info, keys)

    def source_keys_for_reason(self, reason: str, payload: Any) -> str | tuple[str, ...] | None:
        rule = self.done_on_info.get(reason)
        if rule is not None:
            key_or_keys, _op = rule
            return key_or_keys
        if isinstance(payload, dict) and "keys" in payload:
            key_or_keys = payload["keys"]
            if isinstance(key_or_keys, str):
                return key_or_keys
            if isinstance(key_or_keys, (list, tuple)):
                keys = tuple(str(item) for item in key_or_keys)
                return keys if keys else None
        return None

    @staticmethod
    def info_value_for_keys(
        info: Mapping[str, Any],
        keys: str | tuple[str, ...],
    ) -> Any | None:
        if isinstance(keys, str):
            return info.get(keys)
        values = []
        for key in keys:
            if key not in info:
                return None
            values.append(info[key])
        return tuple(values)

    def record_ep_window_rates(self) -> dict[str, float]:
        detail_rates = {
            metric: sum(window) / len(window)
            for metric, window in sorted(self.detail_episode_windows.items())
            if len(window) >= self.ep_window_size
        }
        payload = {
            self.done_ep_window_rate_metric(metric): rate for metric, rate in detail_rates.items()
        }

        rates_by_reason: dict[str, list[float]] = {}
        for metric, rate in detail_rates.items():
            reason = self.done_detail_metric_reason(metric)
            if reason is not None:
                rates_by_reason.setdefault(reason, []).append(rate)

        for reason, rates in sorted(rates_by_reason.items()):
            if len(rates) < 2:
                continue
            payload[train_done_from_rate_metric(reason, "min")] = min(rates)
            payload[train_done_from_rate_metric(reason, "mean")] = float(np.mean(rates))
        return payload

    @staticmethod
    def done_detail_metric_reason(metric: str) -> str | None:
        prefix = "train/done/"
        marker = "/from/"
        if not metric.startswith(prefix) or marker not in metric:
            return None
        reason, _value = metric.removeprefix(prefix).split(marker, 1)
        return reason or None

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


class OutcomeCounterCallback(BaseCallback):
    ep_window_size = 100

    def __init__(
        self,
        wandb_run=None,
        info_events: DoneOnInfoRules | None = None,
    ):
        super().__init__()
        self.wandb_run = wandb_run
        self.info_events = dict(info_events or {})
        self.event_counts: dict[str, int] = {}
        self.event_detail_counts: dict[str, int] = {}
        self.attempt_counts: dict[str, int] = {}
        self.fire_counts: dict[str, int] = {}
        self.attempt_windows: dict[str, deque[bool]] = {}
        self.current_sources: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        self.ensure_slots(len(infos))
        payload: dict[str, int | float] = {}

        for index, info in enumerate(infos):
            if bool(info.get("global_reset", False)):
                self.current_sources[index].clear()
                continue
            done = bool(dones[index]) if index < len(dones) else False
            payload.update(self.record_step(index, info, done))

        if payload and self.wandb_run is not None:
            self.wandb_run.log(
                {
                    GLOBAL_STEP: self.num_timesteps,
                    **payload,
                },
                step=self.num_timesteps,
            )
        return True

    def ensure_slots(self, count: int) -> None:
        while len(self.current_sources) < count:
            self.current_sources.append({})

    @staticmethod
    def info_event_payloads(info: Mapping[str, Any]) -> dict[str, Any]:
        info_events = info.get("info_events")
        if isinstance(info_events, dict):
            return {str(reason): payload for reason, payload in info_events.items() if str(reason)}
        return DoneCounterCallback.done_reason_payloads(dict(info))

    def record_step(
        self,
        index: int,
        info: Mapping[str, Any],
        done: bool,
    ) -> dict[str, int | float]:
        event_payloads = self.info_event_payloads(info)
        payload: dict[str, int | float] = {}
        for event, event_payload in event_payloads.items():
            payload.update(self.record_event(event, event_payload))

        tracked_events = tuple(dict.fromkeys((*self.info_events, *event_payloads)))
        for event in tracked_events:
            payload.update(
                self.record_event_attempt(
                    index=index,
                    event=event,
                    event_payloads=event_payloads,
                    info=info,
                    done=done,
                ),
            )
        return payload

    def record_event(self, event: str, payload: Any) -> dict[str, int | float]:
        self.event_counts[event] = self.event_counts.get(event, 0) + 1
        result: dict[str, int | float] = {
            train_event_reason_metric(event): self.event_counts[event],
        }
        source = self.payload_previous_value(payload)
        if source is not None:
            metric = train_event_value_metric(event, "from", source)
            self.event_detail_counts[metric] = self.event_detail_counts.get(metric, 0) + 1
            result[metric] = self.event_detail_counts[metric]
        self.record_metrics(result)
        return result

    def record_event_attempt(
        self,
        *,
        index: int,
        event: str,
        event_payloads: Mapping[str, Any],
        info: Mapping[str, Any],
        done: bool,
    ) -> dict[str, int | float]:
        source = self.source_value_for_event(event, event_payloads.get(event), info, index)
        if source is not None:
            self.current_sources[index][event] = source

        fired = event in event_payloads
        ended_without_fire = not fired and self.attempt_ended(event_payloads, info, done)
        if not fired and not ended_without_fire:
            return {}

        attempt_source = source if source is not None else self.current_sources[index].get(event)
        if attempt_source is None:
            return {}

        result = self.record_attempt(event, attempt_source, fired=fired)
        self.update_source_after_attempt(index, event, event_payloads.get(event), info, done)
        self.record_metrics(result)
        return result

    @staticmethod
    def payload_previous_value(payload: Any) -> Any | None:
        if isinstance(payload, dict) and "prev" in payload and payload["prev"] is not None:
            return payload["prev"]
        return None

    @staticmethod
    def payload_next_value(payload: Any) -> Any | None:
        if isinstance(payload, dict) and "next" in payload and payload["next"] is not None:
            return payload["next"]
        return None

    def source_value_for_event(
        self,
        event: str,
        payload: Any,
        info: Mapping[str, Any],
        index: int,
    ) -> Any | None:
        previous = self.payload_previous_value(payload)
        if previous is not None:
            return previous
        if event in self.current_sources[index]:
            return self.current_sources[index][event]
        keys = self.source_keys_for_event(event, payload)
        if keys is None:
            return None
        return DoneCounterCallback.info_value_for_keys(info, keys)

    def source_keys_for_event(self, event: str, payload: Any) -> str | tuple[str, ...] | None:
        rule = self.info_events.get(event)
        if rule is not None:
            key_or_keys, _op = rule
            return key_or_keys
        if isinstance(payload, dict) and "keys" in payload:
            key_or_keys = payload["keys"]
            if isinstance(key_or_keys, str):
                return key_or_keys
            if isinstance(key_or_keys, (list, tuple)):
                keys = tuple(str(item) for item in key_or_keys)
                return keys if keys else None
        return None

    @staticmethod
    def attempt_ended(
        event_payloads: Mapping[str, Any],
        info: Mapping[str, Any],
        done: bool,
    ) -> bool:
        return bool(
            done
            or info.get("died", False)
            or info.get("life_loss", False)
            or "life_loss" in event_payloads
            or info.get("TimeLimit.truncated", False)
        )

    def record_attempt(self, event: str, source: Any, *, fired: bool) -> dict[str, int | float]:
        metric = train_outcome_value_metric(event, "from", source)
        self.attempt_counts[metric] = self.attempt_counts.get(metric, 0) + 1
        if fired:
            self.fire_counts[metric] = self.fire_counts.get(metric, 0) + 1
        else:
            self.fire_counts.setdefault(metric, self.fire_counts.get(metric, 0))

        window = self.attempt_windows.setdefault(metric, deque(maxlen=self.ep_window_size))
        window.append(fired)

        payload: dict[str, int | float] = {
            f"{metric}/attempts": self.attempt_counts[metric],
            f"{metric}/fires": self.fire_counts.get(metric, 0),
        }
        payload.update(self.record_attempt_window_rates())
        return payload

    def record_attempt_window_rates(self) -> dict[str, float]:
        detail_rates = {
            metric: sum(window) / len(window)
            for metric, window in sorted(self.attempt_windows.items())
            if len(window) >= self.ep_window_size
        }
        payload = {
            f"{metric}/attempt_window/rate": rate for metric, rate in detail_rates.items()
        }

        rates_by_event: dict[str, list[float]] = {}
        for metric, rate in detail_rates.items():
            event = self.outcome_metric_event(metric)
            if event is not None:
                rates_by_event.setdefault(event, []).append(rate)

        for event, rates in sorted(rates_by_event.items()):
            if len(rates) < 2:
                continue
            payload[train_outcome_from_rate_metric(event, "min")] = min(rates)
            payload[train_outcome_from_rate_metric(event, "mean")] = float(np.mean(rates))
        return payload

    @staticmethod
    def outcome_metric_event(metric: str) -> str | None:
        prefix = "train/outcome/"
        marker = "/from/"
        if not metric.startswith(prefix) or marker not in metric:
            return None
        event, _value = metric.removeprefix(prefix).split(marker, 1)
        return event or None

    def update_source_after_attempt(
        self,
        index: int,
        event: str,
        payload: Any,
        info: Mapping[str, Any],
        done: bool,
    ) -> None:
        if done:
            reset_info = info.get("reset_info")
            keys = self.source_keys_for_event(event, payload)
            if isinstance(reset_info, Mapping) and keys is not None:
                reset_source = DoneCounterCallback.info_value_for_keys(reset_info, keys)
                if reset_source is not None:
                    self.current_sources[index][event] = reset_source
                    return
            self.current_sources[index].pop(event, None)
            return

        next_value = self.payload_next_value(payload)
        if next_value is not None:
            self.current_sources[index][event] = next_value
            return
        keys = self.source_keys_for_event(event, payload)
        if keys is None:
            self.current_sources[index].pop(event, None)
            return
        current_source = DoneCounterCallback.info_value_for_keys(info, keys)
        if current_source is None:
            self.current_sources[index].pop(event, None)
        else:
            self.current_sources[index][event] = current_source

    def record_metrics(self, payload: Mapping[str, int | float]) -> None:
        for key, value in payload.items():
            self.logger.record(key, value)
