from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from stable_baselines3.common.utils import get_schedule_fn

from rlab.callbacks import CallbackHelper
from rlab.metric_names import train_algorithm_metric


def linear_decay_schedule(
    initial_value: float,
    final_value: float,
    total_timesteps: int,
    schedule_timesteps: int = 0,
) -> Callable[[float], float]:
    if schedule_timesteps <= 0:
        schedule_timesteps = total_timesteps
    if schedule_timesteps <= 0:
        raise ValueError("schedule_timesteps must be positive")

    def schedule(progress_remaining: float) -> float:
        progress_remaining = min(max(progress_remaining, 0.0), 1.0)
        elapsed_timesteps = (1.0 - progress_remaining) * total_timesteps
        progress = min(max(elapsed_timesteps / schedule_timesteps, 0.0), 1.0)
        return initial_value + (final_value - initial_value) * progress

    return schedule


def learning_rate_schedule(
    common_config: Mapping[str, Any],
    backend_config: Mapping[str, Any],
) -> float | Callable[[float], float]:
    if backend_config["learning_rate_final"] is None:
        return float(backend_config["learning_rate"])
    return linear_decay_schedule(
        float(backend_config["learning_rate"]),
        float(backend_config["learning_rate_final"]),
        int(common_config["timesteps"]),
        int(backend_config["learning_rate_schedule_timesteps"]),
    )


class EntropyCoefficientScheduleHelper(CallbackHelper):
    def __init__(
        self,
        initial_value: float,
        final_value: float,
        schedule_timesteps: int,
        algorithm_id: str = "ppo",
    ):
        super().__init__()
        if schedule_timesteps <= 0:
            raise ValueError("schedule_timesteps must be positive")
        self.initial_value = initial_value
        self.final_value = final_value
        self.schedule_timesteps = schedule_timesteps
        self.algorithm_id = algorithm_id

    def _current_value(self) -> float:
        progress = min(max(self.num_timesteps / self.schedule_timesteps, 0.0), 1.0)
        return self.initial_value + (self.final_value - self.initial_value) * progress

    def _on_training_start(self) -> None:
        self.model.ent_coef = self._current_value()

    def _on_step(self) -> bool:
        ent_coef = self._current_value()
        self.model.ent_coef = ent_coef
        self.logger.record(
            train_algorithm_metric(self.algorithm_id, "hyperparameter/entropy_coefficient"),
            ent_coef,
        )
        return True


def apply_resume_hyperparameters(
    model,
    common_config: Mapping[str, Any],
    backend_config: Mapping[str, Any],
) -> None:
    lr_schedule = learning_rate_schedule(common_config, backend_config)
    model.learning_rate = lr_schedule
    model.lr_schedule = get_schedule_fn(lr_schedule)
    model.ent_coef = backend_config["ent_coef"]
    model.vf_coef = backend_config["vf_coef"]
    model.n_epochs = backend_config["n_epochs"]
    model.batch_size = backend_config["batch_size"]
    model.clip_range = get_schedule_fn(backend_config["clip_range"])
    model.normalize_advantage = backend_config["normalize_advantage"]
    model.target_kl = backend_config["target_kl"]
    model.policy.optimizer.defaults["eps"] = backend_config["adam_eps"]
    for param_group in model.policy.optimizer.param_groups:
        param_group["eps"] = backend_config["adam_eps"]


def apply_a2c_resume_hyperparameters(
    model,
    common_config: Mapping[str, Any],
    backend_config: Mapping[str, Any],
) -> None:
    lr_schedule = learning_rate_schedule(common_config, backend_config)
    model.learning_rate = lr_schedule
    model.lr_schedule = get_schedule_fn(lr_schedule)
    model.ent_coef = backend_config["ent_coef"]
    model.vf_coef = backend_config["vf_coef"]
    model.max_grad_norm = backend_config["max_grad_norm"]
    model.normalize_advantage = backend_config["normalize_advantage"]
