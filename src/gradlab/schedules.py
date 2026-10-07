from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from stable_baselines3.common.utils import get_schedule_fn

from gradlab.callbacks import CallbackHelper


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
        return scheduled_scalar(initial_value, final_value, elapsed_timesteps, schedule_timesteps)

    return schedule


def learning_rate_schedule(
    common_config: Mapping[str, Any],
    backend_config: Mapping[str, Any],
) -> float | Callable[[float], float]:
    milestones = backend_config.get("learning_rate_milestones")
    if milestones is not None:
        # Config validation owns point ordering and finite, non-negative rates.
        points = ((0, float(backend_config["learning_rate"])),) + tuple(
            (point["step"], point["value"]) for point in milestones
        )
        total = int(common_config["timesteps"])
        if total <= 0:
            raise ValueError("timesteps must be positive")

        def schedule(progress_remaining: float) -> float:
            step = (1.0 - min(max(progress_remaining, 0.0), 1.0)) * total
            for (start, initial), (end, final) in zip(points, points[1:]):
                if step <= end:
                    return scheduled_scalar(initial, final, step - start, end - start)
            return float(points[-1][1])

        return schedule
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
    ):
        super().__init__()
        if schedule_timesteps <= 0:
            raise ValueError("schedule_timesteps must be positive")
        self.initial_value = initial_value
        self.final_value = final_value
        self.schedule_timesteps = schedule_timesteps

    def _current_value(self) -> float:
        return scheduled_scalar(
            self.initial_value, self.final_value, self.num_timesteps, self.schedule_timesteps
        )

    def _on_training_start(self) -> None:
        self.model.ent_coef = self._current_value()

    def _on_step(self) -> bool:
        ent_coef = self._current_value()
        self.model.ent_coef = ent_coef
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


def scheduled_scalar(
    initial: float, final: float | None, step: int | float, duration: int
) -> float:
    """Interpolate by absolute training transitions and hold the endpoint."""
    if final is None:
        return float(initial)
    if duration <= 0:
        raise ValueError("schedule duration must be positive")
    progress = min(max(step / duration, 0.0), 1.0)
    return float(initial) + (float(final) - float(initial)) * progress


def apply_rollout_gamma(model: Any, config: Mapping[str, Any], total: int) -> float:
    """Freeze one discount for rollout collection, timeout bootstrap and GAE."""
    gamma = scheduled_scalar(
        config["gamma"],
        config.get("gamma_final"),
        int(model.num_timesteps),
        int(config.get("gamma_schedule_timesteps", 0) or total),
    )
    model.gamma = gamma
    buffer = getattr(model, "rollout_buffer", None)
    if buffer is not None:
        buffer.gamma = gamma
    return gamma


class GammaScheduleHelper(CallbackHelper):
    def __init__(self, config: Mapping[str, Any], total: int):
        super().__init__()
        self.config = dict(config)
        self.total = total

    def _on_rollout_start(self) -> None:
        apply_rollout_gamma(self.model, self.config, self.total)

    def _on_rollout_end(self) -> None:
        self.logger.record("train/gamma", float(self.model.gamma))
