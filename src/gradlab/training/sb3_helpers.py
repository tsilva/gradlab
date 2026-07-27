from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stable_baselines3.common.logger import HumanOutputFormat

from gradlab.callbacks import CallbackHelper
from gradlab.file_utils import atomic_write_json


SB3_HUMAN_OUTPUT_MAX_LENGTH = 512


def disable_sb3_human_output_truncation(
    model, *, max_length: int = SB3_HUMAN_OUTPUT_MAX_LENGTH
) -> None:
    logger = getattr(model, "_logger", None)
    logger_attr = getattr(type(model), "logger", None)
    if logger is None and not isinstance(logger_attr, property):
        logger = getattr(model, "logger", None)
    if logger is None:
        return
    for output_format in getattr(logger, "output_formats", ()):
        if isinstance(output_format, HumanOutputFormat):
            output_format.max_length = max_length


class Sb3HumanOutputFormatHelper(CallbackHelper):
    def __init__(self, *, max_length: int = SB3_HUMAN_OUTPUT_MAX_LENGTH) -> None:
        super().__init__()
        self.max_length = max_length

    def _on_training_start(self) -> None:
        disable_sb3_human_output_truncation(self.model, max_length=self.max_length)

    def _on_step(self) -> bool:
        return True


class GracefulStopHelper(CallbackHelper):
    def __init__(self, stop_flag: Any, *, marker_path: Path | None = None) -> None:
        super().__init__()
        self.stop_flag = stop_flag
        self.marker_path = marker_path
        self.logged = False

    def _on_step(self) -> bool:
        # Returning False here would interrupt SB3 before the current transition
        # is added to the rollout buffer. The supervisor may request a stop at
        # any time, so acknowledge it here but let the on-policy rollout finish.
        return True

    def acknowledge_safe_boundary(self, *, num_timesteps: int) -> None:
        if not self.stop_flag.requested or self.logged:
            return
        reason = self.stop_flag.reason or "graceful stop"
        print(
            f"graceful stop requested by {reason}; stopped at the safe "
            f"on-policy update boundary at num_timesteps={num_timesteps}",
            flush=True,
        )
        if self.marker_path is not None:
            atomic_write_json(
                self.marker_path,
                {
                    "observed_at": datetime.now(UTC).isoformat(),
                    "reason": reason,
                    "num_timesteps": int(num_timesteps),
                    "pid": os.getpid(),
                    "boundary": "on_policy_update_end",
                },
            )
        self.logged = True


_STOP_AWARE_MODEL_CLASSES: dict[type, type] = {}


def _stop_aware_model_class(original_class: type) -> type:
    existing = _STOP_AWARE_MODEL_CLASSES.get(original_class)
    if existing is not None:
        return existing

    def collect_rollouts(
        model: Any,
        env: Any,
        callback: Any,
        rollout_buffer: Any,
        n_rollout_steps: int,
    ) -> bool:
        graceful_stop = model._gradlab_graceful_stop
        if graceful_stop.stop_flag.requested:
            graceful_stop.acknowledge_safe_boundary(
                num_timesteps=int(model.num_timesteps)
            )
            return False
        return bool(
            original_class.collect_rollouts(
                model,
                env,
                callback,
                rollout_buffer,
                n_rollout_steps=n_rollout_steps,
            )
        )

    def excluded_save_params(model: Any) -> list[str]:
        return [
            *original_class._excluded_save_params(model),
            "_gradlab_graceful_stop",
        ]

    stop_aware_class = type(
        f"GradLabStopAware{original_class.__name__}",
        (original_class,),
        {
            "__module__": __name__,
            "__slots__": (),
            "_gradlab_stop_aware": True,
            "collect_rollouts": collect_rollouts,
            "_excluded_save_params": excluded_save_params,
        },
    )
    _STOP_AWARE_MODEL_CLASSES[original_class] = stop_aware_class
    return stop_aware_class


def install_on_policy_safe_boundary_stop(
    model: Any,
    *,
    graceful_stop: GracefulStopHelper,
) -> Any:
    """Make the next SB3 rollout collection stop before stepping the environment."""

    if not getattr(type(model), "_gradlab_stop_aware", False):
        original_class = type(model)
        model.__class__ = _stop_aware_model_class(original_class)
    model._gradlab_graceful_stop = graceful_stop
    return model
