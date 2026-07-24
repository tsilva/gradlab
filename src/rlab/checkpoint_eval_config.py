from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def checkpoint_eval_max_steps(config: Mapping[str, Any]) -> int:
    explicit = int(config.get("post_train_eval_max_steps") or 0)
    if explicit > 0:
        return explicit
    environment = config.get("checkpoint_eval_environment")
    task = environment.get("task") if isinstance(environment, Mapping) else None
    termination = task.get("termination") if isinstance(task, Mapping) else None
    fallback = int(
        termination.get("max_episode_steps")
        if isinstance(termination, Mapping) and termination.get("max_episode_steps") is not None
        else 0
    )
    if fallback > 0:
        return fallback
    raise ValueError("checkpoint eval max steps are not materialized")
