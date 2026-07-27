from __future__ import annotations

from typing import Any


def bind_policy_action_space(model: Any, action_space: Any) -> None:
    bind = getattr(model, "bind_action_space", None)
    if callable(bind):
        bind(action_space)


def reset_policy_state(model: Any, lanes: Any | None = None) -> None:
    reset = getattr(model, "reset_episode" if lanes is None else "reset_lanes", None)
    if callable(reset):
        reset() if lanes is None else reset(lanes)
