from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import replace
from typing import Any

from rlab.env import EnvConfig, task_termination


TERMINATION_OUTCOMES = ("failure", "success", "timeout", "neutral")
MAX_EPISODE_STEPS_ID = "limit:max_episode_steps"


def configured_termination_conditions(config: EnvConfig) -> tuple[dict[str, Any], ...]:
    termination = task_termination(config)
    conditions: list[dict[str, Any]] = []
    for outcome in TERMINATION_OUTCOMES:
        for event_name in termination.get(outcome, ()):
            name = str(event_name)
            conditions.append(
                {
                    "id": f"event:{name}",
                    "kind": "event",
                    "event": name,
                    "outcome": outcome,
                    "label": name.replace("_", " "),
                }
            )
    max_episode_steps = int(termination.get("max_episode_steps", 0))
    if max_episode_steps > 0:
        conditions.append(
            {
                "id": MAX_EPISODE_STEPS_ID,
                "kind": "limit",
                "event": None,
                "outcome": "timeout",
                "label": f"step limit ({max_episode_steps:,})",
                "value": max_episode_steps,
            }
        )
    return tuple(conditions)


def configured_termination_ids(config: EnvConfig) -> tuple[str, ...]:
    return tuple(str(condition["id"]) for condition in configured_termination_conditions(config))


def with_enabled_termination_conditions(
    config: EnvConfig,
    enabled_ids: Iterable[str],
) -> EnvConfig:
    enabled = {str(condition_id) for condition_id in enabled_ids}
    configured = set(configured_termination_ids(config))
    unknown = sorted(enabled - configured)
    if unknown:
        raise ValueError(f"unknown termination condition(s): {', '.join(unknown)}")

    task = deepcopy(config.task)
    termination = dict(task.get("termination", {}))
    for outcome in TERMINATION_OUTCOMES:
        if outcome not in termination:
            continue
        termination[outcome] = [
            str(event_name)
            for event_name in termination.get(outcome, ())
            if f"event:{event_name}" in enabled
        ]
    if "max_episode_steps" in termination:
        termination["max_episode_steps"] = (
            int(termination["max_episode_steps"]) if MAX_EPISODE_STEPS_ID in enabled else 0
        )
    task["termination"] = termination
    return replace(config, task=task)


def termination_condition_payload(
    base_config: EnvConfig,
    active_config: EnvConfig,
) -> list[dict[str, Any]]:
    active = set(configured_termination_ids(active_config))
    return [
        {
            **condition,
            "enabled": str(condition["id"]) in active,
        }
        for condition in configured_termination_conditions(base_config)
    ]
