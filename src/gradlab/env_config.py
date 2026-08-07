from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from typing import Any

from gradlab.env import validate_obs_crop, validate_obs_resize
from gradlab.environment_fields import EnvConfig
from gradlab.train_config import env_config_arg_fields


def parse_states(value: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if not value:
        return ()
    if isinstance(value, (list, tuple)):
        states = tuple(str(state).strip() for state in value)
        if any(not state for state in states):
            raise ValueError("--states must not contain empty state names")
        return states
    states = tuple(state.strip() for state in value.split(","))
    if any(not state for state in states):
        raise ValueError("--states must not contain empty state names")
    return states


def parse_state_probs(value: str | list[float] | tuple[float, ...]) -> tuple[float, ...]:
    if not value:
        return ()
    if isinstance(value, (list, tuple)):
        probs = tuple(float(prob) for prob in value)
        if any(not math.isfinite(prob) or prob < 0.0 for prob in probs) or not any(
            prob > 0.0 for prob in probs
        ):
            raise ValueError(
                "--state-probs values must be non-negative finite numbers with "
                "at least one positive value",
            )
        return probs
    probs: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            raise ValueError("--state-probs must not contain empty values")
        try:
            prob = float(item)
        except ValueError as exc:
            raise ValueError(f"--state-probs contains a non-numeric value: {item!r}") from exc
        if not math.isfinite(prob) or prob < 0.0:
            raise ValueError(
                "--state-probs values must be non-negative finite numbers with "
                "at least one positive value",
            )
        probs.append(prob)
    if not any(prob > 0.0 for prob in probs):
        raise ValueError(
            "--state-probs values must be non-negative finite numbers with "
            "at least one positive value",
        )
    return tuple(probs)


def parse_obs_crop(
    value: str | list[int] | tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int] | None:
    if value is None or value == "":
        return None
    raw: Any = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.startswith("["):
            try:
                raw = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"--obs-crop contains invalid JSON: {exc.msg}") from exc
        else:
            raw = [int(item.strip()) for item in text.split(",")]
    return validate_obs_crop(raw)


def _normalize_environment_field(field: Any, value: Any) -> Any:
    if field.sequence_items == "str":
        return parse_states(value)
    if field.sequence_items == "number":
        return parse_state_probs(value)
    if field.type_name == "obs_crop":
        return parse_obs_crop(value)
    if field.type_name == "obs_resize":
        return validate_obs_resize(value)
    return value


def env_config_from_args(
    args: argparse.Namespace,
    *,
    include_states: bool = False,
) -> EnvConfig:
    defaults = EnvConfig()

    def value(name: str, default: Any = None) -> Any:
        return getattr(args, name, getattr(defaults, name, default))

    config_kwargs: dict[str, Any] = {}
    for field in env_config_arg_fields():
        if field.mixed_state and not include_states:
            continue
        key = field.dest
        config_kwargs[key] = _normalize_environment_field(field, value(field.dest))
    return EnvConfig(**config_kwargs)


def env_config_from_mapping(config: Mapping[str, Any]) -> EnvConfig:
    config_kwargs: dict[str, Any] = {}
    for field in env_config_arg_fields():
        if field.dest not in config:
            continue
        config_kwargs[field.dest] = _normalize_environment_field(
            field,
            config[field.dest],
        )
    return EnvConfig(**config_kwargs)
