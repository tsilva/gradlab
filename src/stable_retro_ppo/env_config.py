from __future__ import annotations

import argparse
import math
from typing import Any

from stable_retro_ppo.env import EnvConfig


def parse_states(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    states = tuple(state.strip() for state in value.split(","))
    if any(not state for state in states):
        raise ValueError("--states must not contain empty state names")
    return states


def parse_state_probs(value: str) -> tuple[float, ...]:
    if not value:
        return ()
    probs: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            raise ValueError("--state-probs must not contain empty values")
        try:
            prob = float(item)
        except ValueError as exc:
            raise ValueError(f"--state-probs contains a non-numeric value: {item!r}") from exc
        if not math.isfinite(prob) or prob <= 0.0:
            raise ValueError("--state-probs values must be positive finite numbers")
        probs.append(prob)
    return tuple(probs)


def env_config_from_args(
    args: argparse.Namespace,
    *,
    max_episode_steps_attr: str = "max_episode_steps",
    include_states: bool = False,
    include_env_threads: bool = False,
) -> EnvConfig:
    defaults = EnvConfig()

    def value(name: str, default: Any = None) -> Any:
        return getattr(args, name, getattr(defaults, name, default))

    max_episode_steps = value(max_episode_steps_attr, defaults.max_episode_steps)
    config_kwargs: dict[str, Any] = {
        "game": value("game"),
        "state": value("state"),
        "frame_skip": value("frame_skip"),
        "max_pool_frames": value("max_pool_frames"),
        "sticky_action_prob": value("sticky_action_prob"),
        "max_episode_steps": max_episode_steps,
        "hud_crop_top": value("hud_crop_top"),
        "use_retro_reward": value("use_retro_reward"),
        "clip_rewards": value("clip_rewards"),
        "reward_mode": value("reward_mode"),
        "progress_reward_cap": value("progress_reward_cap"),
        "progress_reward_scale": value("progress_reward_scale"),
        "terminal_reward": value("terminal_reward"),
        "reward_scale": value("reward_scale"),
        "time_penalty": value("time_penalty"),
        "death_penalty": value("death_penalty"),
        "completion_reward": value("completion_reward"),
        "score_progress_clipped": value("score_progress_clipped"),
        "no_progress_timeout_steps": value("no_progress_timeout_steps"),
        "no_progress_min_delta": value("no_progress_min_delta"),
        "completion_x_threshold": value("completion_x_threshold"),
        "terminate_on_life_loss": value("terminate_on_life_loss"),
        "terminate_on_level_change": value("terminate_on_level_change"),
        "terminate_on_completion": value("terminate_on_completion"),
        "action_set": value("action_set"),
    }
    if include_states:
        config_kwargs["states"] = parse_states(value("states", ""))
        config_kwargs["state_probs"] = parse_state_probs(value("state_probs", ""))
    if include_env_threads:
        config_kwargs["env_threads"] = value("env_threads")
    return EnvConfig(**config_kwargs)
