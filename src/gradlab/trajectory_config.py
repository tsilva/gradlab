"""Resolved limits and the deliberately narrow training recording capability."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math

PROVIDER = "env-breakoutatari2600-turbo-native"
PROVIDER_VERSION = "0.5.13"
FORMAT = "gradlab.training-trajectories.v1"


@dataclass(frozen=True)
class CollectionConfig:
    enabled: bool = False
    contribution_bytes: int = 10 * 1024**3
    memory_bytes: int = 64 * 1024**2
    disk_bytes: int = 512 * 1024**2
    chunk_bytes: int = 32 * 1024**2
    scratch_headroom_bytes: int = 1024**3
    max_active_episodes: int = 1
    sample_probability: float = 0.05
    budget_stages: int = 20
    drain_seconds: float = 120.0


def resolve_collection(value, train_config: Mapping) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) - set(CollectionConfig.__dataclass_fields__):
        raise ValueError("trajectory_collection contains unknown settings")
    settings = asdict(CollectionConfig(**value))
    if not isinstance(settings["enabled"], bool):
        raise ValueError("trajectory_collection.enabled must be a boolean")
    for name in (
        "contribution_bytes",
        "memory_bytes",
        "disk_bytes",
        "chunk_bytes",
        "scratch_headroom_bytes",
        "max_active_episodes",
        "budget_stages",
    ):
        number = settings[name]
        if type(number) is not int or number <= 0:
            raise ValueError(f"trajectory_collection.{name} must be a positive integer")
    for name in ("drain_seconds", "sample_probability"):
        number = settings[name]
        if (
            isinstance(number, bool)
            or not isinstance(number, int | float)
            or not math.isfinite(number)
            or number <= 0
        ):
            raise ValueError(f"trajectory_collection.{name} must be finite and positive")
    if settings["sample_probability"] > 1 or settings["max_active_episodes"] > 16:
        raise ValueError("collection probability or active episode limit is too large")
    chunk = settings["chunk_bytes"]
    if not 1024**2 <= chunk <= 128 * 1024**2:
        raise ValueError("collection chunks must be between 1 and 128 MiB")
    if settings["memory_bytes"] < chunk + 2 * 1024**2 * settings["max_active_episodes"]:
        raise ValueError("collection memory limit must include chunk and handoff working space")
    if settings["disk_bytes"] < 2 * chunk or settings["contribution_bytes"] < chunk:
        raise ValueError("collection disk and contribution limits must accommodate chunks")
    if settings["budget_stages"] > 1000:
        raise ValueError("collection budget_stages must not exceed 1000")
    if settings["enabled"]:
        backend = train_config.get("training_backend") or {}
        if (
            train_config.get("game") != "Breakout-Atari2600-v0"
            or train_config.get("env_provider") != PROVIDER
            or backend.get("id") not in {"sb3.ppo", "sb3.a2c"}
            or train_config.get("sticky_action_prob", 0) != 0
        ):
            raise ValueError(
                "collection requires verified native Breakout with SB3 PPO/A2C and no sticky actions"
            )
        if not isinstance(train_config.get("timesteps"), int) or train_config["timesteps"] <= 0:
            raise ValueError("collection requires a finite positive planned training budget")
    return settings
