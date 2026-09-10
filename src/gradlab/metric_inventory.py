from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from typing import Any

from gradlab.action_codecs import (
    VIZDOOM_DEATHMATCH_MULTIDISCRETE_CODEC,
    VIZDOOM_SHARED_MULTIDISCRETE_CODEC,
)
from gradlab.action_contract import declared_action_contract
from gradlab.metric_names import METRIC_DEFINITIONS, metric_value_segment, validate_metric_name
from gradlab.ranking import parse_objective_rank
from gradlab.reward_transform import reward_transform_from_reward

ACTOR_CRITIC_METRICS = frozenset(
    {
        "train/explained_variance",
        "train/policy_loss/mean",
        "train/value_loss/mean",
        "train/learning_rate",
        "train/entropy/mean",
        "train/noise/std/mean",
        "train/action/fraction/max",
        "train/value/mean",
        "train/value/std",
        "train/advantage/mean",
        "train/advantage/std",
    }
)
PPO_METRICS = frozenset({"train/kl/mean", "train/clip/fraction"})


def _unavailable_action_metrics(config: Mapping[str, Any]) -> frozenset[str]:
    """Use declared policy encodings; leave unknown runtime capabilities conditional."""
    action = config.get("task", {}).get("action", {})
    codec_type = (action.get("codec") or {}).get("type")
    if codec_type in {"discrete_lookup", VIZDOOM_SHARED_MULTIDISCRETE_CODEC}:
        return frozenset({"train/noise/std/mean"})
    if codec_type == VIZDOOM_DEATHMATCH_MULTIDISCRETE_CODEC:
        return frozenset({"train/noise/std/mean", "train/action/fraction/max"})
    if action.get("set", "native") != "native":
        return frozenset()
    contract = declared_action_contract(config)
    mode = contract.get("mode") if contract is not None else None
    if mode in {"discrete", "custom_discrete"}:
        return frozenset({"train/noise/std/mean"})
    if mode == "multi_discrete":
        return frozenset({"train/noise/std/mean", "train/action/fraction/max"})
    return frozenset()


def required_metric_names(config: Mapping[str, Any]) -> frozenset[str]:
    names = {item.metric for item in parse_objective_rank(config.get("selection_rank"))}
    names.update(str(rule["metric"]) for rule in config.get("checkpoint_eval_acceptance", ()) or ())
    names.update(
        str(rule["metric"])
        for rule in (config.get("early_stop") or {}).get("conditions", {}).values()
        if "metric" in rule
    )
    return frozenset(names)


def redundant_reward_metrics(
    task: Mapping[str, Any], *, required: frozenset[str] = frozenset()
) -> frozenset[str]:
    """Suppress only duplicates established by this exact task contract."""
    reward = task.get("reward", {})
    omitted: set[str] = set()
    if not reward_transform_from_reward(reward).active:
        omitted.update({"train/reward/task/mean", "train/reward/task/std"})
        components = active_reward_components(task)
        if (
            task.get("id") == "identity"
            and reward.get("reward_mode") in {"native", "events"}
            and len(components) == 1
        ):
            omitted.update(
                f"train/reward/part/{components[0]}/{stat}"
                for stat in ("mean", "fraction", "share")
            )
    omitted.update(f"train/reward/event/{event}/mean" for event in reward.get("event_rewards", {}))
    return frozenset(omitted - required)


def active_reward_components(task: Mapping[str, object]) -> tuple[str, ...]:
    reward = task.get("reward")
    if not isinstance(reward, Mapping):
        return ()
    components: list[str] = []
    reward_mode = str(reward.get("reward_mode") or "")
    if reward_mode == "sample-factory-v0":
        return (
            "kill",
            "death",
            "hit",
            "damage",
            "health",
            "armor",
            "weapon",
            "ammo",
            "weapon_hold",
        )
    if reward_mode == "native" or bool(reward.get("use_native_reward")):
        components.append("native")
    if isinstance(reward.get("cell_novelty"), Mapping):
        components.append("cell_novelty")
    event_rewards = reward.get("event_rewards")
    if isinstance(event_rewards, Mapping) or isinstance(reward.get("event_delta_rewards"), Mapping):
        components.append("event")
    if (
        float(reward.get("progress_reward_scale") or 0.0) != 0.0
        or float(reward.get("progress_reward_boost_scale") or 0.0) != 0.0
    ):
        components.append("progress")
    if reward_mode == "score":
        components.append("score")
    if (
        float(reward.get("terminal_reward") or 0.0) != 0.0
        or float(reward.get("completion_reward") or 0.0) != 0.0
    ):
        components.append("completion")
    if float(reward.get("death_penalty") or 0.0) != 0.0:
        components.append("death")
    if float(reward.get("time_penalty") or 0.0) != 0.0:
        components.append("time")
    return tuple(components)


@dataclass(frozen=True)
class MetricInventory:
    """Applicable series for one resolved run, independent of window readiness."""

    names: frozenset[str]
    reward_components: tuple[str, ...]

    def matches(self, selector: str) -> bool:
        pattern = re.escape(selector)
        pattern = re.sub(r"\\\{[a-z_]+\\\}", r"[A-Za-z0-9_.-]+", pattern)
        return any(re.fullmatch(pattern, name) for name in self.names)


def resolve_metric_inventory(config: Mapping[str, Any]) -> MetricInventory:
    task = config.get("task", {})
    backend = config.get("training_backend", {})
    backend_id = str(backend.get("id", ""))
    algorithm = backend_id.rsplit(".", 1)[-1]
    components = active_reward_components(task)
    termination = task.get("termination", {})
    success = bool(termination.get("success"))
    events = task.get("events", {})
    reward = task.get("reward", {})
    archive = config.get("state_archive")
    curriculum = isinstance(archive, Mapping) and archive.get("curriculum") is not None
    evaluation = config.get("checkpoint_eval_backend", "none") != "none"
    starts = config.get("states") or (config.get("state") or "default",)
    dimensions = {
        "progress": tuple(config.get("episode_progress_fields", ())),
        "start": tuple(metric_value_segment(start) for start in starts) if success else (),
        "reason": tuple(dict.fromkeys((*events, "timeout", "terminated", "unclassified"))),
        "component": components,
        "event": tuple(
            dict.fromkeys(
                (*reward.get("event_rewards", {}), *reward.get("event_delta_rewards", {}))
            )
        ),
        "condition": tuple((config.get("early_stop") or {}).get("conditions", {})),
    }
    names: set[str] = set()
    required = required_metric_names(config)
    omitted = (redundant_reward_metrics(task) | _unavailable_action_metrics(config)) - required
    for definition in METRIC_DEFINITIONS:
        name = definition.name
        if name.startswith(("eval/", "leader/")) and not evaluation:
            continue
        if name.startswith("train/success/") and not success:
            continue
        if name.startswith("train/curriculum/") and not curriculum:
            continue
        if name == "train/unique_cells/mean" and not reward.get("cell_novelty"):
            continue
        if name in ACTOR_CRITIC_METRICS and algorithm not in {"ppo", "a2c"}:
            continue
        if name in PPO_METRICS and algorithm != "ppo":
            continue
        if name == "train/program/steps" and algorithm not in {"go-explore", "jerk"}:
            continue
        if any(
            name.startswith(f"train/{candidate}/") and algorithm != candidate
            for candidate in ("go-explore", "jerk")
        ):
            continue
        fields = re.findall(r"\{([a-z_]+)\}", name)
        for values in product(*(dimensions[field] for field in fields)):
            concrete = validate_metric_name(name.format_map(dict(zip(fields, values, strict=True))))
            if concrete in omitted:
                continue
            if (
                len(starts) == 1
                and concrete not in required
                and (
                    concrete in {"train/success/mean", "eval/success/mean"}
                    or concrete == f"train/success/{metric_value_segment(starts[0])}/fraction"
                )
            ):
                continue
            names.add(concrete)
    # Scientific consumers remain required even when capability metadata cannot infer them.
    names.update(required)
    for name in names:
        validate_metric_name(name)
    return MetricInventory(frozenset(names), components)
