from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
import re
from typing import Any

from gradlab.metric_names import METRIC_DEFINITIONS, metric_value_segment, validate_metric_name
from gradlab.ranking import parse_objective_rank


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
        "algorithm": (algorithm,) if algorithm in {"ppo", "a2c"} else (),
        "progress": tuple(config.get("episode_progress_fields", ())),
        "start": tuple(metric_value_segment(start) for start in starts) if success else (),
        "reason": tuple(dict.fromkeys((*events, "timeout", "terminated", "unclassified"))),
        "component": components,
        "event": tuple(dict.fromkeys((*reward.get("event_rewards", {}), *reward.get("event_delta_rewards", {})))),
        "condition": tuple((config.get("early_stop") or {}).get("conditions", {})),
    }
    names: set[str] = set()
    for definition in METRIC_DEFINITIONS:
        name = definition.name
        if name.startswith(("eval/", "leader/")) and not evaluation:
            continue
        if name.startswith("train/target/success/") and not success:
            continue
        if name.startswith("train/curriculum/archive/") and not curriculum:
            continue
        if name == "train/target/unique_cells_mean" and not reward.get("cell_novelty"):
            continue
        if name.startswith("train/reward/pre_transform/") and reward.get("reward_scale", 1.0) == 1.0 and not reward.get("reward_clip"):
            continue
        if any(name.startswith(f"train/{candidate}/") and algorithm != candidate for candidate in ("ppo", "a2c", "go-explore", "jerk")):
            continue
        fields = re.findall(r"\{([a-z_]+)\}", name)
        for values in product(*(dimensions[field] for field in fields)):
            names.add(validate_metric_name(name.format_map(dict(zip(fields, values, strict=True)))))
    # Scientific consumers remain required even when capability metadata cannot infer them.
    names.update(item.metric for item in parse_objective_rank(config.get("selection_rank")))
    names.update(str(rule["metric"]) for rule in config.get("checkpoint_eval_acceptance", ()) or ())
    names.update(str(rule["metric"]) for rule in (config.get("early_stop") or {}).get("conditions", {}).values() if "metric" in rule)
    for name in names:
        validate_metric_name(name)
    return MetricInventory(frozenset(names), components)
