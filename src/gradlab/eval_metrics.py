from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from gradlab.json_utils import json_value
from gradlab.metric_names import (
    EVAL_FULL_EPISODE_RETURN_SHAPED_MAX,
    EVAL_FULL_EPISODE_RETURN_SHAPED_MEAN,
    EVAL_FULL_OUTCOME_SUCCESS_STARTS_RATE_MEAN,
    EVAL_FULL_OUTCOME_SUCCESS_STARTS_RATE_MIN,
    eval_full_progress_metric,
)
from gradlab.policy_runtime import reset_policy_state
from gradlab.env_registry import EvalSemantics, environment_spec
from gradlab.task_kernels import Outcome

DEFAULT_EVAL_SEMANTICS = environment_spec(
    "env-stableretro-turbo",
    "SuperMarioBros-Nes-v0",
).eval_semantics


def default_eval_semantics() -> EvalSemantics:
    return DEFAULT_EVAL_SEMANTICS


def is_completion_event(
    info: dict[str, Any],
    semantics: EvalSemantics | None = None,
) -> bool:
    semantics = semantics or default_eval_semantics()
    for key in semantics.completion_info_keys:
        if key in info:
            return bool(info.get(key))
    return False


def is_level_complete(info: dict[str, Any]) -> bool:
    return is_completion_event(info, default_eval_semantics())


def drain_runtime_records(env: Any) -> list[Any]:
    """Drain all records from the native vector runtime."""
    drain = getattr(env, "drain_records", None)
    if not callable(drain):
        raise TypeError("this workflow requires GradLabVecEnv.drain_records()")
    return list(drain())


def episode_records(records: list[Any]) -> list[Any]:
    return [record for record in records if hasattr(record, "episode_return")]


def batch_metrics_for_lane(records: list[Any], lane: int) -> dict[str, Any]:
    """Materialize the latest task metric batch for an interactive consumer."""
    for record in reversed(records):
        if hasattr(record, "lane") or not hasattr(record, "num_envs"):
            continue
        metrics = getattr(record, "metrics", {}) or {}
        result: dict[str, Any] = {}
        for name, values in metrics.items():
            value = np.asarray(values)[lane]
            result[str(name)] = value.item() if isinstance(value, np.generic) else value
        return result
    return {}


def drain_episode_records(env: Any) -> list[Any]:
    """Drain canonical episode records from the native vector runtime."""
    return episode_records(drain_runtime_records(env))


def outcome_name(value: Any) -> str:
    if isinstance(value, Outcome):
        return value.name.lower()
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name.lower()
    if isinstance(value, str):
        return value.lower()
    try:
        return Outcome(int(value)).name.lower()
    except TypeError, ValueError:
        return "neutral"


def episode_is_complete(episode: Mapping[str, Any]) -> bool:
    if "level_complete" in episode:
        return bool(episode.get("level_complete"))
    return str(episode.get("outcome", "")).lower() == "success"


def episode_result_from_record(
    record: Any,
    *,
    semantics: EvalSemantics | None = None,
    terminal_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate the runtime's provider-neutral episode record to eval output."""
    semantics = semantics or default_eval_semantics()
    metrics = dict(getattr(record, "metrics", {}) or {})
    events = tuple(str(event) for event in (getattr(record, "events", ()) or ()))
    outcome = outcome_name(getattr(record, "outcome", Outcome.NEUTRAL))
    info = serializable_info(dict(terminal_info or {}))
    # Canonical task metrics are authoritative for overlapping provider fields.
    info.update(metrics)
    info = json_value(info)

    start_id = getattr(record, "start_id", None)
    result: dict[str, Any] = {
        "env_index": int(getattr(record, "lane", 0)),
        "episode_index": int(getattr(record, "episode_index", 0)),
        "start_state": start_id,
        "return": float(getattr(record, "episode_return", 0.0)),
        "score": int(info.get("score", 0) or 0),
        "lives": int(info.get("lives", 0) or 0),
        "time": int(info.get("time", 0) or 0),
        "steps": int(getattr(record, "episode_length", 0)),
        "terminated": bool(getattr(record, "terminated", False)),
        "truncated": bool(getattr(record, "truncated", False)),
        "outcome": outcome,
        "events": list(events),
        "final_info": info,
    }

    died = bool(metrics.get("died", False)) or "life_loss" in events
    if semantics.completion_reason:
        explicit_success = outcome == "success"
        explicit_failure = outcome == "failure"
        completion_signal = semantics.completion_reason in events or is_completion_event(
            info, semantics
        )
        result["level_complete"] = bool(
            (explicit_success and not died)
            or (completion_signal and not died and not explicit_failure)
        )

    for field in semantics.progress_fields:
        value = metrics.get(field.result_key, info.get(field.info_key))
        if value is None and field.result_key == "max_level_x_pos":
            value = metrics.get("max_x_pos", 0)
        result[field.result_key] = int(value or 0)

    if semantics.death_flag_key:
        death_x_pos = metrics.get("death_x_pos", info.get(semantics.death_position_key or ""))
        if died and death_x_pos is None:
            death_x_pos = result.get("max_x_pos", 0)
        result["died"] = died
        result["death_x_pos"] = int(death_x_pos) if death_x_pos is not None else None
    return result


def death_location_histogram(death_x_positions: list[int], bin_size: int = 100) -> dict[str, int]:
    bins: dict[str, int] = {}
    for x_pos in death_x_positions:
        start = (int(x_pos) // bin_size) * bin_size
        key = f"{start}-{start + bin_size - 1}"
        bins[key] = bins.get(key, 0) + 1
    return dict(sorted(bins.items(), key=lambda item: int(item[0].split("-", 1)[0])))


def episode_start_state(episode: dict[str, Any]) -> str | None:
    state = episode.get("start_state")
    return str(state) if state else None


def serializable_info(info: dict[str, Any]) -> dict[str, Any]:
    result = dict(info)
    result.pop("terminal_observation", None)
    return result


def episode_reasons(episode: Mapping[str, Any]) -> set[str]:
    if episode_is_complete(episode):
        return set()
    return episode_reason_names(
        episode.get("events", ()) or (),
        terminated=bool(episode.get("terminated")),
        truncated=bool(episode.get("truncated")),
    )


def episode_reason_names(
    events: Sequence[object],
    *,
    terminated: bool,
    truncated: bool,
) -> set[str]:
    """Return the shared train/eval terminal-reason taxonomy."""
    reasons = {str(event) for event in events if str(event) != "timeout"}
    if not reasons:
        if truncated:
            reasons.add("timeout")
        else:
            reasons.add("terminated" if terminated else "unclassified")
    return reasons


def eval_outcome_metrics(
    episode_results: list[dict[str, Any]],
    *,
    event_names: Sequence[str] = (),
    track_success: bool = False,
) -> dict[str, int | float]:
    del event_names
    metrics: dict[str, int | float] = {}
    success_rates: list[float] = []

    states = sorted(
        {state for episode in episode_results if (state := episode_start_state(episode))}
    )
    for state in states:
        state_episodes = [
            episode for episode in episode_results if episode_start_state(episode) == state
        ]
        denominator = len(state_episodes)
        if track_success:
            success_count = sum(episode_is_complete(episode) for episode in state_episodes)
            success_rate = success_count / denominator
            success_rates.append(success_rate)
    if success_rates:
        metrics[EVAL_FULL_OUTCOME_SUCCESS_STARTS_RATE_MIN] = min(success_rates)
        metrics[EVAL_FULL_OUTCOME_SUCCESS_STARTS_RATE_MEAN] = float(np.mean(success_rates))
    return metrics


def eval_by_start_records(episode_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    states = sorted(
        {state for episode in episode_results if (state := episode_start_state(episode))}
    )
    for state in states:
        episodes = [episode for episode in episode_results if episode_start_state(episode) == state]
        returns = np.asarray([episode["return"] for episode in episodes], dtype=np.float64)
        success_count = sum(episode_is_complete(episode) for episode in episodes)
        reason_counts: dict[str, int] = {}
        for episode in episodes:
            for reason in episode_reasons(episode):
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        records.append(
            {
                "start_id": state,
                "episode_count": len(episodes),
                "success_count": success_count,
                "success_rate": success_count / len(episodes),
                "shaped_return_mean": float(np.mean(returns)),
                "failure_reasons": dict(sorted(reason_counts.items())),
            }
        )
    return records


def primary_progress_value(
    result: dict[str, Any],
    semantics: EvalSemantics | None = None,
) -> float:
    semantics = semantics or default_eval_semantics()
    for field in semantics.progress_fields:
        if field.rank:
            return float(result.get(field.result_key, 0.0) or 0.0)
    return 0.0


def episode_rank(
    result: dict[str, Any],
    semantics: EvalSemantics | None = None,
) -> tuple[float, ...]:
    semantics = semantics or default_eval_semantics()
    values: list[float] = []
    for item in semantics.best_episode_rank:
        if item == "completion":
            values.append(float(episode_is_complete(result)))
        elif item == "progress":
            values.append(primary_progress_value(result, semantics))
        elif item == "reward":
            values.append(float(result.get("return", 0.0) or 0.0))
        else:
            values.append(float(result.get(item, 0.0) or 0.0))
    return tuple(values or [float(result.get("return", 0.0) or 0.0)])


def progress_summary_fields(result_key: str) -> tuple[str, str]:
    if result_key == "max_x_pos":
        return ("max_x_mean", "max_x_max")
    if result_key == "max_level_x_pos":
        return ("max_level_x_mean", "max_level_x_max")
    return (f"{result_key}_mean", f"{result_key}_max")


def progress_metric_name(result_key: str) -> str:
    if result_key == "max_x_pos":
        return "x"
    if result_key == "max_level_x_pos":
        return "level_x"
    return result_key.removeprefix("max_").removesuffix("_pos")


def single_env_action(action) -> int | np.ndarray:
    action_array = np.asarray(action)
    if action_array.shape == ():
        return int(action_array)
    first = np.asarray(action_array[0])
    if first.shape == ():
        return int(first)
    return first.astype(np.int8, copy=True)


def summarize_episode_results(
    episode_results: list[dict[str, Any]],
    *,
    deterministic: bool,
    extra: dict[str, Any] | None = None,
    semantics: EvalSemantics | None = None,
    event_names: Sequence[str] = (),
    track_success: bool = False,
) -> dict[str, Any]:
    if not episode_results:
        raise ValueError("episode_results must not be empty")
    semantics = semantics or default_eval_semantics()

    returns = np.array([episode["return"] for episode in episode_results], dtype=np.float64)
    lengths = np.array([episode["steps"] for episode in episode_results], dtype=np.float64)
    progress_metrics: dict[str, int | float] = {}
    for field in semantics.progress_fields:
        values = np.array(
            [episode.get(field.result_key, 0) for episode in episode_results],
            dtype=np.float64,
        )
        mean_key, max_key = progress_summary_fields(field.result_key)
        progress_metrics[mean_key] = float(values.mean())
        progress_metrics[max_key] = int(values.max())
        progress_name = progress_metric_name(field.result_key)
        progress_metrics[eval_full_progress_metric(progress_name, "mean")] = float(values.mean())
        progress_metrics[eval_full_progress_metric(progress_name, "max")] = int(values.max())
    death_x_positions = [
        int(episode["death_x_pos"])
        for episode in episode_results
        if episode.get("death_x_pos") is not None
    ]
    completion_count = sum(1 for episode in episode_results if episode_is_complete(episode))
    death_count = sum(1 for episode in episode_results if episode.get("died"))
    episode_count = len(episode_results)
    metrics: dict[str, Any] = {
        "episodes": episode_count,
        "deterministic": deterministic,
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "return_median": float(np.median(returns)),
        "episode_length_mean": float(lengths.mean()),
        EVAL_FULL_EPISODE_RETURN_SHAPED_MEAN: float(returns.mean()),
        EVAL_FULL_EPISODE_RETURN_SHAPED_MAX: float(returns.max()),
        "episode_results": episode_results,
    }
    metrics.update(progress_metrics)
    if track_success:
        metrics["success_count"] = completion_count
        metrics["success_rate"] = completion_count / episode_count
    if semantics.death_flag_key:
        metrics.update(
            {
                "death_count": death_count,
                "death_rate": death_count / episode_count,
                "death_x_histogram": death_location_histogram(death_x_positions),
            }
        )
    metrics.update(
        eval_outcome_metrics(
            episode_results,
            event_names=event_names,
            track_success=track_success,
        )
    )
    if extra:
        metrics = {**extra, **metrics}
    return metrics


def run_eval_episode(
    env,
    model,
    watchdog_steps: int,
    deterministic: bool,
    seed: int,
    capture_actions: bool = False,
    default_start_state: str | None = None,
    semantics: EvalSemantics | None = None,
    observation_callback: Callable[[object], object] | None = None,
    policy_runtime: Any | None = None,
    action_selection_mode: str | None = None,
) -> dict[str, Any]:
    semantics = semantics or default_eval_semantics()
    reset_policy_state(model)
    env.seed(seed)
    obs = env.reset()
    actions: list[Any] = []

    for _step_idx in range(watchdog_steps):
        if policy_runtime is None:
            action, _ = model.predict(obs, deterministic=deterministic)
        else:
            action = policy_runtime.decide(
                obs,
                action_selection_mode=action_selection_mode,
                execution_context=(
                    env.policy_execution_context(model)
                    if callable(getattr(env, "policy_execution_context", None))
                    else None
                ),
            ).actions
        action_value = single_env_action(action)
        if capture_actions:
            actions.append(action_value)
        obs, _rewards, dones, infos = env.step(action)
        if observation_callback is not None:
            observation_callback(obs)
        info = dict(infos[0])
        records = drain_episode_records(env)
        if records:
            result = episode_result_from_record(
                records[0],
                semantics=semantics,
                terminal_info=info,
            )
            if result.get("start_state") is None:
                result["start_state"] = default_start_state
            result["actions"] = actions
            return result
        if bool(dones[0]):
            raise RuntimeError("GradLabVecEnv returned done without an episode record")
    raise RuntimeError("evaluation watchdog expired without a scientific episode-boundary record")
