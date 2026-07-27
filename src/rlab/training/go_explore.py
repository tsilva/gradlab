from __future__ import annotations

import re
import struct
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces

from rlab.action_contract import configured_action_meanings
from rlab.artifacts import install_model_bundle
from rlab.batch_runtime import BatchMetricRecord, EpisodeRecord
from rlab.early_stop import MetricEarlyStopStateMachine, MetricSample
from rlab.env import make_training_batch_runtime, preflight_state_archive_provider
from rlab.file_utils import atomic_write_json, file_sha256
from rlab.go_explore import GoExploreSearch
from rlab.metric_names import (
    TRAIN_GO_EXPLORE_ARCHIVE_BLOB_BYTES,
    TRAIN_GO_EXPLORE_ARCHIVE_BLOB_COUNT,
    TRAIN_GO_EXPLORE_ARCHIVE_CELL_COUNT,
    TRAIN_GO_EXPLORE_ARCHIVE_ENTRY_COUNT,
    TRAIN_GO_EXPLORE_ARCHIVE_RECENT_NEW_CELL_RATE,
    TRAIN_GO_EXPLORE_ARCHIVE_RECENT_VISIT_WINDOW,
    TRAIN_GO_EXPLORE_ARCHIVE_SELECTION_COUNT,
    TRAIN_GO_EXPLORE_ARCHIVE_UPDATE_COUNT,
    TRAIN_GO_EXPLORE_ARCHIVE_VISIT_COUNT,
    TRAIN_GO_EXPLORE_ARCHIVE_VISITS_PER_CELL,
    TRAIN_GO_EXPLORE_BEST_COMPLETED,
    TRAIN_GO_EXPLORE_BEST_PROGRAM_RUNS,
    TRAIN_GO_EXPLORE_BEST_PROGRAM_STEPS,
    TRAIN_GO_EXPLORE_BEST_PROGRESS,
    TRAIN_GO_EXPLORE_BEST_RETURN,
    TRAIN_GO_EXPLORE_IMPROVEMENT_COUNT,
    TRAIN_GO_EXPLORE_SUCCESS_GUIDED_CELL_COUNT,
    TRAIN_GO_EXPLORE_SUCCESS_GUIDED_SELECTION_COUNT,
    TRAIN_THROUGHPUT_LOOP_FPS,
    train_early_stop_metric,
)
from rlab.policy_bundle import write_canonical_json
from rlab.training_backend import BackendContext, CHECKPOINT_EVAL_ACCEPTANCE


DEFAULT_CONFIG: dict[str, Any] = {
    "explore_steps": 128,
    "run_duration_mean": 4.0,
    "run_duration_max": 32,
    "fallback_action": "noop",
    "log_interval_steps": 10_000,
}
_CELL_KEY = struct.Struct("<BBBBHHBBB")
_CELL_INFO_KEYS = (
    "levelHi",
    "levelLo",
    "area_id",
    "area_pointer",
    "x_pos",
    "y_pos",
    "loop_command_active",
    "loop_correct_count",
    "loop_pass_count",
    "player_motion",
    "player_power",
    "player_task",
)


def normalize_config(
    backend_id: str,
    config: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    if backend_id != "rlab.go-explore":
        raise ValueError(f"Go-Explore backend does not define {backend_id!r}")
    unexpected = sorted(set(config) - set(DEFAULT_CONFIG))
    if unexpected:
        raise ValueError(f"{label} has unexpected fields: {unexpected}")
    normalized = {**DEFAULT_CONFIG, **dict(config)}
    for key in ("explore_steps", "run_duration_max", "log_interval_steps"):
        value = normalized[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{label}.{key} must be a positive integer")
    mean = normalized["run_duration_mean"]
    if (
        not isinstance(mean, int | float)
        or isinstance(mean, bool)
        or not np.isfinite(float(mean))
        or float(mean) < 1.0
    ):
        raise ValueError(f"{label}.run_duration_mean must be a finite number >= 1")
    normalized["run_duration_mean"] = float(mean)
    fallback = normalized["fallback_action"]
    if not isinstance(fallback, str) or not fallback.strip():
        raise ValueError(f"{label}.fallback_action must be a non-empty string")
    return normalized


def _column(infos: Mapping[str, Any], name: str, n_envs: int) -> np.ndarray:
    if name not in infos:
        raise ValueError(f"Go-Explore requires provider info {name!r}")
    values = np.asarray(infos[name])
    if values.shape != (n_envs,):
        raise ValueError(f"Go-Explore provider info {name!r} must be lane-aligned")
    presence = infos.get(f"_{name}")
    if presence is not None and not np.all(np.asarray(presence, dtype=np.bool_)):
        raise ValueError(f"Go-Explore provider info {name!r} is missing on some lanes")
    return values


def _cell_keys(infos: Mapping[str, Any], n_envs: int) -> tuple[bytes, ...]:
    values = {name: _column(infos, name, n_envs) for name in _CELL_INFO_KEYS}
    keys: list[bytes] = []
    for lane in range(n_envs):
        correct = min(max(int(values["loop_correct_count"][lane]), 0), 7)
        passed = min(max(int(values["loop_pass_count"][lane]), 0), 7)
        route_phase = (
            int(int(values["player_task"][lane]) != 8)
            | (int(bool(values["loop_command_active"][lane])) << 1)
            | (correct << 2)
            | (passed << 5)
        )
        keys.append(
            _CELL_KEY.pack(
                int(values["levelHi"][lane]) & 0xFF,
                int(values["levelLo"][lane]) & 0xFF,
                int(values["area_id"][lane]) & 0xFF,
                int(values["area_pointer"][lane]) & 0xFF,
                (int(values["x_pos"][lane]) // 8) & 0xFFFF,
                (int(values["y_pos"][lane]) // 16) & 0xFFFF,
                route_phase,
                int(int(values["player_motion"][lane]) == 0),
                int(values["player_power"][lane]) & 0xFF,
            )
        )
    return tuple(keys)


def _reset_info_columns(runtime: Any) -> dict[str, np.ndarray]:
    return {
        name: np.asarray([runtime.reset_infos[lane][name] for lane in range(runtime.num_envs)])
        for name in _CELL_INFO_KEYS
    }


def _checkpoint_prefix(game: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", game).strip("_").lower()
    return f"go_explore_jerk_{slug or 'retro'}"


def _save_policy(
    search: GoExploreSearch,
    runtime: Any,
    context: BackendContext,
    *,
    model_path: Path,
    kind: str,
    step: int,
) -> Path:
    installed = install_model_bundle(
        model_path,
        save_checkpoint=lambda path: search.policy().save(path),
        args=context.args,
        config=context.environment,
        kind=kind,
        checkpoint_step_value=step,
        state_archive_summary=runtime.state_archive_summary(),
    )
    checkpoint_id = context.metric_store.record_checkpoint(
        run_name=str(context.args.run_name),
        kind=kind,
        step=step,
        path=installed,
        sha256=None,
        eval_required=context.args.checkpoint_eval_backend != "none",
    )
    print(
        f"{kind} Go-Explore JERK policy ready: id={checkpoint_id} step={step} path={installed}",
        flush=True,
    )
    return installed


def _metric_payload(
    search: GoExploreSearch,
    runtime: Any,
    *,
    elapsed: float,
) -> dict[str, int | float]:
    candidate = search.best_candidate()
    archive = runtime.state_archive_summary() or {}
    return {
        TRAIN_GO_EXPLORE_ARCHIVE_CELL_COUNT: search.archive_count,
        TRAIN_GO_EXPLORE_ARCHIVE_ENTRY_COUNT: int(archive.get("entry_count", 0)),
        TRAIN_GO_EXPLORE_ARCHIVE_BLOB_COUNT: int(archive.get("blob_count", 0)),
        TRAIN_GO_EXPLORE_ARCHIVE_BLOB_BYTES: int(archive.get("blob_bytes", 0)),
        TRAIN_GO_EXPLORE_ARCHIVE_SELECTION_COUNT: search.archive_selection_count,
        TRAIN_GO_EXPLORE_ARCHIVE_VISIT_COUNT: search.archive_visit_count,
        TRAIN_GO_EXPLORE_ARCHIVE_UPDATE_COUNT: search.archive_update_count,
        TRAIN_GO_EXPLORE_ARCHIVE_RECENT_NEW_CELL_RATE: (search.archive_recent_new_cell_rate),
        TRAIN_GO_EXPLORE_ARCHIVE_RECENT_VISIT_WINDOW: (search.archive_recent_visit_window),
        TRAIN_GO_EXPLORE_ARCHIVE_VISITS_PER_CELL: search.archive_visits_per_cell,
        TRAIN_GO_EXPLORE_SUCCESS_GUIDED_CELL_COUNT: search.success_guided_cell_count,
        TRAIN_GO_EXPLORE_SUCCESS_GUIDED_SELECTION_COUNT: (search.success_guided_selection_count),
        TRAIN_GO_EXPLORE_BEST_PROGRESS: candidate.progress if candidate else 0.0,
        TRAIN_GO_EXPLORE_BEST_RETURN: candidate.episode_return if candidate else 0.0,
        TRAIN_GO_EXPLORE_BEST_PROGRAM_STEPS: candidate.step_count if candidate else 0,
        TRAIN_GO_EXPLORE_BEST_PROGRAM_RUNS: len(candidate.runs) if candidate else 0,
        TRAIN_GO_EXPLORE_BEST_COMPLETED: int(candidate.completed) if candidate else 0,
        TRAIN_GO_EXPLORE_IMPROVEMENT_COUNT: search.improvement_count,
        TRAIN_THROUGHPUT_LOOP_FPS: search.global_step / max(elapsed, 1e-9),
    }


def _publish_metrics(
    context: BackendContext,
    search: GoExploreSearch,
    runtime: Any,
    *,
    elapsed: float,
    early_stop: MetricEarlyStopStateMachine | None,
) -> bool:
    payload = _metric_payload(search, runtime, elapsed=elapsed)
    update = (
        early_stop.update(
            {
                metric: MetricSample(value=float(payload[metric]), step=search.global_step)
                for metric in {
                    str(condition["metric"]) for condition in early_stop.conditions.values()
                }
                if metric in payload
            }
        )
        if early_stop is not None
        else None
    )
    if update is not None:
        for condition_id, observation in update.observations.items():
            payload.update(
                {
                    train_early_stop_metric(condition_id, "value"): observation.value,
                    train_early_stop_metric(condition_id, "best"): observation.best_value,
                    train_early_stop_metric(
                        condition_id, "patience/elapsed_steps"
                    ): observation.elapsed_steps,
                    train_early_stop_metric(
                        condition_id, "patience/progress"
                    ): observation.patience_progress,
                    train_early_stop_metric(condition_id, "would_trigger"): float(
                        observation.would_trigger
                    ),
                }
            )
    context.metric_store.append_metrics(
        payload,
        step=search.global_step,
        source="train",
        publish=context.wandb_enabled,
    )
    if update is None or update.stop_decision is None:
        return False
    atomic_write_json(
        context.run_dir / f"early_stop_decision-{str(context.args.attempt_id)}.json",
        update.stop_decision,
    )
    print(
        f"early stop: condition={update.stop_decision['condition_id']} "
        f"step={update.stop_decision['metric_step']}",
        flush=True,
    )
    return True


def run_go_explore(context: BackendContext) -> None:
    args = context.args
    config = context.environment
    n_envs = int(args.resolved_n_envs)
    if int(args.timesteps) % n_envs != 0:
        raise ValueError("Go-Explore timesteps must be divisible by n_envs")
    preflight = preflight_state_archive_provider(
        config=config,
        n_envs=n_envs,
        seed=args.seed,
        rom_binding=context.rom_binding,
        state_archive=args.state_archive,
    )
    if preflight is None:
        raise ValueError("Go-Explore requires state_archive")
    preflight_path = context.run_dir / "state_archive_preflight.json"
    write_canonical_json(preflight_path, preflight)
    args.state_archive_preflight_sha256 = file_sha256(preflight_path)
    runtime = make_training_batch_runtime(
        config,
        n_envs,
        args.seed,
        rom_binding=context.rom_binding,
        state_archive=args.state_archive,
        state_archive_root=context.run_dir / "state-archive",
    )
    try:
        if not isinstance(runtime.action_space, spaces.Discrete):
            raise ValueError("Go-Explore requires a discrete task action space")
        search = GoExploreSearch(
            n_envs=n_envs,
            seed=args.seed,
            action_names=configured_action_meanings(config),
            fallback_action=args.fallback_action,
            explore_steps=args.explore_steps,
            run_duration_mean=args.run_duration_mean,
            run_duration_max=args.run_duration_max,
        )
        runtime.reset(seed=args.seed)
        all_lanes = np.ones(n_envs, dtype=np.bool_)
        initial_entries = runtime.capture_archive_entries(
            all_lanes,
            metadata_by_lane={
                lane: {"algorithm": "go-explore", "kind": "initial"} for lane in range(n_envs)
            },
        )
        search.initialize(
            _cell_keys(_reset_info_columns(runtime), n_envs),
            initial_entries,
        )
        context.mark_ready()
        started_at = time.perf_counter()
        next_log = args.log_interval_steps
        next_checkpoint = args.checkpoint_freq if args.checkpoint_freq > 0 else None
        saved_checkpoint_steps: set[int] = set()
        early_stop = (
            MetricEarlyStopStateMachine(args.early_stop, label="early_stop")
            if args.early_stop
            else None
        )
        early_stopped = False
        while search.global_step < args.timesteps and not context.stop_flag.requested:
            batch = runtime.step(search.next_actions())
            records = runtime.drain_records()
            records_by_lane = {
                int(record.lane): record for record in records if isinstance(record, EpisodeRecord)
            }
            metric_record = next(
                (record for record in records if isinstance(record, BatchMetricRecord)),
                None,
            )
            progresses = (
                np.asarray(metric_record.metrics["global_max_x_pos"], dtype=np.float64)
                if metric_record is not None and "global_max_x_pos" in metric_record.metrics
                else (
                    _column(batch.transition_info, "xscrollHi", n_envs).astype(np.float64) * 256
                    + _column(batch.transition_info, "xscrollLo", n_envs)
                )
            )
            dones = np.logical_or(batch.terminated, batch.truncated)
            observation = search.observe(
                batch.rewards,
                dones,
                _cell_keys(batch.transition_info, n_envs),
                records_by_lane,
                progresses=progresses,
            )
            if np.any(observation.archive_mask):
                entries = runtime.capture_archive_entries(
                    observation.archive_mask,
                    metadata_by_lane={
                        int(lane): {
                            "algorithm": "go-explore",
                            "cell_key": _cell_keys(
                                batch.transition_info,
                                n_envs,
                            )[int(lane)].hex(),
                        }
                        for lane in np.flatnonzero(observation.archive_mask)
                    },
                )
                search.commit_archive(entries)
            completion_events = search.take_completion_events()
            if any(event.improved for event in completion_events):
                step = search.global_step
                _save_policy(
                    search,
                    runtime,
                    context,
                    model_path=context.checkpoint_dir
                    / f"{_checkpoint_prefix(config.game)}_{step}_steps.zip",
                    kind="checkpoint",
                    step=step,
                )
                saved_checkpoint_steps.add(step)
            if np.any(observation.restart_mask):
                entry_ids = search.restart(observation.restart_mask)
                runtime.restore_archive_entries(observation.restart_mask, entry_ids)
            step = search.global_step
            if step >= next_log:
                early_stopped = _publish_metrics(
                    context,
                    search,
                    runtime,
                    elapsed=time.perf_counter() - started_at,
                    early_stop=early_stop,
                )
                next_log += args.log_interval_steps
            while next_checkpoint is not None and step >= next_checkpoint:
                if step not in saved_checkpoint_steps:
                    _save_policy(
                        search,
                        runtime,
                        context,
                        model_path=context.checkpoint_dir
                        / f"{_checkpoint_prefix(config.game)}_{step}_steps.zip",
                        kind="checkpoint",
                        step=step,
                    )
                    saved_checkpoint_steps.add(step)
                next_checkpoint += args.checkpoint_freq
            if early_stopped:
                break
        _publish_metrics(
            context,
            search,
            runtime,
            elapsed=time.perf_counter() - started_at,
            early_stop=None,
        )
        if context.stop_flag.requested and args.checkpoint_freq > 0:
            _save_policy(
                search,
                runtime,
                context,
                model_path=context.checkpoint_dir
                / f"{_checkpoint_prefix(config.game)}_interrupted_{search.global_step}_steps.zip",
                kind="interrupted",
                step=search.global_step,
            )
        _save_policy(
            search,
            runtime,
            context,
            model_path=context.run_dir / "final_model.zip",
            kind="final",
            step=search.global_step,
        )
        atomic_write_json(
            context.run_dir / "state_archive_closure.json",
            {
                "schema_version": 1,
                "status": "closed",
                "step": search.global_step,
                "archive": runtime.state_archive_summary(),
            },
        )
    finally:
        runtime.close()


class GoExploreBackend:
    def validate(
        self,
        common_config: Mapping[str, Any],
        backend_config: Mapping[str, Any],
    ) -> None:
        del backend_config
        archive = common_config.get("state_archive")
        if not isinstance(archive, Mapping):
            raise ValueError("rlab.go-explore requires state_archive")
        if archive.get("restore_semantics", "continuation") != "continuation":
            raise ValueError("rlab.go-explore requires continuation archive restores")
        recorder = archive.get("recorder")
        if not isinstance(recorder, Mapping) or recorder.get("mode") != "backend":
            raise ValueError("rlab.go-explore requires state_archive.recorder.mode='backend'")
        if archive.get("curriculum") is not None:
            raise ValueError("rlab.go-explore owns selection; curriculum must be null")

    def run(self, context: BackendContext) -> None:
        run_go_explore(context)


_BACKEND = GoExploreBackend()


def backend_for_id(backend_id: str) -> GoExploreBackend:
    if backend_id != "rlab.go-explore":
        raise ValueError(f"Go-Explore backend does not define {backend_id!r}")
    return _BACKEND


def contract_payload(backend_id: str) -> dict[str, Any]:
    backend_for_id(backend_id)
    return {
        "schema_version": 1,
        "status": "available",
        "defaults": DEFAULT_CONFIG,
        "state_archive_priority_metrics": [],
    }


def acceptance_mode(
    backend_id: str,
    backend_config: Mapping[str, Any],
) -> str:
    del backend_config
    backend_for_id(backend_id)
    return CHECKPOINT_EVAL_ACCEPTANCE


def state_archive_priority_metrics(backend_id: str) -> tuple[str, ...]:
    backend_for_id(backend_id)
    return ()


def runtime_metadata(
    backend_id: str,
    backend_config: Mapping[str, Any],
) -> Mapping[str, str]:
    del backend_config
    backend_for_id(backend_id)
    return {
        "training_backend_id": backend_id,
        "algorithm_id": "jerk",
        "search_algorithm_id": "go-explore",
        "model_class": "rlab.jerk.JerkPolicy",
    }
