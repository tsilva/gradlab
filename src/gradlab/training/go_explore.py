from __future__ import annotations

import re
import struct
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces

from gradlab.action_contract import configured_action_meanings
from gradlab.artifacts import install_model_bundle
from gradlab.batch_runtime import BatchMetricRecord, EpisodeRecord
from gradlab.env import make_training_batch_runtime, preflight_state_archive_provider
from gradlab.env_providers import MARIO_BASE_INFO_KEYS
from gradlab.env_registry import SUPERMARIOBROS_NES_TURBO_PROVIDER
from gradlab.file_utils import file_sha256
from gradlab.go_explore import GoExploreSearch
from gradlab.metric_names import (
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
)
from gradlab.policy_bundle import write_canonical_json
from gradlab.state_archive import state_archive_artifact_summary
from gradlab.training_backend import BackendContext, CHECKPOINT_EVAL_ACCEPTANCE
from gradlab.training_lifecycle import TerminalReason, TrainingResult


DEFAULT_CONFIG: dict[str, Any] = {
    "explore_steps": 128,
    "run_duration_mean": 4.0,
    "run_duration_max": 32,
    "fallback_action": "noop",
    "log_interval_steps": 10_000,
    "compaction_interval_steps": 250_000,
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
GO_EXPLORE_PROVIDER_INFO_KEYS = frozenset(_CELL_INFO_KEYS) | MARIO_BASE_INFO_KEYS


def normalize_config(
    backend_id: str,
    config: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    if backend_id != "gradlab.go-explore":
        raise ValueError(f"Go-Explore backend does not define {backend_id!r}")
    unexpected = sorted(set(config) - set(DEFAULT_CONFIG))
    if unexpected:
        raise ValueError(f"{label} has unexpected fields: {unexpected}")
    normalized = {**DEFAULT_CONFIG, **dict(config)}
    for key in (
        "explore_steps",
        "run_duration_max",
        "log_interval_steps",
        "compaction_interval_steps",
    ):
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
    infos = {
        name: np.asarray(
            [
                runtime.reset_infos[lane][name]
                for lane in range(runtime.num_envs)
                if name in runtime.reset_infos[lane]
            ]
        )
        for name in _CELL_INFO_KEYS
    }
    return {name: _column(infos, name, runtime.num_envs) for name in _CELL_INFO_KEYS}


def _runtime_environment_config(config: Any) -> Any:
    env_args = dict(config.env_args)
    configured_filter = env_args.get("info_filter")
    configured_keys: set[str] = set()
    if isinstance(configured_filter, Mapping):
        if str(configured_filter.get("mode", "all")) != "all":
            raise ValueError("gradlab.go-explore requires info_filter mode='all'")
        keys = configured_filter.get("keys")
        if keys is not None:
            if isinstance(keys, str | bytes) or not isinstance(keys, list | tuple):
                raise ValueError("gradlab.go-explore info_filter.keys must be a sequence")
            configured_keys.update(str(key) for key in keys)
    elif str(configured_filter) != "all":
        raise ValueError("gradlab.go-explore requires info_filter='all'")
    task = config.task if isinstance(config.task, Mapping) else {}
    signals = task.get("signals")
    if isinstance(signals, Mapping):
        for source in signals.values():
            configured_keys.update(
                (str(source),) if isinstance(source, str) else (str(name) for name in source)
            )
    env_args["info_filter"] = {
        "mode": "all",
        "keys": tuple(sorted(GO_EXPLORE_PROVIDER_INFO_KEYS | configured_keys)),
    }
    return replace(config, env_args=env_args)


def _checkpoint_prefix(game: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", game).strip("_").lower()
    return f"go_explore_{slug or 'retro'}"


def _save_policy(
    search: GoExploreSearch,
    runtime: Any,
    context: BackendContext,
    *,
    model_path: Path,
    kind: str,
    step: int,
    terminal: bool = False,
) -> Path | None:
    return context.session.checkpoints.save(
        kind=kind,
        step=step,
        model_path=model_path,
        terminal=terminal,
        save_bundle=lambda path, artifact_kind, artifact_step: install_model_bundle(
            path,
            save_checkpoint=lambda destination: search.policy().save(
                destination,
                artifact_discriminator=f"{artifact_kind}:{artifact_step}",
            ),
            train_config=context.train_config,
            config=context.environment,
            kind=artifact_kind,
            checkpoint_step_value=artifact_step,
            state_archive_summary=state_archive_artifact_summary(runtime),
        ),
    )


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


def run_go_explore(context: BackendContext) -> TrainingResult:
    common_config = context.train_config
    backend_config = context.backend_config
    config = context.environment
    runtime_config = _runtime_environment_config(config)
    n_envs = int(common_config["resolved_n_envs"])
    if int(common_config["timesteps"]) % n_envs != 0:
        raise ValueError("Go-Explore timesteps must be divisible by n_envs")
    preflight = preflight_state_archive_provider(
        config=runtime_config,
        n_envs=n_envs,
        seed=int(common_config["seed"]),
        rom_binding=context.rom_binding,
        state_archive=common_config["state_archive"],
    )
    if preflight is None:
        raise ValueError("Go-Explore requires state_archive")
    preflight_path = context.run_dir / "state_archive_preflight.json"
    write_canonical_json(preflight_path, preflight)
    common_config["state_archive_preflight_sha256"] = file_sha256(preflight_path)
    runtime = make_training_batch_runtime(
        runtime_config,
        n_envs,
        int(common_config["seed"]),
        rom_binding=context.rom_binding,
        state_archive=common_config["state_archive"],
        state_archive_root=context.run_dir / "state-archive",
    )
    try:
        if not isinstance(runtime.action_space, spaces.Discrete):
            raise ValueError("Go-Explore requires a discrete task action space")
        search = GoExploreSearch(
            n_envs=n_envs,
            seed=int(common_config["seed"]),
            action_names=configured_action_meanings(config),
            fallback_action=str(backend_config["fallback_action"]),
            explore_steps=int(backend_config["explore_steps"]),
            run_duration_mean=float(backend_config["run_duration_mean"]),
            run_duration_max=int(backend_config["run_duration_max"]),
        )
        runtime.reset(seed=int(common_config["seed"]))
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
        budget = context.session.configure_budget(
            requested_limit=int(common_config["timesteps"]),
            step_quantum=n_envs,
        )
        context.mark_ready()
        started_at = time.perf_counter()
        log_interval_steps = int(backend_config["log_interval_steps"])
        compaction_interval_steps = int(backend_config["compaction_interval_steps"])
        checkpoint_freq = int(common_config["checkpoint_freq"])
        next_log = ((search.global_step // log_interval_steps) + 1) * log_interval_steps
        next_compaction = (
            (search.global_step // compaction_interval_steps) + 1
        ) * compaction_interval_steps
        next_checkpoint = (
            ((search.global_step // checkpoint_freq) + 1) * checkpoint_freq
            if checkpoint_freq > 0
            else None
        )
        saved_checkpoint_steps: set[int] = set()
        early_stopped = False
        stopped_on_completion = False
        while search.global_step < budget.execution_total and not context.stop_flag.requested:
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
            cell_keys = _cell_keys(batch.transition_info, n_envs)
            observation = search.observe(
                batch.rewards,
                dones,
                cell_keys,
                records_by_lane,
                progresses=progresses,
            )
            if np.any(observation.archive_mask):
                entries = runtime.capture_archive_entries(
                    observation.archive_mask,
                    metadata_by_lane={
                        int(lane): {
                            "algorithm": "go-explore",
                            "cell_key": cell_keys[int(lane)].hex(),
                        }
                        for lane in np.flatnonzero(observation.archive_mask)
                    },
                )
                search.commit_archive(entries)
            completion_events = search.take_completion_events()
            improved = any(event.improved for event in completion_events)
            stop_for_completion = (
                bool(completion_events) and context.session.should_stop_on_first_completion()
            )
            if np.any(observation.restart_mask) and not stop_for_completion:
                entry_ids = search.restart(observation.restart_mask)
                runtime.restore_archive_entries(observation.restart_mask, entry_ids)
            step = search.global_step
            context.session.advance(step, records)
            if stop_for_completion:
                stopped_on_completion = True
                break
            if improved and step < budget.execution_total:
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
            while step >= next_compaction:
                compacted = runtime.retain_state_archive_entries(
                    tuple(cell.entry_id for cell in search.archive.values())
                )
                context.session.event(
                    "compacted ephemeral Go-Explore archive "
                    f"step={step} retained={compacted['retained_entries']} "
                    f"removed={compacted['removed_entries']}"
                )
                next_compaction += compaction_interval_steps
            if step >= next_log:
                early_stopped = context.session.report(
                    step=step,
                    metrics=_metric_payload(
                        search,
                        runtime,
                        elapsed=time.perf_counter() - started_at,
                    ),
                )
                next_log += log_interval_steps
            while next_checkpoint is not None and step >= next_checkpoint:
                if step < budget.execution_total and step not in saved_checkpoint_steps:
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
                next_checkpoint += checkpoint_freq
            if early_stopped:
                break
        if stopped_on_completion:
            context.session.event(
                f"level completed; stopping Go-Explore at step={search.global_step}"
            )
        context.session.report(
            step=search.global_step,
            metrics=_metric_payload(
                search,
                runtime,
                elapsed=time.perf_counter() - started_at,
            ),
        )
        default_reason = (
            TerminalReason.ALGORITHM_SUCCESS
            if stopped_on_completion
            else TerminalReason.RESOURCE_LIMIT
        )
        reason = context.session.terminal_reason(default_reason)
        terminal_kind = "interrupted" if reason == TerminalReason.INTERRUPTED else "final"
        _save_policy(
            search,
            runtime,
            context,
            model_path=context.run_dir / "final_model.zip",
            kind=terminal_kind,
            step=search.global_step,
            terminal=True,
        )
        return TrainingResult(
            reason=reason,
            step=search.global_step,
            model_kind=terminal_kind,
        )
    finally:
        runtime.close()


class GoExploreBackend:
    backend_id = "gradlab.go-explore"

    def normalize_config(
        self,
        config: Mapping[str, Any],
        *,
        label: str,
    ) -> dict[str, Any]:
        return normalize_config(self.backend_id, config, label=label)

    def validate(
        self,
        common_config: Mapping[str, Any],
        backend_config: Mapping[str, Any],
    ) -> None:
        del backend_config
        provider_id = str(common_config.get("env_provider") or "")
        if provider_id != SUPERMARIOBROS_NES_TURBO_PROVIDER.provider_id:
            raise ValueError("gradlab.go-explore requires env_provider='supermariobrosnes-turbo'")
        task = common_config.get("task")
        if not isinstance(task, Mapping) or task.get("id") != "mario":
            raise ValueError("gradlab.go-explore requires task.id='mario'")
        archive = common_config.get("state_archive")
        if not isinstance(archive, Mapping):
            raise ValueError("gradlab.go-explore requires state_archive")
        if archive.get("persistence") != "ephemeral":
            raise ValueError("gradlab.go-explore requires an ephemeral working state archive")
        if archive.get("restore_semantics", "continuation") != "continuation":
            raise ValueError("gradlab.go-explore requires continuation archive restores")
        recorder = archive.get("recorder")
        if not isinstance(recorder, Mapping) or recorder.get("mode") != "backend":
            raise ValueError("gradlab.go-explore requires state_archive.recorder.mode='backend'")
        if archive.get("curriculum") is not None:
            raise ValueError("gradlab.go-explore owns selection; curriculum must be null")

    def run(self, context: BackendContext) -> TrainingResult:
        return run_go_explore(context)

    def acceptance_mode(self, backend_config: Mapping[str, Any]) -> str:
        del backend_config
        return CHECKPOINT_EVAL_ACCEPTANCE

    def contract_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": "available",
            "defaults": DEFAULT_CONFIG,
            "provider_info_keys": sorted(GO_EXPLORE_PROVIDER_INFO_KEYS),
            "search_archive_persistence": "ephemeral",
            "persisted_artifact": "best-action-program",
            "state_archive_priority_metrics": [],
        }

    def state_archive_priority_metrics(self) -> tuple[str, ...]:
        return ()

    def runtime_metadata(self, backend_config: Mapping[str, Any]) -> Mapping[str, str]:
        del backend_config
        return {
            "training_backend_id": self.backend_id,
            "algorithm_id": "action-program",
            "search_algorithm_id": "go-explore",
            "model_class": "gradlab.action_program.ActionProgramPolicy",
        }


BACKENDS = {GoExploreBackend.backend_id: GoExploreBackend()}
