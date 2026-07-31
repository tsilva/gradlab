from __future__ import annotations

import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gymnasium import spaces

from gradlab.artifacts import install_model_bundle
from gradlab.action_contract import (
    action_contract_meanings,
    configured_action_meanings,
    runtime_action_contract,
)
from gradlab.batch_runtime import EpisodeRecord
from gradlab.env import make_training_vec_env
from gradlab.jerk import JerkSearch
from gradlab.metric_names import (
    TRAIN_ALGORITHM_JERK_BEST_RETURN_MEAN,
    TRAIN_ALGORITHM_JERK_BEST_SEQUENCE_LENGTH,
    TRAIN_ALGORITHM_JERK_ARCHIVE_SELECTED_PREFIX_RETURN_MEAN,
    TRAIN_ALGORITHM_JERK_EXPLOIT_PROBABILITY,
    TRAIN_ALGORITHM_JERK_RETAINED_COUNT,
    TRAIN_THROUGHPUT_LOOP_FPS,
)
from gradlab.training_backend import (
    CHECKPOINT_EVAL_ACCEPTANCE,
    FIRST_TRAINING_SUCCESS_ACCEPTANCE,
    BackendContext,
)
from gradlab.training_lifecycle import (
    ProgressField,
    ProgressValueFormat,
    TerminalReason,
    TrainingExecutionMode,
    TrainingResult,
)
from gradlab.training_metrics import episode_succeeded


DEFAULT_CONFIG: dict[str, Any] = {
    "acceptance_mode": CHECKPOINT_EVAL_ACCEPTANCE,
    "archive_replay_probability_initial": 0.25,
    "archive_replay_probability_max": 0.9,
    "protected_prefix_steps": 128,
    "max_prefix_shorten_steps": 128,
    "retained_limit": 256,
    "fallback_action": "noop",
    "log_interval_steps": 10_000,
}

_POSITIVE_INTEGER_FIELDS = {
    "max_prefix_shorten_steps",
    "retained_limit",
    "log_interval_steps",
}
_NON_NEGATIVE_INTEGER_FIELDS = {"protected_prefix_steps"}
_PROBABILITY_FIELDS = {
    "archive_replay_probability_initial",
    "archive_replay_probability_max",
}
_ACTION_FIELDS = {"fallback_action"}
_ACCEPTANCE_MODES = {CHECKPOINT_EVAL_ACCEPTANCE, FIRST_TRAINING_SUCCESS_ACCEPTANCE}
JERK_PROGRESS_FIELDS = (
    ProgressField(
        TRAIN_ALGORITHM_JERK_RETAINED_COUNT,
        "retained",
        ProgressValueFormat.COUNT,
    ),
    ProgressField(
        TRAIN_ALGORITHM_JERK_BEST_RETURN_MEAN,
        "best return",
    ),
    ProgressField(
        TRAIN_ALGORITHM_JERK_EXPLOIT_PROBABILITY,
        "archive replay",
        ProgressValueFormat.PERCENT,
    ),
)


def normalize_config(
    backend_id: str,
    config: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    if backend_id != "gradlab.jerk":
        raise ValueError(f"JERK backend module does not define {backend_id!r}")
    unexpected = sorted(set(config) - set(DEFAULT_CONFIG))
    if unexpected:
        raise ValueError(f"{label} has unexpected fields: {unexpected}")
    normalized = {**DEFAULT_CONFIG, **dict(config)}
    for key in _POSITIVE_INTEGER_FIELDS:
        value = normalized[key]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{label}.{key} must be a positive integer")
    for key in _NON_NEGATIVE_INTEGER_FIELDS:
        value = normalized[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{label}.{key} must be a non-negative integer")
    for key in _PROBABILITY_FIELDS:
        value = normalized[key]
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise ValueError(f"{label}.{key} must be a number")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{label}.{key} must be in [0, 1]")
    for key in _ACTION_FIELDS:
        if not isinstance(normalized[key], str) or not normalized[key].strip():
            raise ValueError(f"{label}.{key} must be a non-empty string")
    if normalized["acceptance_mode"] not in _ACCEPTANCE_MODES:
        allowed = ", ".join(sorted(_ACCEPTANCE_MODES))
        raise ValueError(f"{label}.acceptance_mode must be one of: {allowed}")
    if (
        normalized["archive_replay_probability_initial"]
        > normalized["archive_replay_probability_max"]
    ):
        raise ValueError(
            f"{label}.archive_replay_probability_initial must not exceed "
            f"{label}.archive_replay_probability_max"
        )
    return normalized


def _is_success(record: EpisodeRecord) -> bool:
    return episode_succeeded(record)


def _checkpoint_prefix(game: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", game).strip("_").lower()
    return f"jerk_{slug or 'retro'}"


def _save_policy_bundle(
    *,
    search: JerkSearch,
    context: BackendContext,
    model_path: Path,
    kind: str,
    step: int,
    env: Any,
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
            action_contract=runtime_action_contract(env),
        ),
    )


def _search_metric_payload(search: JerkSearch) -> dict[str, int | float]:
    candidate = search.best_candidate()
    return {
        TRAIN_ALGORITHM_JERK_RETAINED_COUNT: search.retained_count,
        TRAIN_ALGORITHM_JERK_EXPLOIT_PROBABILITY: search.archive_replay_probability,
        TRAIN_ALGORITHM_JERK_ARCHIVE_SELECTED_PREFIX_RETURN_MEAN: (
            search.archive_selected_prefix_return_mean
        ),
        TRAIN_ALGORITHM_JERK_BEST_SEQUENCE_LENGTH: (
            len(candidate.actions) if candidate is not None else 0
        ),
        TRAIN_ALGORITHM_JERK_BEST_RETURN_MEAN: (
            candidate.mean_return if candidate is not None else 0.0
        ),
    }


def _metric_payload(
    search: JerkSearch,
    *,
    step: int,
    elapsed: float,
) -> dict[str, int | float]:
    return {
        **_search_metric_payload(search),
        TRAIN_THROUGHPUT_LOOP_FPS: step / max(elapsed, 1e-9),
    }


def run_jerk(context: BackendContext) -> TrainingResult:
    common_config = context.train_config
    backend_config = context.backend_config
    config = context.environment
    n_envs = int(common_config["resolved_n_envs"])
    env = make_training_vec_env(
        config=config,
        n_envs=n_envs,
        seed=int(common_config["seed"]),
        rom_binding=getattr(context, "rom_binding", None),
    )
    try:
        if int(common_config["timesteps"]) % n_envs != 0:
            raise ValueError("JERK timesteps must be divisible by the environment count")
        if not isinstance(env.action_space, spaces.Discrete):
            raise ValueError("JERK requires a discrete task action space")
        runtime = getattr(env, "runtime", None)
        contract = getattr(runtime, "action_contract", None)
        action_names = (
            action_contract_meanings(contract)
            if isinstance(contract, Mapping)
            else configured_action_meanings(config)
        )
        if not action_names:
            raise ValueError("JERK requires declared semantic IDs for every discrete action")
        search = JerkSearch(
            n_envs=n_envs,
            seed=int(common_config["seed"]),
            total_timesteps=int(common_config["timesteps"]),
            action_names=action_names,
            fallback_action=str(backend_config["fallback_action"]),
            archive_replay_probability_initial=backend_config["archive_replay_probability_initial"],
            archive_replay_probability_max=backend_config["archive_replay_probability_max"],
            protected_prefix_steps=int(backend_config["protected_prefix_steps"]),
            max_prefix_shorten_steps=int(backend_config["max_prefix_shorten_steps"]),
            retained_limit=int(backend_config["retained_limit"]),
        )
        env.reset()
        budget = context.session.configure_budget(
            requested_limit=int(common_config["timesteps"]),
            step_quantum=n_envs,
            progress_fields=JERK_PROGRESS_FIELDS,
        )
        context.mark_ready()
        started_at = time.perf_counter()
        next_log = int(backend_config["log_interval_steps"])
        checkpoint_freq = int(common_config["checkpoint_freq"])
        next_checkpoint = checkpoint_freq if checkpoint_freq > 0 else None
        configured_starts = tuple(
            str(start)
            for start in (
                tuple(config.states)
                if getattr(config, "states", ())
                else (getattr(config, "state", None),)
            )
            if start
        )
        fallback_start = configured_starts[0] if configured_starts else "default"
        acceptance_mode = str(backend_config["acceptance_mode"])
        accepted = False
        early_stopped = False
        while search.global_step < budget.execution_total and not context.stop_flag.requested:
            actions = search.next_actions()
            _observations, rewards, dones, _infos = env.step(actions)
            records = env.drain_records()
            records_by_lane: dict[int, EpisodeRecord] = {}
            success_records: list[EpisodeRecord] = []
            for record in records:
                if isinstance(record, EpisodeRecord):
                    records_by_lane[int(record.lane)] = record
                    if str(getattr(record, "start_origin", "target")) == "target" and _is_success(
                        record
                    ):
                        success_records.append(record)
            search.observe(rewards, dones, records_by_lane)
            step = search.global_step
            context.session.advance(
                step,
                records,
                progress_metrics=_search_metric_payload(search),
            )
            stop_for_completion = context.session.observe_completion(
                step=step,
                qualified=bool(success_records),
            )
            if stop_for_completion:
                context.session.event(f"level completed; stopping JERK at step={step}")
                break
            if context.stop_flag.requested:
                break
            if acceptance_mode == FIRST_TRAINING_SUCCESS_ACCEPTANCE and success_records:
                accepted = True
                accepted_path = context.checkpoint_dir / (
                    f"{_checkpoint_prefix(config.game)}_{step}_steps.zip"
                )
                _save_policy_bundle(
                    search=search,
                    context=context,
                    model_path=accepted_path,
                    kind="checkpoint",
                    step=step,
                    env=env,
                )
                context.session.event(
                    f"accepted JERK action program at first training success: step={step} "
                    f"start={success_records[0].start_id or fallback_start}"
                )
                break
            if step >= next_log:
                early_stopped = context.session.report(
                    step=step,
                    metrics=_metric_payload(
                        search,
                        step=step,
                        elapsed=time.perf_counter() - started_at,
                    ),
                )
                next_log += int(backend_config["log_interval_steps"])
            while next_checkpoint is not None and step >= next_checkpoint:
                if step < budget.execution_total:
                    checkpoint_path = context.checkpoint_dir / (
                        f"{_checkpoint_prefix(config.game)}_{step}_steps.zip"
                    )
                    _save_policy_bundle(
                        search=search,
                        context=context,
                        model_path=checkpoint_path,
                        kind="checkpoint",
                        step=step,
                        env=env,
                    )
                next_checkpoint += checkpoint_freq
            if early_stopped:
                break

        step = search.global_step
        context.session.report(
            step=step,
            metrics=_metric_payload(
                search,
                step=step,
                elapsed=time.perf_counter() - started_at,
            ),
        )
        default_reason = (
            TerminalReason.TRAINING_ACCEPTANCE if accepted else TerminalReason.RESOURCE_EXHAUSTION
        )
        reason = context.session.terminal_reason(default_reason)
        if context.session.should_persist_interrupted_checkpoint(reason) and checkpoint_freq > 0:
            _save_policy_bundle(
                search=search,
                context=context,
                model_path=context.checkpoint_dir
                / f"{_checkpoint_prefix(config.game)}_interrupted_{step}_steps.zip",
                kind="interrupted",
                step=step,
                env=env,
            )
        terminal_kind = context.session.terminal_model_kind(reason)
        context.train_config["training_terminal"] = context.session.terminal_provenance(
            terminal_reason=reason,
            final_step=step,
        )
        final_path = context.run_dir / "final_model.zip"
        _save_policy_bundle(
            search=search,
            context=context,
            model_path=final_path,
            kind=terminal_kind,
            step=step,
            env=env,
            terminal=True,
        )
        context.session.event(
            f"saved {final_path} retained={search.retained_count} "
            f"episodes={search.completed_episodes} accepted={accepted} "
            f"early_stopped={early_stopped}"
        )
        if (
            context.session.execution_policy.mode == TrainingExecutionMode.SUPERVISED
            and acceptance_mode == FIRST_TRAINING_SUCCESS_ACCEPTANCE
            and not accepted
            and not context.stop_flag.requested
            and step >= int(common_config["timesteps"])
        ):
            raise RuntimeError(
                f"JERK exhausted {common_config['timesteps']} transitions "
                "without a goal success event"
            )
        return context.session.result(
            terminal_reason=reason,
            final_step=step,
            model_kind=terminal_kind,
        )
    finally:
        env.close()


class JerkBackend:
    backend_id = "gradlab.jerk"

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
        if common_config.get("policy_model") is not None:
            raise ValueError("gradlab.jerk does not support train_config.policy_model")
        normalized = backend_config
        if (
            normalized["acceptance_mode"] == FIRST_TRAINING_SUCCESS_ACCEPTANCE
            and common_config.get("checkpoint_eval_backend") != "none"
        ):
            raise ValueError(
                "training_backend.config.acceptance_mode=first_training_success requires "
                "checkpoint_eval_backend=none"
            )

    def run(self, context: BackendContext) -> TrainingResult:
        return run_jerk(context)

    def acceptance_mode(self, backend_config: Mapping[str, Any]) -> str:
        return str(
            self.normalize_config(
                backend_config,
                label="training_backend.config",
            )["acceptance_mode"]
        )

    def contract_payload(self) -> dict[str, Any]:
        return {"schema_version": 1, "status": "available", "defaults": DEFAULT_CONFIG}

    def state_archive_priority_metrics(self) -> tuple[str, ...]:
        return ()

    def runtime_metadata(self, backend_config: Mapping[str, Any]) -> Mapping[str, str]:
        del backend_config
        return {
            "training_backend_id": self.backend_id,
            "algorithm_id": "action-program",
            "search_algorithm_id": "jerk",
            "model_class": "gradlab.action_program.ActionProgramPolicy",
        }


BACKENDS = {JerkBackend.backend_id: JerkBackend()}
