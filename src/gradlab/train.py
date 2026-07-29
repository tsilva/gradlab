from __future__ import annotations

# ruff: noqa: E402

import argparse
import os
import signal
import sys
import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from gradlab.artifacts import write_run_description
from gradlab.env import (
    assert_provider_runtime_available,
    default_run_dir,
    resolve_env_config,
    resolve_mixed_state_config,
    task_termination,
)
from gradlab.env_config import env_config_from_mapping
from gradlab.metric_store import MetricStore, metric_store_path
from gradlab.provider_config import provider_num_envs
from gradlab.policy_bundle import load_recipe_document
from gradlab.seeds import validate_training_seed
from gradlab.train_config import load_materialized_train_config
from gradlab.training_backend import (
    BackendContext,
    GracefulStopFlag,
    load_training_backend,
    training_backend_config,
    training_backend_config_hash,
    training_backend_id,
    training_backend_runtime_metadata,
)
from gradlab.training_lifecycle import (
    LEARNER_READY_FILENAME,
    TRAINING_RESULT_FILENAME,
    ProgressSink,
    TrainingExecutionMode,
    TrainingExecutionPolicy,
    TrainingResult,
    TrainingSession,
)
from gradlab.training_metrics import EpisodeMetricsReducer
from gradlab.rom_assets import (
    manifest_from_train_config,
    portable_rom_asset_identity,
    validate_rom_asset_manifest,
)
from gradlab.rom_runtime import (
    RomRuntimeBinding,
    bind_cached_rom,
    bind_rom_path,
    runtime_cache_root,
)


GRACEFUL_STOP_SIGNAL = getattr(signal, "SIGUSR1", None)
INTERNAL_LEARNER_ENV = "GRADLAB_INTERNAL_LEARNER"


@dataclass(frozen=True)
class TrainingRuntimeControl:
    """Ephemeral host controls that never enter a materialized run contract."""

    progress_sink: ProgressSink | None = None
    stop_flag: GracefulStopFlag | None = None
    signal_handlers_owned_by_host: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a configured backend on a registered provider environment"
    )
    parser.add_argument(
        "--train-config-json",
        type=Path,
        required=True,
        help="Authoritative materialized train configuration JSON.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=tuple(mode.value for mode in TrainingExecutionMode),
        required=True,
        help="Required internal learner lifecycle policy.",
    )
    return parser


def effective_n_envs(config: Mapping[str, object]) -> int:
    return provider_num_envs(config, explicit_n_envs=config.get("n_envs"))


def parse_train_invocation(
    argv: Sequence[str] | None = None,
) -> tuple[dict[str, object], TrainingExecutionMode]:
    parser = build_parser()
    parsed = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    config = load_materialized_train_config(Path(parsed.train_config_json))
    validate_training_seed(
        config["seed"],
        label="train_config.seed",
        seed_span=effective_n_envs(config),
    )
    return config, TrainingExecutionMode(parsed.execution_mode)


def parse_train_config(argv: Sequence[str] | None = None) -> dict[str, object]:
    config, _execution_mode = parse_train_invocation(argv)
    return config


def signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"signal-{signum}"


def graceful_stop_signals(*, include_sigint: bool = False) -> tuple[int, ...]:
    signals: list[int] = []
    if GRACEFUL_STOP_SIGNAL is not None:
        signals.append(int(GRACEFUL_STOP_SIGNAL))
    if include_sigint and int(signal.SIGINT) not in signals:
        signals.append(int(signal.SIGINT))
    return tuple(signals)


@contextmanager
def graceful_stop_signal_scope(
    stop_flag: GracefulStopFlag,
    *,
    include_sigint: bool = False,
) -> Iterator[tuple[int, ...]]:
    installed = graceful_stop_signals(include_sigint=include_sigint)
    previous = {signum: signal.getsignal(signum) for signum in installed}

    def handle_graceful_stop(signum, _frame) -> None:
        stop_flag.request(signal_name(signum))

    for signum in installed:
        signal.signal(signum, handle_graceful_stop)
    try:
        yield installed
    finally:
        for signum in reversed(installed):
            signal.signal(signum, previous[signum])


def main(
    argv: list[str] | None = None,
    *,
    runtime_rom_binding: RomRuntimeBinding | None = None,
    runtime_control: TrainingRuntimeControl | None = None,
) -> int:
    if os.environ.get(INTERNAL_LEARNER_ENV) != "1":
        raise RuntimeError(
            "gradlab.train is an internal learner entrypoint; use `gradlab experiment launch` to launch "
            "a dstack training task"
        )
    train_config, execution_mode = parse_train_invocation(argv)
    execution_policy = TrainingExecutionPolicy.for_mode(execution_mode)
    recipe_json_path = Path(str(train_config.get("recipe_json_path") or ""))
    if not recipe_json_path.is_file():
        raise RuntimeError("training requires the canonical versioned recipe.json")
    load_recipe_document(recipe_json_path)
    backend_id = training_backend_id(train_config)
    backend_config = training_backend_config(train_config)
    backend = load_training_backend(backend_id)
    backend.validate(train_config, backend_config)
    train_config.update(training_backend_runtime_metadata(backend_id, backend_config))
    train_config["training_backend_config_hash"] = training_backend_config_hash(train_config)
    train_config["training_execution"] = execution_policy.to_document()

    environment = resolve_env_config(env_config_from_mapping(train_config))
    n_envs = effective_n_envs(train_config)
    environment = resolve_mixed_state_config(environment, n_envs=n_envs)
    manifest = manifest_from_train_config(train_config, expected_game=environment.game)
    if runtime_rom_binding is not None:
        if manifest is None:
            raise ValueError("runtime ROM binding requires a ROM asset manifest")
        bound_manifest = validate_rom_asset_manifest(
            runtime_rom_binding.manifest,
            expected_game=environment.game,
            require_object_uri=False,
        )
        if portable_rom_asset_identity(bound_manifest) != portable_rom_asset_identity(manifest):
            raise ValueError("runtime ROM binding does not match the training ROM asset")
        rom_binding = bind_rom_path(manifest, runtime_rom_binding.path)
    else:
        rom_binding = (
            bind_cached_rom(
                manifest,
                cache_root=runtime_cache_root(container_default=True),
            )
            if manifest is not None
            else None
        )
    assert_provider_runtime_available(environment, rom_binding=rom_binding)
    train_config["resolved_n_envs"] = n_envs

    run_dir = Path(default_run_dir(str(train_config["run_name"]), str(train_config["runs_dir"])))
    checkpoint_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    if execution_policy.persist_intermediate_checkpoints:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / LEARNER_READY_FILENAME).unlink(missing_ok=True)
    (run_dir / TRAINING_RESULT_FILENAME).unlink(missing_ok=True)
    store = MetricStore(metric_store_path(run_dir))
    store.init()
    store.register_recovery_manifest(
        {
            "version": "supervisor-sqlite-recovery-v1",
            "run_name": str(train_config["run_name"]),
            "outbox": "sqlite-wal",
            "durable_delivery": "private-r2-segments",
            "configuration_sha256": hashlib.sha256(
                json.dumps(train_config, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest(),
            "absolute_source_paths": [
                str(metric_store_path(run_dir).resolve()),
                str(run_dir.resolve()),
            ],
            "wandb_routing": {
                "project": str(train_config.get("wandb_project") or ""),
                "run_id": str(train_config.get("wandb_run_id") or ""),
            },
            "recovery_owner": "run-supervisor",
        }
    )
    write_run_description(train_config, str(run_dir))
    run_description = str(train_config.get("run_description") or "").strip()

    stop_flag = runtime_control.stop_flag if runtime_control is not None else None
    stop_flag = stop_flag or GracefulStopFlag()
    context = BackendContext(
        train_config=train_config,
        environment=environment,
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        metric_store=store,
        wandb_enabled=bool(train_config["wandb"]),
        stop_flag=stop_flag,
        rom_binding=rom_binding,
        session=TrainingSession(
            run_dir=run_dir,
            backend_id=backend_id,
            metric_store=store,
            wandb_enabled=bool(train_config["wandb"]),
            stop_flag=stop_flag,
            early_stop_config=train_config.get("early_stop"),
            attempt_id=str(train_config["attempt_id"]),
            run_id=str(train_config.get("wandb_run_id") or train_config["run_name"]),
            reducer=EpisodeMetricsReducer(
                event_names=tuple(task_termination(environment).get("failure", ())),
                configured_starts=tuple(
                    environment.states or ((environment.state,) if environment.state else ())
                ),
                track_success=bool(
                    isinstance(environment.task.get("termination"), Mapping)
                    and environment.task["termination"].get("success")
                ),
            ),
            execution_policy=execution_policy,
            completion_signal_available=bool(task_termination(environment).get("success")),
            progress_sink=(
                runtime_control.progress_sink if runtime_control is not None else None
            ),
        ),
    )
    context.session.configure_checkpoints(
        run_name=str(train_config["run_name"]),
        eval_required=train_config["checkpoint_eval_backend"] != "none",
    )

    host_owns_signals = bool(
        runtime_control is not None and runtime_control.signal_handlers_owned_by_host
    )
    installed_signals = graceful_stop_signals(include_sigint=execution_policy.handle_sigint)
    signal_scope = (
        nullcontext(installed_signals)
        if host_owns_signals
        else graceful_stop_signal_scope(
            stop_flag,
            include_sigint=execution_policy.handle_sigint,
        )
    )
    with signal_scope as active_signals:
        context.session.event(
            f"run description: {run_description}"
            if run_description
            else "warning: --run-description is empty"
        )
        if active_signals:
            label = "signals" if len(active_signals) > 1 else "signal"
            context.session.event(
                f"graceful stop {label}: "
                + ", ".join(signal_name(signum) for signum in active_signals)
            )
        if (
            execution_mode == TrainingExecutionMode.LOCAL_DEMO
            and not context.session.completion_signal_available
        ):
            context.session.event(
                "no declared success signal; local training will continue until its configured "
                "budget or early-stop condition"
            )
        try:
            result = backend.run(context)
            if not isinstance(result, TrainingResult):
                raise RuntimeError(
                    f"training backend {backend_id!r} did not return a TrainingResult"
                )
            terminal_model_path = run_dir / result.model_path
            if result.model_path != "final_model.zip" or not terminal_model_path.is_file():
                raise RuntimeError(
                    f"training backend {backend_id!r} did not produce the terminal model "
                    f"{run_dir / 'final_model.zip'}"
                )
            if (
                execution_mode == TrainingExecutionMode.LOCAL_DEMO
                and checkpoint_dir.is_dir()
                and any(checkpoint_dir.iterdir())
            ):
                raise RuntimeError("local-demo training produced an intermediate checkpoint")
            context.session.finalize(result)
        except BaseException as exc:
            context.session.fail(exc)
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
