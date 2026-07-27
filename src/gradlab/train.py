from __future__ import annotations

# ruff: noqa: E402

import argparse
import os
import signal
import sys
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from gradlab.artifacts import write_run_description
from gradlab.env import (
    assert_provider_runtime_available,
    default_run_dir,
    resolve_env_config,
    resolve_mixed_state_config,
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
from gradlab.rom_assets import manifest_from_train_config
from gradlab.rom_runtime import bind_cached_rom, runtime_cache_root


GRACEFUL_STOP_SIGNAL = getattr(signal, "SIGUSR1", None)
INTERNAL_LEARNER_ENV = "GRADLAB_INTERNAL_LEARNER"


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
    return parser


def effective_n_envs(config: Mapping[str, object]) -> int:
    return provider_num_envs(config, explicit_n_envs=config.get("n_envs"))


def parse_train_config(argv: Sequence[str] | None = None) -> dict[str, object]:
    parser = build_parser()
    parsed = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    config = load_materialized_train_config(Path(parsed.train_config_json))
    validate_training_seed(
        config["seed"],
        label="train_config.seed",
        seed_span=effective_n_envs(config),
    )
    return config


def signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"signal-{signum}"


def install_graceful_stop_handler(stop_flag: GracefulStopFlag) -> int | None:
    if GRACEFUL_STOP_SIGNAL is None:
        return None

    def handle_graceful_stop(signum, _frame) -> None:
        stop_flag.request(signal_name(signum))

    signal.signal(GRACEFUL_STOP_SIGNAL, handle_graceful_stop)
    return int(GRACEFUL_STOP_SIGNAL)


def main(argv: list[str] | None = None) -> int:
    if os.environ.get(INTERNAL_LEARNER_ENV) != "1":
        raise RuntimeError(
            "gradlab.train is an internal learner entrypoint; use `gradlab experiment launch` to launch "
            "a dstack training task"
        )
    train_config = parse_train_config(argv)
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

    environment = resolve_env_config(env_config_from_mapping(train_config))
    n_envs = effective_n_envs(train_config)
    environment = resolve_mixed_state_config(environment, n_envs=n_envs)
    manifest = manifest_from_train_config(train_config, expected_game=environment.game)
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

    run_dir = Path(
        default_run_dir(str(train_config["run_name"]), str(train_config["runs_dir"]))
    )
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "learner_ready.json").unlink(missing_ok=True)
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
    if run_description:
        print(f"run description: {run_description}", flush=True)
    else:
        print("warning: --run-description is empty", flush=True)

    stop_flag = GracefulStopFlag()
    graceful_stop_signal = install_graceful_stop_handler(stop_flag)
    if graceful_stop_signal is not None:
        print(f"graceful stop signal: {signal_name(graceful_stop_signal)}", flush=True)

    context = BackendContext(
        train_config=train_config,
        environment=environment,
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        metric_store=store,
        wandb_enabled=bool(train_config["wandb"]),
        stop_flag=stop_flag,
        rom_binding=rom_binding,
    )
    backend.run(context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
