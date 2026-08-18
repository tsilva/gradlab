from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import shlex
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse

from gradlab.cli_parser import ExactArgumentParser
from gradlab.clock import utc_now as _utc_now
from gradlab.config_loader import RECIPE_TEMPLATE_VALUES, render_template_vars
from gradlab.env import task_termination
from gradlab.env_config import env_config_from_mapping
from gradlab.env_registry import resolve_env_provider
from gradlab.learner_profiles import (
    local_learner_profile_names,
    resolve_local_learner_profile,
)
from gradlab.local_paths import default_runs_dir
from gradlab.policy_bundle import (
    build_recipe_document,
    canonical_json_sha256,
    write_canonical_json,
)
from gradlab.recipe_catalog import LOCAL_RUN_RECEIPT, recipe_identity, resolve_recipe_source
from gradlab.recipe_documents import (
    compose_resolved_train_documents,
    prepare_checkpoint_eval_mode,
    repo_git_commit,
)
from gradlab.rom_assets import (
    DEFAULT_LOCAL_ROM_CACHE,
    direct_rom_asset_manifest,
    rom_asset_manifest_for_game,
)
from gradlab.rom_runtime import RomRuntimeBinding, bind_rom_path
from gradlab.seeds import DEFAULT_TRAIN_SEED
from gradlab.train import INTERNAL_LEARNER_ENV
from gradlab.training_lifecycle import (
    TRAINING_RESULT_FILENAME,
    TrainingExecutionMode,
    TrainingExecutionPolicy,
)
from gradlab.vizdoom_assets import bind_required_local_vizdoom_iwad


LOCAL_ROM_CACHE_ENV = "GRADLAB_ROM_CACHE_DIR"


def build_parser() -> argparse.ArgumentParser:
    parser = ExactArgumentParser(
        prog="gradlab train",
        description=(
            "Train a checked-in recipe locally and produce a directly playable policy bundle. "
            "Local runs are training-only and cannot establish goal acceptance or promotion."
        ),
    )
    parser.add_argument(
        "recipe",
        help=(
            "Built-in <goal-path>/<recipe> reference or a recipe YAML under an experiments tree."
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument(
        "--profile",
        choices=local_learner_profile_names(),
        help="Named native local learner profile; profiled runs use their configured lifecycle.",
    )
    parser.add_argument(
        "--set",
        dest="recipe_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable recipe override included in the portable recipe hash.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=default_runs_dir(),
        help="Local run root; defaults to ~/.config/gradlab/runs.",
    )
    parser.add_argument(
        "--run-name",
        help=(
            "Relative output directory under --runs-dir. By default gradlab creates a unique "
            "goal/recipe/timestamp-seed directory."
        ),
    )
    parser.add_argument(
        "--run-description",
        help="Override the recipe's rendered description for this local run.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Preserve the recipe's W&B logging settings. Local training disables W&B by default.",
    )
    parser.add_argument(
        "--rom-path",
        type=Path,
        help=(
            "Use a provider-compatible raw .nes ROM or the pinned Doom II IWAD in "
            "place for this run without registering or copying it."
        ),
    )
    parser.add_argument(
        "--no-tui",
        action="store_true",
        help="Disable the full-screen local training interface and use plain progress output.",
    )
    return parser


def _timestamp_slug() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return normalized or "recipe"


def _default_run_name(goal_id: str, recipe_id: str, seed: int) -> str:
    return f"{_slug(goal_id)}/{_slug(recipe_id)}/{_timestamp_slug()}-seed-{seed}"


def _safe_run_dir(runs_dir: Path, run_name: str) -> Path:
    pure = PurePosixPath(run_name)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or "\\" in run_name
    ):
        raise ValueError("--run-name must be a safe relative path under --runs-dir")
    root = runs_dir.expanduser().resolve()
    run_dir = (root / Path(*pure.parts)).resolve()
    if not run_dir.is_relative_to(root):
        raise ValueError("--run-name escapes --runs-dir")
    return run_dir


def _portable_path(path: Path, *, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _source_distribution() -> dict[str, str]:
    try:
        distribution = importlib.metadata.distribution("gradlab")
    except importlib.metadata.PackageNotFoundError:
        return {"name": "gradlab", "version": "uninstalled"}
    return {
        "name": str(distribution.metadata.get("Name") or "gradlab"),
        "version": str(distribution.version),
    }


def _distribution_direct_url() -> dict:
    try:
        text = importlib.metadata.distribution("gradlab").read_text("direct_url.json")
        document = json.loads(text or "{}")
    except (
        importlib.metadata.PackageNotFoundError,
        json.JSONDecodeError,
        OSError,
        TypeError,
    ):
        return {}
    return dict(document) if isinstance(document, dict) else {}


def _play_uvx_launcher() -> list[str]:
    distribution = _source_distribution()
    direct_url = _distribution_direct_url()
    parsed = urlparse(str(direct_url.get("url") or ""))
    if parsed.scheme == "file":
        source_path = Path(unquote(parsed.path))
        if source_path.is_dir():
            launcher = ["uvx", "--from", str(source_path)]
            dir_info = direct_url.get("dir_info")
            if isinstance(dir_info, dict) and dir_info.get("editable") is True:
                launcher.extend(("--with-editable", str(source_path)))
            launcher.extend(("--refresh-package", "gradlab", "gradlab"))
            return launcher
    return ["uvx", f"gradlab@{distribution['version']}"]


def _installed_source_commit() -> str | None:
    direct_url = _distribution_direct_url()
    vcs_info = direct_url.get("vcs_info") if isinstance(direct_url, dict) else None
    value = (
        str(vcs_info.get("commit_id") or "").strip().lower() if isinstance(vcs_info, dict) else ""
    )
    return value if re.fullmatch(r"[0-9a-f]{40}", value) else None


def _runtime_packages() -> list[str]:
    packages = {f"python=={platform.python_version()}"}
    for distribution in importlib.metadata.distributions():
        name = str(distribution.metadata.get("Name") or "").strip()
        version = str(distribution.version or "").strip()
        if name and version:
            packages.add(f"{name}=={version}")
    return sorted(packages, key=str.casefold)


def _render_run_description(
    document: dict,
    *,
    goal_path: Path,
    seed: int,
    explicit: str | None,
) -> str:
    if explicit is not None:
        description = explicit.strip()
        if not description:
            raise ValueError("--run-description must be non-empty")
        return description
    goal_id, recipe_id = recipe_identity(document)
    rendered = render_template_vars(
        {"description": document["description"]},
        path=goal_path,
        label="local recipe description",
        extra_context={
            **RECIPE_TEMPLATE_VALUES,
            "seed": seed,
            "goal_id": goal_id,
            "goal_slug": goal_id,
            "recipe_id": recipe_id,
            "recipe_slug": recipe_id,
            "slug": recipe_id,
            "env_id": document["train_config"].get("game") or "",
        },
    )
    return str(rendered["description"]).strip()


def _write_receipt(run_dir: Path, payload: dict) -> None:
    write_canonical_json(run_dir / LOCAL_RUN_RECEIPT, payload)


def _should_use_training_tui(*, disabled: bool) -> bool:
    if disabled:
        return False
    term = os.environ.get("TERM", "").strip().lower()
    stdin_isatty = getattr(sys.stdin, "isatty", lambda: False)
    stdout_isatty = getattr(sys.stdout, "isatty", lambda: False)
    return bool(stdin_isatty() and stdout_isatty() and term not in {"", "dumb"})


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("gradlab train must run on the Python main thread")
    profile = resolve_local_learner_profile(args.profile)
    if profile is not None:
        profile.validate_host()
    use_training_tui = _should_use_training_tui(disabled=bool(args.no_tui))
    source = resolve_recipe_source(args.recipe)
    overrides = list(args.recipe_overrides)
    if profile is not None:
        overrides.extend(profile.recipe_overrides)
    if not args.wandb:
        overrides.append("logging.wandb_mode=disabled")
    execution_mode = (
        profile.execution_mode if profile is not None else TrainingExecutionMode.LOCAL_DEMO
    )

    source_commit = repo_git_commit(source.repository_root) or _installed_source_commit()

    def prepare_local_document(value: dict[str, Any]) -> None:
        prepare_checkpoint_eval_mode(
            value,
            checkpoint_eval_backend="none",
        )
        bind_required_local_vizdoom_iwad(value, requested_path=args.rom_path)

    resolved_documents = compose_resolved_train_documents(
        source.goal_path,
        source.recipe_path,
        recipe_overrides=overrides,
        prepare_materialized=prepare_local_document,
        source_sha=source_commit or "",
    )
    document = resolved_documents.effective
    goal_id, recipe_id = recipe_identity(document)
    description = _render_run_description(
        document,
        goal_path=source.goal_path,
        seed=args.seed,
        explicit=args.run_description,
    )
    run_name = args.run_name or _default_run_name(goal_id, recipe_id, args.seed)
    run_dir = _safe_run_dir(args.runs_dir, run_name)

    config = dict(document["train_config"])
    config.update(
        {
            "seed": int(args.seed),
            "run_name": run_name,
            "run_description": description,
            "runs_dir": str(args.runs_dir.expanduser().resolve()),
            "goal_path": _portable_path(
                source.goal_path,
                root=source.repository_root,
            ),
            "recipe_path": _portable_path(
                source.recipe_path,
                root=source.repository_root,
            ),
            "goal_slug": goal_id,
            "recipe_slug": recipe_id,
            "source_sha": source_commit or "",
            "checkpoint_eval_backend": "none",
        }
    )
    provider = resolve_env_provider(str(config["env_provider"]))
    uses_local_rom_cache = provider.requires_external_rom_asset
    runtime_rom_binding: RomRuntimeBinding | None = None
    if (
        args.rom_path is not None
        and not uses_local_rom_cache
        and provider.provider_id != "vizdoom-turbo"
    ):
        raise ValueError(
            f"--rom-path is not valid for ROM-free provider {provider.provider_id!r}"
        )
    if uses_local_rom_cache:
        if args.rom_path is not None:
            manifest = direct_rom_asset_manifest(str(config["game"]), args.rom_path)
            runtime_rom_binding = bind_rom_path(manifest, args.rom_path.expanduser())
            config["rom_asset_manifest"] = manifest
        else:
            config["rom_asset_manifest"] = rom_asset_manifest_for_game(str(config["game"]))
    document["train_config"] = config
    document["description"] = description
    base_config = dict(resolved_documents.base["train_config"])
    if "rom_asset_manifest" in config:
        base_config["rom_asset_manifest"] = config["rom_asset_manifest"]
    resolved_documents.base["train_config"] = base_config

    recipe_document = build_recipe_document(
        document,
        repo_root=source.repository_root,
        source_commit=source_commit,
        source_distribution=_source_distribution(),
        run_description=description,
        seed=int(args.seed),
        runtime_packages=_runtime_packages(),
        base_materialized_recipe=resolved_documents.base,
        canonical_goal=resolved_documents.canonical_goal,
    )
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"local run directory already exists: {run_dir}; choose another --run-name"
        ) from exc
    recipe_path = run_dir / "recipe.json"
    write_canonical_json(recipe_path, recipe_document)
    config["recipe_json_path"] = str(recipe_path)
    config_path = run_dir / "train-config.json"
    write_canonical_json(config_path, config)

    started_at = _utc_now()
    receipt = {
        "document_type": "gradlab.local-run",
        "format_version": 1,
        "status": "running",
        "training_execution": TrainingExecutionPolicy.for_mode(
            execution_mode
        ).to_document(),
        "recipe_ref": source.reference,
        "goal_id": goal_id,
        "recipe_id": recipe_id,
        "recipe_sha256": canonical_json_sha256(recipe_document),
        "seed": int(args.seed),
        "learner_profile": profile.profile_id if profile is not None else None,
        "started_at": started_at,
        "model": None,
    }
    _write_receipt(run_dir, receipt)
    local_notices = (
        f"local training-only run: recipe={source.reference} seed={args.seed} output={run_dir}",
        *(
            (f"learner profile: {profile.profile_id} device={profile.device}",)
            if profile is not None
            else ()
        ),
        "checkpoint evaluation is disabled; this run cannot establish promotion or acceptance",
    )
    if not use_training_tui:
        for notice in local_notices:
            print(notice, flush=True)

    from gradlab.train import main as learner_main

    previous_internal = os.environ.get(INTERNAL_LEARNER_ENV)
    previous_rom_cache = os.environ.get(LOCAL_ROM_CACHE_ENV)
    os.environ[INTERNAL_LEARNER_ENV] = "1"
    if uses_local_rom_cache and runtime_rom_binding is None:
        os.environ[LOCAL_ROM_CACHE_ENV] = str(DEFAULT_LOCAL_ROM_CACHE)
    try:
        try:
            learner_args = [
                "--train-config-json",
                str(config_path),
                "--execution-mode",
                execution_mode.value,
            ]

            def invoke_learner(runtime_control=None) -> int:
                kwargs = {"runtime_rom_binding": runtime_rom_binding}
                if runtime_control is not None:
                    kwargs["runtime_control"] = runtime_control
                return learner_main(learner_args, **kwargs)

            if use_training_tui:
                try:
                    from gradlab.training_tui import (
                        LocalTrainingIdentity,
                        run_local_training_tui,
                    )
                except Exception as exc:
                    print(
                        f"warning: local training TUI unavailable; using plain output: {exc}",
                        flush=True,
                    )
                    for notice in local_notices:
                        print(notice, flush=True)
                    result = invoke_learner()
                else:
                    result = run_local_training_tui(
                        identity=LocalTrainingIdentity(
                            recipe=source.reference,
                            seed=int(args.seed),
                            output=str(run_dir),
                            notices=local_notices,
                            completion_signal_available=bool(
                                task_termination(env_config_from_mapping(config)).get("success")
                            ),
                        ),
                        learner=invoke_learner,
                    )
            else:
                result = invoke_learner()
        except BaseException as exc:
            receipt.update(
                {
                    "status": "failed",
                    "failed_at": _utc_now(),
                    "terminal_reason": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            _write_receipt(run_dir, receipt)
            raise
    finally:
        if previous_internal is None:
            os.environ.pop(INTERNAL_LEARNER_ENV, None)
        else:
            os.environ[INTERNAL_LEARNER_ENV] = previous_internal
        if uses_local_rom_cache and runtime_rom_binding is None:
            if previous_rom_cache is None:
                os.environ.pop(LOCAL_ROM_CACHE_ENV, None)
            else:
                os.environ[LOCAL_ROM_CACHE_ENV] = previous_rom_cache

    model_path = run_dir / "final_model.zip"
    result_path = run_dir / TRAINING_RESULT_FILENAME
    if result != 0 or not model_path.is_file() or not result_path.is_file():
        receipt.update(
            {
                "status": "failed",
                "failed_at": _utc_now(),
                "terminal_reason": "failed",
                "error": f"local training did not produce its terminal contract at {run_dir}",
            }
        )
        _write_receipt(run_dir, receipt)
        raise RuntimeError(f"local training did not produce {model_path} and {result_path}")
    try:
        training_result = json.loads(result_path.read_text(encoding="utf-8"))
        terminal_status = str(training_result.get("status") or "")
        terminal_reason = str(training_result["terminal_reason"])
        terminal_execution_mode = str(training_result["execution_mode"])
        first_completion_step = training_result.get("first_completion_step")
        if first_completion_step is not None:
            first_completion_step = int(first_completion_step)
        final_step = int(training_result["final_step"])
        requested_limit = int(training_result["requested_limit"])
        execution_limit = int(training_result["execution_limit"])
        terminal_model_kind = str(training_result["model_kind"])
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        receipt.update(
            {
                "status": "failed",
                "failed_at": _utc_now(),
                "terminal_reason": "failed",
                "error_type": type(exc).__name__,
                "error": "learner produced an invalid terminal result",
            }
        )
        _write_receipt(run_dir, receipt)
        raise RuntimeError("local learner produced an invalid terminal result") from exc
    if terminal_execution_mode != execution_mode.value:
        receipt.update(
            {
                "status": "failed",
                "failed_at": _utc_now(),
                "terminal_reason": "failed",
                "error": "learner reported the wrong execution mode",
            }
        )
        _write_receipt(run_dir, receipt)
        raise RuntimeError("local learner reported the wrong execution mode")
    if terminal_status not in {"completed", "interrupted"}:
        receipt.update(
            {
                "status": "failed",
                "failed_at": _utc_now(),
                "terminal_reason": str(training_result.get("terminal_reason") or "failed"),
                "error": "learner reported a non-playable terminal result",
            }
        )
        _write_receipt(run_dir, receipt)
        raise RuntimeError("local learner reported a non-playable terminal result")
    receipt.update(
        {
            "status": terminal_status,
            f"{terminal_status}_at": _utc_now(),
            "terminal_reason": terminal_reason,
            "first_completion_step": first_completion_step,
            "final_step": final_step,
            "requested_limit": requested_limit,
            "execution_limit": execution_limit,
            "model_kind": terminal_model_kind,
            "model": model_path.name,
        }
    )
    _write_receipt(run_dir, receipt)
    print(f"trained model: {model_path}", flush=True)
    if terminal_status == "interrupted":
        play_command = [
            *_play_uvx_launcher(),
            "play",
            "--model",
            str(model_path),
        ]
        if runtime_rom_binding is not None:
            play_command.extend(("--rom-path", str(runtime_rom_binding.path)))
        print(f"play interrupted model: {shlex.join(play_command)}", flush=True)
        return 130
    play_command = [
        *_play_uvx_launcher(),
        "play",
        "--recipe",
        source.reference,
        "--runs-dir",
        str(args.runs_dir.expanduser().resolve()),
    ]
    if runtime_rom_binding is not None:
        play_command.extend(("--rom-path", str(runtime_rom_binding.path)))
    print(f"play it: {shlex.join(play_command)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
