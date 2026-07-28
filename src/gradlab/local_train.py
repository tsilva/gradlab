from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from gradlab.config_loader import RECIPE_TEMPLATE_VALUES, render_template_vars
from gradlab.env_registry import resolve_env_provider
from gradlab.policy_bundle import (
    build_recipe_document,
    canonical_json_sha256,
    write_canonical_json,
)
from gradlab.recipe_catalog import LOCAL_RUN_RECEIPT, recipe_identity, resolve_recipe_source
from gradlab.recipe_documents import compose_train_document, prepare_checkpoint_eval_mode
from gradlab.rom_assets import (
    DEFAULT_LOCAL_ROM_CACHE,
    direct_rom_asset_manifest,
    rom_asset_manifest_for_game,
)
from gradlab.rom_runtime import RomRuntimeBinding, bind_rom_path
from gradlab.seeds import DEFAULT_TRAIN_SEED
from gradlab.train import INTERNAL_LEARNER_ENV


LOCAL_ROM_CACHE_ENV = "GRADLAB_ROM_CACHE_DIR"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gradlab train",
        description=(
            "Train a checked-in recipe locally and produce a directly playable policy bundle. "
            "Local runs are training-only and cannot establish goal acceptance or promotion."
        ),
    )
    parser.add_argument(
        "recipe",
        help=(
            "Built-in <goal-path>/<recipe> reference or a recipe YAML under an "
            "experiments tree."
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAIN_SEED)
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
        default=Path("runs"),
        help="Local run root; defaults to ./runs.",
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
        "--rom",
        "--rom-path",
        dest="rom_path",
        type=Path,
        help=(
            "Use a provider-compatible raw .nes ROM in place for this run without "
            "registering or copying it."
        ),
    )
    return parser


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


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


def _git_commit(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip().lower()
    return value if re.fullmatch(r"[0-9a-f]{40}", value) else None


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


def _installed_source_commit() -> str | None:
    try:
        direct_url_text = importlib.metadata.distribution("gradlab").read_text("direct_url.json")
        direct_url = json.loads(direct_url_text or "{}")
    except (
        importlib.metadata.PackageNotFoundError,
        json.JSONDecodeError,
        OSError,
        TypeError,
    ):
        return None
    vcs_info = direct_url.get("vcs_info") if isinstance(direct_url, dict) else None
    value = (
        str(vcs_info.get("commit_id") or "").strip().lower()
        if isinstance(vcs_info, dict)
        else ""
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    source = resolve_recipe_source(args.recipe)
    overrides = list(args.recipe_overrides)
    if not args.wandb:
        overrides.extend(("logging.wandb=false", "logging.wandb_mode=disabled"))

    document = compose_train_document(
        source.goal_path,
        source.recipe_path,
        recipe_overrides=overrides,
        prepare_materialized=lambda value: prepare_checkpoint_eval_mode(
            value,
            checkpoint_eval_backend="none",
        ),
    )
    goal_id, recipe_id = recipe_identity(document)
    description = _render_run_description(
        document,
        goal_path=source.goal_path,
        seed=args.seed,
        explicit=args.run_description,
    )
    run_name = args.run_name or _default_run_name(goal_id, recipe_id, args.seed)
    run_dir = _safe_run_dir(args.runs_dir, run_name)

    source_commit = _git_commit(source.repository_root) or _installed_source_commit()
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
            "stop_on_acceptance": False,
        }
    )
    provider = resolve_env_provider(str(config["env_provider"]))
    uses_local_rom_cache = provider.requires_external_rom_asset
    runtime_rom_binding: RomRuntimeBinding | None = None
    if args.rom_path is not None and not uses_local_rom_cache:
        raise ValueError(
            f"--rom is not valid for ROM-free provider {provider.provider_id!r}"
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

    recipe_document = build_recipe_document(
        document,
        repo_root=source.repository_root,
        source_commit=source_commit,
        source_distribution=_source_distribution(),
        run_description=description,
        seed=int(args.seed),
        runtime_packages=_runtime_packages(),
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
        "recipe_ref": source.reference,
        "goal_id": goal_id,
        "recipe_id": recipe_id,
        "recipe_sha256": canonical_json_sha256(recipe_document),
        "seed": int(args.seed),
        "started_at": started_at,
        "model": None,
    }
    _write_receipt(run_dir, receipt)
    print(
        f"local training-only run: recipe={source.reference} seed={args.seed} "
        f"output={run_dir}",
        flush=True,
    )
    print(
        "checkpoint evaluation is disabled; this run cannot establish promotion or acceptance",
        flush=True,
    )

    from gradlab.train import main as learner_main

    previous_internal = os.environ.get(INTERNAL_LEARNER_ENV)
    previous_rom_cache = os.environ.get(LOCAL_ROM_CACHE_ENV)
    os.environ[INTERNAL_LEARNER_ENV] = "1"
    if uses_local_rom_cache and runtime_rom_binding is None:
        os.environ[LOCAL_ROM_CACHE_ENV] = str(DEFAULT_LOCAL_ROM_CACHE)
    try:
        learner_args = ["--train-config-json", str(config_path)]
        result = (
            learner_main(learner_args, runtime_rom_binding=runtime_rom_binding)
            if runtime_rom_binding is not None
            else learner_main(learner_args)
        )
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
    if result != 0 or not model_path.is_file():
        raise RuntimeError(f"local training did not produce {model_path}")
    receipt.update(
        {
            "status": "completed",
            "completed_at": _utc_now(),
            "model": model_path.name,
        }
    )
    _write_receipt(run_dir, receipt)
    print(f"trained model: {model_path}", flush=True)
    distribution = _source_distribution()
    play_command = [
        "uvx",
        f"gradlab@{distribution['version']}",
        "play",
        "--recipe",
        source.reference,
        "--runs-dir",
        str(args.runs_dir.expanduser().resolve()),
    ]
    if runtime_rom_binding is not None:
        play_command.extend(("--rom", str(runtime_rom_binding.path)))
    print(f"play it: {shlex.join(play_command)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
