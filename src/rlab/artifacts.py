from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

from rlab.env import EnvConfig, resolve_env_config
from rlab.env_metadata import (
    PLAYBACK_ENV_ARG_KEYS,
    assert_metadata_runtime_versions,
    env_config_from_config_dict,
    env_config_from_metadata,
    training_metadata,
)
from rlab.file_utils import fsync_path
from rlab.policy_bundle import (
    CHECKPOINT_FILENAME,
    MODEL_FILENAME,
    build_model_document,
    load_model_document,
    load_policy_bundle_from_checkpoint,
    load_recipe_document,
    model_document_as_metadata,
    model_document_path,
    recipe_document_path,
    write_canonical_json,
)


def checkpoint_step(path: Path) -> int | None:
    match = re.search(r"_(\d+)_steps$", path.stem)
    return int(match.group(1)) if match is not None else None


def build_model_provenance(
    args: argparse.Namespace,
    config: EnvConfig,
    model_path: Path,
    kind: str,
    checkpoint_step_value: int | None = None,
    snapshot_curriculum_session: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    training = training_metadata(
        config,
        rom_asset_manifest=getattr(args, "rom_asset_manifest", None),
    )
    provenance = {
        "kind": kind,
        "filename": model_path.name,
        "run_name": getattr(args, "run_name", ""),
        "run_description": getattr(args, "run_description", ""),
        "attempt_id": getattr(args, "attempt_id", ""),
        "compute_target": getattr(args, "compute_target", ""),
        "dstack_task": getattr(args, "dstack_task", ""),
        "wandb_run_id": getattr(args, "wandb_run_id", ""),
        "wandb_project": getattr(args, "wandb_project", ""),
        "campaign_id": getattr(args, "campaign_id", ""),
        "game_family": getattr(args, "game_family", ""),
        "goal_slug": getattr(args, "goal_slug", ""),
        "goal_path": getattr(args, "goal_path", ""),
        "goal_sha256": getattr(args, "goal_sha256", ""),
        "goal_contract_sha256": getattr(args, "goal_contract_sha256", ""),
        "effective_goal_contract_sha256": getattr(
            args, "effective_goal_contract_sha256", ""
        ),
        "reward_program_kind": getattr(args, "reward_program_kind", ""),
        "reward_program_revision": getattr(args, "reward_program_revision", ""),
        "reward_shape": getattr(args, "reward_shape", ""),
        "reward_shape_sha256": getattr(args, "reward_shape_sha256", ""),
        "reward_shape_is_default": getattr(args, "reward_shape_is_default", False),
        "recipe_slug": getattr(args, "recipe_slug", ""),
        "recipe_path": getattr(args, "recipe_path", ""),
        "recipe_sha256": getattr(args, "recipe_sha256", ""),
        "runtime_image_ref": getattr(args, "runtime_image_ref", ""),
        "seed": getattr(args, "seed", None),
        "repo_git_commit": str(
            getattr(args, "source_sha", "") or getattr(args, "repo_git_commit", "") or ""
        ).strip(),
        "checkpoint_step": checkpoint_step(model_path)
        if checkpoint_step(model_path) is not None
        else checkpoint_step_value,
        "training_backend_id": str(
            getattr(args, "training_backend_id", "") or ""
        ).strip(),
        "training_backend_config_hash": str(
            getattr(args, "training_backend_config_hash", "") or ""
        ).strip(),
        "algorithm_id": str(getattr(args, "algorithm_id", "") or "").strip(),
        "model_class": str(getattr(args, "model_class", "") or "").strip(),
        "training_metadata": training,
    }
    preflight_sha256 = str(
        getattr(args, "snapshot_curriculum_preflight_sha256", "") or ""
    ).strip()
    if preflight_sha256:
        provenance["snapshot_curriculum_preflight_sha256"] = preflight_sha256
    if snapshot_curriculum_session is not None:
        provenance["snapshot_curriculum_session"] = deepcopy(
            dict(snapshot_curriculum_session)
        )
    return provenance


def write_policy_bundle_sidecars(
    model_path: Path,
    recipe_source: Path,
    metadata: Mapping[str, Any],
) -> tuple[Path, Path]:
    load_recipe_document(recipe_source)
    recipe_sidecar = recipe_document_path(model_path)
    model_sidecar = model_document_path(model_path)
    if model_sidecar.is_file() or recipe_sidecar.is_file():
        existing = load_policy_bundle_from_checkpoint(model_path)
        if existing is None:
            raise ValueError(f"incomplete versioned policy sidecars for {model_path}")
        if recipe_source.read_bytes() != recipe_sidecar.read_bytes():
            raise ValueError(
                f"canonical recipe changed after checkpoint creation: {model_path}"
            )
        return model_sidecar, recipe_sidecar
    shutil.copyfile(recipe_source, recipe_sidecar)
    write_canonical_json(
        model_sidecar,
        build_model_document(model_path, recipe_sidecar, metadata),
    )
    load_model_document(model_sidecar)
    return model_sidecar, recipe_sidecar


def install_model_bundle(
    model_path: Path,
    *,
    save_checkpoint: Callable[[Path], None],
    args: argparse.Namespace,
    config: EnvConfig,
    kind: str,
    checkpoint_step_value: int | None,
    snapshot_curriculum_session: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically install checkpoint bytes and their reproducible policy sidecars."""

    model_path.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=".checkpoint-staging-", dir=model_path.parent)
    )
    staged_checkpoint = staging_dir / model_path.name
    staged_paths: list[Path] = [staged_checkpoint]
    try:
        save_checkpoint(staged_checkpoint)
        if not staged_checkpoint.is_file():
            raise FileNotFoundError(
                f"checkpoint saver did not create {staged_checkpoint}"
            )
        fsync_path(staged_checkpoint)
        provenance = build_model_provenance(
            args,
            config,
            model_path,
            kind,
            checkpoint_step_value=checkpoint_step_value,
            snapshot_curriculum_session=snapshot_curriculum_session,
        )
        recipe_source = Path(str(getattr(args, "recipe_json_path", "") or ""))
        if not recipe_source.is_file():
            raise FileNotFoundError(
                "checkpoint creation requires args.recipe_json_path to name a "
                f"canonical recipe document: {recipe_source}"
            )
        staged_model_document = model_document_path(staged_checkpoint)
        staged_recipe = recipe_document_path(staged_checkpoint)
        staged_paths.extend((staged_model_document, staged_recipe))
        write_policy_bundle_sidecars(staged_checkpoint, recipe_source, provenance)
        if load_policy_bundle_from_checkpoint(staged_checkpoint) is None:
            raise ValueError(
                f"checkpoint bundle validation failed: {staged_checkpoint}"
            )
        for staged in staged_paths:
            fsync_path(staged)

        destinations: list[tuple[Path, Path]] = [
            (staged_recipe, recipe_document_path(model_path)),
            (staged_model_document, model_document_path(model_path)),
        ]
        if model_path.exists():
            expected = [(staged_checkpoint, model_path), *destinations]
            mismatches = [
                destination
                for staged, destination in expected
                if not destination.is_file()
                or staged.read_bytes() != destination.read_bytes()
            ]
            if mismatches:
                raise FileExistsError(
                    "checkpoint destination conflicts with an existing committed bundle: "
                    + ", ".join(str(path) for path in mismatches)
                )
            return model_path

        for staged, destination in destinations:
            os.replace(staged, destination)
            staged_paths.remove(staged)
        fsync_path(model_path.parent)
        os.replace(staged_checkpoint, model_path)
        staged_paths.remove(staged_checkpoint)
        fsync_path(model_path.parent)
        if load_policy_bundle_from_checkpoint(model_path) is None:
            raise ValueError(f"checkpoint bundle validation failed: {model_path}")
        return model_path
    finally:
        for staged in staged_paths:
            staged.unlink(missing_ok=True)
        shutil.rmtree(staging_dir, ignore_errors=True)


def load_model_metadata(model_path: Path) -> dict[str, Any]:
    bundle = load_policy_bundle_from_checkpoint(model_path)
    if bundle is None:
        versioned_path = model_document_path(model_path)
        if model_path.name == CHECKPOINT_FILENAME:
            versioned_path = model_path.with_name(MODEL_FILENAME)
        raise FileNotFoundError(
            f"{model_path} is missing its current policy bundle beginning at "
            f"{versioned_path}"
        )
    return model_document_as_metadata(bundle.model)


def playback_env_config(
    config: EnvConfig,
    *,
    respect_task_termination: bool = True,
) -> EnvConfig:
    if respect_task_termination:
        return config
    task = deepcopy(config.task)
    termination = dict(task.get("termination", {}))
    termination.update(failure=[], success=[], timeout=[], max_episode_steps=0)
    task["termination"] = termination
    events = dict(task.get("events", {}))
    events.pop("stalled", None)
    task["events"] = events
    return replace(config, task=task)


def load_playback_env_config(
    model_path: Path,
    *,
    respect_task_termination: bool = True,
) -> EnvConfig:
    metadata = load_model_metadata(model_path)
    assert_metadata_runtime_versions(metadata)
    saved_config = env_config_from_metadata(metadata)
    if not saved_config:
        raise SystemExit(
            f"{model_path} is missing playback metadata. Recreate or re-upload the "
            "checkpoint with current model metadata before using rlab play."
        )
    config = env_config_from_config_dict(saved_config)
    if config is None:
        raise SystemExit(
            f"{model_path} playback metadata does not contain an environment config"
        )
    return playback_env_config(
        resolve_env_config(config),
        respect_task_termination=respect_task_termination,
    )


def apply_config_defaults(
    args: argparse.Namespace,
    config: dict[str, Any],
    parser_defaults: dict[str, object],
    explicit_dests: set[str],
) -> None:
    for arg_name, config_keys in PLAYBACK_ENV_ARG_KEYS.items():
        if arg_name not in parser_defaults or not hasattr(args, arg_name):
            continue
        if arg_name in explicit_dests:
            continue
        current_value = getattr(args, arg_name)
        default_value = parser_defaults[arg_name]
        if current_value != default_value and current_value not in ("", None):
            continue
        for config_key in config_keys:
            if config_key in config and config[config_key] is not None:
                setattr(args, arg_name, config[config_key])
                break


def write_run_description(args: argparse.Namespace, run_dir: str) -> None:
    description = args.run_description.strip()
    Path(run_dir, "run_description.txt").write_text(
        f"{description}\n" if description else "",
        encoding="utf-8",
    )
