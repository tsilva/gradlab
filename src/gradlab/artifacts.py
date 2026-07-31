from __future__ import annotations

import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

from gradlab.env import EnvConfig
from gradlab.env_metadata import runtime_versions_metadata
from gradlab.file_utils import fsync_path
from gradlab.policy_bundle import (
    build_model_document,
    load_model_document,
    load_policy_bundle_from_checkpoint,
    load_recipe_document,
    model_document_path,
    recipe_document_path,
    write_canonical_json,
)


def checkpoint_step(path: Path) -> int | None:
    match = re.search(r"_(\d+)_steps$", path.stem)
    return int(match.group(1)) if match is not None else None


def build_model_provenance(
    train_config: Mapping[str, Any],
    config: EnvConfig,
    model_path: Path,
    kind: str,
    checkpoint_step_value: int | None = None,
    state_archive_summary: Mapping[str, Any] | None = None,
    action_contract: Mapping[str, Any] | None = None,
    policy_execution_contract: Mapping[str, Any] | None = None,
    checkpoint_source_path: Path | None = None,
) -> dict[str, Any]:
    del config
    if action_contract is not None:
        from gradlab.action_contract import validate_runtime_action_contract

        validate_runtime_action_contract(action_contract)
    if policy_execution_contract is not None:
        from gradlab.policy_execution import normalize_policy_execution_contract

        policy_execution_contract = normalize_policy_execution_contract(
            policy_execution_contract
        )
    provenance = {
        "kind": kind,
        "filename": model_path.name,
        "run_name": train_config.get("run_name", ""),
        "run_description": train_config.get("run_description", ""),
        "attempt_id": train_config.get("attempt_id", ""),
        "compute_target": train_config.get("compute_target", ""),
        "dstack_task": train_config.get("dstack_task", ""),
        "wandb_run_id": train_config.get("wandb_run_id", ""),
        "wandb_project": train_config.get("wandb_project", ""),
        "campaign_id": train_config.get("campaign_id", ""),
        "game_family": train_config.get("game_family", ""),
        "goal_slug": train_config.get("goal_slug", ""),
        "goal_path": train_config.get("goal_path", ""),
        "goal_sha256": train_config.get("goal_sha256", ""),
        "goal_contract_sha256": train_config.get("goal_contract_sha256", ""),
        "effective_goal_contract_sha256": train_config.get("effective_goal_contract_sha256", ""),
        "reward_program_kind": train_config.get("reward_program_kind", ""),
        "reward_program_revision": train_config.get("reward_program_revision", ""),
        "reward_shape": train_config.get("reward_shape", ""),
        "reward_shape_sha256": train_config.get("reward_shape_sha256", ""),
        "reward_shape_is_default": train_config.get("reward_shape_is_default", False),
        "recipe_slug": train_config.get("recipe_slug", ""),
        "recipe_path": train_config.get("recipe_path", ""),
        "recipe_sha256": train_config.get("recipe_sha256", ""),
        "runtime_image_ref": train_config.get("runtime_image_ref", ""),
        "seed": train_config.get("seed"),
        "repo_git_commit": str(train_config.get("source_sha") or "").strip(),
        "checkpoint_step": checkpoint_step(model_path)
        if checkpoint_step(model_path) is not None
        else checkpoint_step_value,
        "training_backend_id": str(train_config.get("training_backend_id") or "").strip(),
        "training_backend_config_hash": str(
            train_config.get("training_backend_config_hash") or ""
        ).strip(),
        "algorithm_id": str(train_config.get("algorithm_id") or "").strip(),
        "search_algorithm_id": str(train_config.get("search_algorithm_id") or "").strip(),
        "model_class": str(train_config.get("model_class") or "").strip(),
        "training_execution": deepcopy(dict(train_config.get("training_execution") or {})),
        "training_terminal": deepcopy(dict(train_config.get("training_terminal") or {})),
        "training_metadata": {
            "versions": runtime_versions_metadata(),
            **(
                {"action_contract": deepcopy(dict(action_contract))}
                if action_contract is not None
                else {}
            ),
            **(
                {
                    "policy_execution_contract": deepcopy(
                        dict(policy_execution_contract)
                    )
                }
                if policy_execution_contract is not None
                else {}
            ),
        },
    }
    preflight_sha256 = str(train_config.get("state_archive_preflight_sha256") or "").strip()
    if preflight_sha256:
        provenance["state_archive_preflight_sha256"] = preflight_sha256
    if state_archive_summary is not None:
        provenance["state_archive_summary"] = deepcopy(dict(state_archive_summary))
    if str(train_config.get("algorithm_id") or "").strip() == "cell-graph":
        from gradlab.cell_graph import CellGraphPolicy

        graph = CellGraphPolicy.load(checkpoint_source_path or model_path).payload()
        provenance["cell_graph"] = {
            "detector_sha256": graph["detector_sha256"],
            "snapshot_mode": graph["snapshot_mode"],
            "summary": deepcopy(dict(graph["summary"])),
        }
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
            raise ValueError(f"canonical recipe changed after checkpoint creation: {model_path}")
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
    train_config: Mapping[str, Any],
    config: EnvConfig,
    kind: str,
    checkpoint_step_value: int | None,
    state_archive_summary: Mapping[str, Any] | None = None,
    action_contract: Mapping[str, Any] | None = None,
    policy_execution_contract: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically install checkpoint bytes and their reproducible policy sidecars."""

    model_path.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=".checkpoint-staging-", dir=model_path.parent))
    staged_checkpoint = staging_dir / model_path.name
    staged_paths: list[Path] = [staged_checkpoint]
    try:
        save_checkpoint(staged_checkpoint)
        if not staged_checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint saver did not create {staged_checkpoint}")
        fsync_path(staged_checkpoint)
        provenance = build_model_provenance(
            train_config,
            config,
            model_path,
            kind,
            checkpoint_step_value=checkpoint_step_value,
            state_archive_summary=state_archive_summary,
            action_contract=action_contract,
            policy_execution_contract=policy_execution_contract,
            checkpoint_source_path=staged_checkpoint,
        )
        recipe_source = Path(str(train_config.get("recipe_json_path") or ""))
        if not recipe_source.is_file():
            raise FileNotFoundError(
                "checkpoint creation requires train_config.recipe_json_path to name a "
                f"canonical recipe document: {recipe_source}"
            )
        staged_model_document = model_document_path(staged_checkpoint)
        staged_recipe = recipe_document_path(staged_checkpoint)
        staged_paths.extend((staged_model_document, staged_recipe))
        write_policy_bundle_sidecars(staged_checkpoint, recipe_source, provenance)
        if load_policy_bundle_from_checkpoint(staged_checkpoint) is None:
            raise ValueError(f"checkpoint bundle validation failed: {staged_checkpoint}")
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
                if not destination.is_file() or staged.read_bytes() != destination.read_bytes()
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


def write_run_description(train_config: Mapping[str, Any], run_dir: str) -> None:
    description = str(train_config.get("run_description") or "").strip()
    Path(run_dir, "run_description.txt").write_text(
        f"{description}\n" if description else "",
        encoding="utf-8",
    )
