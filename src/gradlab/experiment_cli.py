from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any
from urllib.parse import quote

from gradlab.cli_parser import ExactArgumentParser
from gradlab.clock import parse_utc_datetime
from gradlab.dstack_backend import (
    DSTACK_PROJECT_ENV,
    LOCAL_FLEET_ENV,
    TERMINAL_DSTACK_STATUSES,
    ComputeRequest,
    DstackBackend,
    TaskRequest,
    DstackTask,
    resolve_dstack_project,
    resolve_local_fleet,
)
from gradlab.env_registry import resolve_env_provider
from gradlab.file_utils import file_sha256
from gradlab.goal_variants import goal_variant_catalog_contract
from gradlab.json_utils import canonical_json_text, json_safe
from gradlab.modal_eval_config import load_modal_eval_config
from gradlab.operator_credentials import (
    OperatorConfigurationError,
    OperatorEnvironmentReport,
)
from gradlab.operator_environment import load_repository_operator_environment
from gradlab.r2_store import R2Bucket, RunStorageConfig
from gradlab.policy_bundle import build_recipe_document, canonical_json_sha256
from gradlab.recipe_documents import (
    compose_resolved_train_documents,
    prepare_checkpoint_eval_mode,
)
from gradlab.recipe_variants import recipe_variant_id
from gradlab.rom_assets import (
    ROM_ASSET_IDENTITY_ALGORITHM,
    ROM_ASSET_PREFIX,
    ROM_ASSET_SCHEMA_VERSION,
    discover_rom_path,
    provider_rom_identity,
    validate_rom_asset_manifest,
)
from gradlab.run_authority import Lease, RunAuthority
from gradlab.run_contracts import (
    RUN_ID_PATTERN,
    RunManifest,
    TerminalReceipt,
    default_liveness_policy,
    new_attempt_id,
    new_run_id,
    utc_now,
)
from gradlab.runtime_refs import (
    DEFAULT_IMAGE_ARTIFACT,
    DEFAULT_IMAGE_WORKFLOW,
    DEFAULT_RUNTIME_READINESS_TIMEOUT_SECONDS,
    clean_git_source_sha,
    current_git_branch,
    runtime_release_from_args,
)
from gradlab.wandb_utils import (
    canonical_wandb_environment,
    wandb_entity_from_env,
)
from gradlab.wandb_publisher import WandbProjector, publish_terminal_summary
from gradlab.vizdoom_assets import (
    VIZDOOM_IWAD_OBJECT_PREFIX,
    bind_vizdoom_iwad_to_document,
    required_vizdoom_iwad_binding,
    validate_vizdoom_iwad_binding,
    verify_vizdoom_iwad_file,
)


DEFAULT_MAX_DURATION_SECONDS = 48 * 60 * 60
DEFAULT_ROM_MOUNT = "/var/lib/gradlab/rom-cache:/rom-cache"
QUIESCENCE_SECONDS = 30.0
COMMON_SECRET_ENV = (
    "WANDB_API_KEY",
    "WANDB_ENTITY",
    "GRADLAB_CONTROL_R2_URI",
    "GRADLAB_CONTROL_R2_ENDPOINT_URL",
    "GRADLAB_CONTROL_R2_REGION",
    "GRADLAB_CONTROL_R2_ACCESS_KEY_ID",
    "GRADLAB_CONTROL_R2_SECRET_ACCESS_KEY",
    "GRADLAB_EVAL_R2_URI",
    "GRADLAB_EVAL_R2_ENDPOINT_URL",
    "GRADLAB_EVAL_R2_REGION",
    "GRADLAB_EVAL_R2_ACCESS_KEY_ID",
    "GRADLAB_EVAL_R2_SECRET_ACCESS_KEY",
    "GRADLAB_MODELS_R2_URI",
    "GRADLAB_MODELS_R2_ENDPOINT_URL",
    "GRADLAB_MODELS_R2_REGION",
    "GRADLAB_MODELS_R2_ACCESS_KEY_ID",
    "GRADLAB_MODELS_R2_SECRET_ACCESS_KEY",
    "GRADLAB_MODELS_R2_PUBLIC_BASE_URL",
)
OPERATOR_DSTACK_ENV = (
    DSTACK_PROJECT_ENV,
    "DSTACK_SERVER_URL",
    "DSTACK_TOKEN",
)
OPERATOR_MODAL_ENV = (
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
)
RECONCILE_STOP_REASONS = (
    "learner_failure",
    "startup_timeout",
    "invalid_result",
    "exit_contract_mismatch",
    "teardown_timeout",
    "supervisor_startup_failure",
    "forced_cancel_before_drain",
)


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(arguments)} failed")
    return result.stdout.strip()


def repository_root() -> Path:
    return Path(_git(Path.cwd(), "rev-parse", "--show-toplevel")).resolve()


def _load_environment(root: Path) -> OperatorEnvironmentReport:
    return load_repository_operator_environment(root)


def _tracked_committed_path(root: Path, path: Path, *, label: str) -> Path:
    resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must be inside the repository: {resolved}") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} does not exist: {relative}")
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(relative)],
        cwd=root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if tracked.returncode:
        raise ValueError(f"{label} must be checked in: {relative}")
    changed = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", str(relative)],
        cwd=root,
        check=False,
    )
    if changed.returncode:
        raise ValueError(f"{label} has uncommitted changes: {relative}")
    return resolved


def _parse_duration(value: str | int | float) -> float:
    if isinstance(value, int | float):
        result = float(value)
    else:
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smhd]?)\s*", value)
        if match is None:
            raise argparse.ArgumentTypeError("duration must look like 30s, 10m, 2h, or 1d")
        scale = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[match.group(2)]
        result = float(match.group(1)) * scale
    if result <= 0:
        raise argparse.ArgumentTypeError("duration must be positive")
    return result


def _require_run_id(value: str) -> str:
    text = str(value).strip()
    if RUN_ID_PATTERN.fullmatch(text) is None:
        raise argparse.ArgumentTypeError("run id must match gradlab-<32 lowercase hex>")
    return text


def _require_sha256_arg(value: str) -> str:
    text = str(value).strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise argparse.ArgumentTypeError("evidence hash must be 64 lowercase hexadecimal digits")
    return text


def _storage(root: Path) -> tuple[RunStorageConfig, RunAuthority]:
    _load_environment(root)
    try:
        storage = RunStorageConfig.from_env()
    except ValueError as exc:
        raise OperatorConfigurationError(str(exc)) from exc
    return storage, RunAuthority(storage)


def _required_operator_environment(checkpoint_eval_backend: str) -> tuple[str, ...]:
    return (
        *OPERATOR_DSTACK_ENV,
        *COMMON_SECRET_ENV,
        *(OPERATOR_MODAL_ENV if str(checkpoint_eval_backend) == "modal" else ()),
    )


def _operator_preflight(
    root: Path,
    *,
    checkpoint_eval_backend: str,
    dstack_project: str | None = None,
    local_target: str | None = None,
) -> tuple[
    RunStorageConfig,
    RunAuthority,
    DstackBackend,
    dict[str, Any],
]:
    environment_report = _load_environment(root)
    required = _required_operator_environment(checkpoint_eval_backend)
    missing = [name for name in required if not str(os.environ.get(name) or "").strip()]
    if missing:
        raise OperatorConfigurationError(
            "operator environment is incomplete; missing "
            + ", ".join(sorted(missing))
            + f". Configure {environment_report.config_path} from "
            "ops/operator.example.toml or provide the values through the process environment"
        )
    truncated = [name for name in required if "…" in str(os.environ.get(name) or "")]
    if truncated:
        raise OperatorConfigurationError(
            f"{sorted(truncated)[0]} is visibly truncated; use the exact "
            "machine-readable value, not human-formatted command output"
        )
    try:
        storage = RunStorageConfig.from_env()
    except ValueError as exc:
        raise OperatorConfigurationError(str(exc)) from exc
    dstack_backend = DstackBackend(project=dstack_project)
    try:
        dstack_backend.preflight()
    except (RuntimeError, ValueError) as exc:
        raise OperatorConfigurationError(f"dstack preflight failed: {exc}") from exc
    scopes = {
        "control": storage.control,
        "evaluation": storage.evaluation,
        "models": storage.models,
    }
    for label, config in scopes.items():
        try:
            next(R2Bucket(config).iter_keys(""), None)
        except Exception as exc:
            raise OperatorConfigurationError(
                f"{label} R2 read preflight failed ({type(exc).__name__}); "
                f"verify the {label} endpoint, bucket, and credential pair"
            ) from exc
    sources = {name: environment_report.source_for(name, os.environ) for name in sorted(required)}
    configured_local_fleet = str(local_target or os.environ.get(LOCAL_FLEET_ENV) or "").strip()
    local_fleet_source = (
        "command-line"
        if str(local_target or "").strip()
        else environment_report.source_for(LOCAL_FLEET_ENV, os.environ)
    )
    report = {
        "status": "ready",
        "checkpoint_eval_backend": checkpoint_eval_backend,
        "operator_config": {
            "path": str(environment_report.config_path),
            "present": environment_report.config_present,
        },
        "resolved_sources": sources,
        "storage": {name: "readable" for name in scopes},
        "dstack": {
            "project": dstack_backend.project,
            "server": "authenticated",
        },
        "compute": {
            "local_fleet": configured_local_fleet or None,
            "source": local_fleet_source,
        },
        "wandb": {"entity": wandb_entity_from_env()},
        "modal": {
            "credentials": ("resolved" if checkpoint_eval_backend == "modal" else "not-required")
        },
    }
    return (
        storage,
        RunAuthority(storage),
        dstack_backend,
        report,
    )


def _stage_rom(
    authority: RunAuthority,
    *,
    env_provider: str,
    game: str,
    rom_path: Path | None,
) -> dict[str, Any] | None:
    if not resolve_env_provider(env_provider).requires_external_rom_asset:
        if rom_path is not None:
            raise ValueError(f"--rom-path is invalid for ROM-free provider {env_provider!r}")
        return None
    source = discover_rom_path(game, rom_path=rom_path)
    digest = file_sha256(source)
    key = f"{ROM_ASSET_PREFIX}/objects/sha256/{digest}/{source.name}"
    manifest = validate_rom_asset_manifest(
        {
            "schema_version": ROM_ASSET_SCHEMA_VERSION,
            "game": game,
            "filename": source.name,
            "size_bytes": source.stat().st_size,
            "sha256": digest,
            "object_uri": authority.evaluation.uri(key),
            "provider_rom_identity": provider_rom_identity(source),
            "provider_rom_identity_algorithm": ROM_ASSET_IDENTITY_ALGORITHM,
        },
        expected_game=game,
    )
    authority.evaluation.put_file(
        key,
        source,
        sha256=digest,
        content_type="application/octet-stream",
    )
    authority.evaluation.put_json(
        f"{ROM_ASSET_PREFIX}/manifests/{game}/{digest}.json",
        manifest,
        create_only=True,
    )
    return manifest


def _bind_vizdoom_iwad_for_launch(
    *,
    env_provider: str,
    rom_path: Path | None,
) -> dict[str, Any] | None:
    if env_provider not in {"gradoom", "vizdoom-turbo"}:
        return None
    return required_vizdoom_iwad_binding(rom_path)


def _stage_vizdoom_iwad(
    authority: RunAuthority,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = validate_vizdoom_iwad_binding(binding, verify_file=True)
    source = verify_vizdoom_iwad_file(Path(normalized["path"]), normalized)
    key = (
        f"{VIZDOOM_IWAD_OBJECT_PREFIX}/objects/sha256/"
        f"{normalized['sha256']}/{normalized['filename']}"
    )
    staged = {
        **normalized,
        "object_uri": authority.evaluation.uri(key),
    }
    authority.evaluation.put_file(
        key,
        source,
        sha256=str(normalized["sha256"]),
        content_type="application/octet-stream",
    )
    authority.evaluation.put_json(
        (f"{VIZDOOM_IWAD_OBJECT_PREFIX}/manifests/sha256/{normalized['sha256']}.json"),
        staged,
        create_only=True,
    )
    return validate_vizdoom_iwad_binding(staged)


def _compute(args: argparse.Namespace) -> ComputeRequest:
    target = (
        resolve_local_fleet(args.target)
        if str(args.compute) in {"auto", "local"}
        else (str(args.target).strip() or None)
    )
    request = ComputeRequest(
        kind=args.compute,
        target=target,
        max_price=args.max_price,
        max_cost_usd=args.max_cost_usd,
        allow_on_demand=bool(args.allow_on_demand),
        max_duration_seconds=int(args.max_duration),
    )
    request.validate()
    return request


def _manifest_dstack_project(compute: Mapping[str, Any]) -> str:
    return resolve_dstack_project(str(compute.get("dstack_project") or "") or None)


def _dstack_backend_for_compute(compute: Mapping[str, Any]) -> DstackBackend:
    return DstackBackend(project=_manifest_dstack_project(compute))


def _retry_compute_request(compute: Mapping[str, Any]) -> dict[str, Any]:
    request = dict(compute["request"])
    if (
        str(request.get("kind") or "") in {"auto", "local"}
        and not str(request.get("target") or "").strip()
    ):
        selected = compute.get("selected")
        selected_target = str(
            (selected.get("target") or "") if isinstance(selected, Mapping) else ""
        ).strip()
        if not selected_target:
            raise RuntimeError("current run has no recorded local fleet for retry")
        request["target"] = selected_target
    return request


def _task_name(run_id: str, attempt_id: str, *, initial: bool) -> str:
    if initial:
        return run_id
    digest = hashlib.sha256(f"dstack-attempt-v1:{run_id}:{attempt_id}".encode()).hexdigest()
    return f"gradlab-{digest[:32]}"


def _task_request(manifest: RunManifest, *, manifest_uri: str) -> TaskRequest:
    compute = ComputeRequest(
        **dict(manifest.compute.get("selected") or manifest.compute["request"])
    )
    local = compute.kind == "local"
    plain_env = (
        {"MODAL_ENVIRONMENT": str(manifest.modal["environment_name"])}
        if bool(manifest.modal["enabled"])
        else {}
    )
    fault_fixture = str(manifest.compute.get("supervision_fault_fixture") or "").strip()
    if fault_fixture:
        plain_env["GRADLAB_SUPERVISION_FAULT_FIXTURE"] = fault_fixture
    return TaskRequest(
        run_id=manifest.run_id,
        task_name=str(manifest.compute["dstack_task"]),
        image=manifest.image_digest,
        manifest_uri=manifest_uri,
        compute=compute,
        plain_env=plain_env,
        secret_env=(
            *COMMON_SECRET_ENV,
            *(
                (
                    "MODAL_TOKEN_ID",
                    "MODAL_TOKEN_SECRET",
                )
                if bool(manifest.modal["enabled"])
                else ()
            ),
        ),
        rom_mount=(
            DEFAULT_ROM_MOUNT
            if local
            and (
                isinstance(manifest.modal.get("rom_asset_manifest"), dict)
                or isinstance(manifest.modal.get("vizdoom_iwad_binding"), dict)
            )
            else None
        ),
    )


def _wandb_identity(
    document: dict[str, Any],
    run_id: str,
    *,
    goal_slug: str,
    recipe_slug: str,
    recipe_variant: str,
    seed: int,
) -> dict[str, Any]:
    config = dict(document["train_config"])
    project, family = canonical_wandb_environment(
        config.get("env_provider"),
        config.get("game"),
    )
    relative_goal = (
        goal_slug.removeprefix(f"{project}/") if goal_slug.startswith(f"{project}/") else goal_slug
    )
    display_goal = relative_goal.replace("/", "--")
    display_name = (
        f"{display_goal}__{recipe_slug}__s{int(seed)}__{run_id.removeprefix('gradlab-')[:8]}"
    )
    campaign_id = str(document.get("campaign_id") or "").strip()
    group = (
        f"campaign::{campaign_id}"
        if campaign_id
        else f"cohort::{goal_slug}::{recipe_slug}::{recipe_variant}"
    )
    entity = wandb_entity_from_env()
    return {
        "run_id": run_id,
        "entity": entity,
        "project": project,
        "display_name": display_name,
        "group": group,
        "game_family": family,
        "url": (
            f"https://wandb.ai/{quote(entity, safe='')}/"
            f"{quote(project, safe='')}/runs/{quote(run_id, safe='')}"
        ),
    }


def _bind_launch_contract(
    document: dict[str, Any],
    *,
    asset: dict[str, Any] | None,
    vizdoom_iwad: dict[str, Any] | None = None,
    checkpoint_eval_backend: str,
) -> dict[str, Any]:
    contract_document = dict(document)
    contract_config = dict(contract_document["train_config"])
    if asset is None:
        contract_config.pop("rom_asset_manifest", None)
    else:
        contract_config["rom_asset_manifest"] = asset
    contract_config["checkpoint_eval_backend"] = checkpoint_eval_backend
    contract_document["train_config"] = contract_config
    if vizdoom_iwad is not None:
        bind_vizdoom_iwad_to_document(contract_document, vizdoom_iwad)
    return contract_document


def _manifest_rom_asset(modal: Mapping[str, Any]) -> dict[str, Any] | None:
    asset = modal.get("rom_asset_manifest")
    if asset is None:
        return None
    if not isinstance(asset, Mapping):
        raise ValueError("manifest modal ROM asset must be an object or null")
    return dict(asset)


def _manifest_vizdoom_iwad(modal: Mapping[str, Any]) -> dict[str, Any] | None:
    binding = modal.get("vizdoom_iwad_binding")
    if binding is None:
        return None
    if not isinstance(binding, Mapping):
        raise ValueError("manifest modal ViZDoom IWAD binding must be an object or null")
    return validate_vizdoom_iwad_binding(binding)


def cmd_launch(args: argparse.Namespace) -> int:
    root = repository_root()
    source_sha = clean_git_source_sha(root)
    branch = current_git_branch(root)
    goal_path = _tracked_committed_path(root, args.goal_file, label="goal")
    recipe_path = _tracked_committed_path(root, args.recipe_file, label="recipe")
    recipe_overrides = tuple(str(value) for value in args.recipe_overrides)
    requested_checkpoint_eval_backend = args.checkpoint_eval_backend
    resolved_documents = compose_resolved_train_documents(
        goal_path,
        recipe_path,
        recipe_overrides=recipe_overrides,
        prepare_materialized=partial(
            prepare_checkpoint_eval_mode,
            checkpoint_eval_backend=requested_checkpoint_eval_backend,
        ),
        source_sha=source_sha,
    )
    document = resolved_documents.effective
    checkpoint_eval_backend = str(document["train_config"]["checkpoint_eval_backend"])
    config = dict(document["train_config"])
    env_provider = str(config["env_provider"])
    vizdoom_iwad = _bind_vizdoom_iwad_for_launch(
        env_provider=env_provider,
        rom_path=args.rom_path,
    )
    storage, authority, dstack_backend, _preflight_report = _operator_preflight(
        root,
        checkpoint_eval_backend=checkpoint_eval_backend,
        local_target=args.target,
    )
    compute = _compute(args)
    selected_compute, selected_offer = dstack_backend.select_compute(compute)
    release = runtime_release_from_args(
        args,
        repo_root=root,
        wait_for_modal=checkpoint_eval_backend == "modal",
    )
    if release.source_sha != source_sha:
        raise RuntimeError("runtime release source does not match committed HEAD")
    if vizdoom_iwad is not None:
        vizdoom_iwad = _stage_vizdoom_iwad(authority, vizdoom_iwad)
    asset = (
        None
        if vizdoom_iwad is not None
        else _stage_rom(
            authority,
            env_provider=env_provider,
            game=str(config["game"]),
            rom_path=args.rom_path,
        )
    )
    run_id = new_run_id()
    attempt_id = new_attempt_id()
    dstack_task = _task_name(run_id, attempt_id, initial=True)
    goal_slug = goal_path.parent.relative_to(root / "experiments" / "goals").as_posix()
    recipe_slug = recipe_path.stem
    goal_variant = dict(document["goal_variant"])
    variant_id = recipe_variant_id(
        recipe_slug=recipe_slug,
        source_sha=source_sha,
        recipe_overrides=recipe_overrides,
    )
    wandb = _wandb_identity(
        document,
        run_id,
        goal_slug=goal_slug,
        recipe_slug=recipe_slug,
        recipe_variant=variant_id,
        seed=int(args.seed),
    )
    modal_app = str(release.modal_app_name or "").strip()
    if checkpoint_eval_backend == "modal" and not modal_app:
        raise RuntimeError("exact-source runtime has no immutable Modal deployment receipt")
    modal_config = load_modal_eval_config(root / "experiments" / "modal_eval.yaml")
    contract_document = _bind_launch_contract(
        document,
        asset=asset,
        vizdoom_iwad=vizdoom_iwad,
        checkpoint_eval_backend=checkpoint_eval_backend,
    )
    base_contract_document = _bind_launch_contract(
        resolved_documents.base,
        asset=asset,
        vizdoom_iwad=vizdoom_iwad,
        checkpoint_eval_backend=checkpoint_eval_backend,
    )
    portable_recipe = build_recipe_document(
        contract_document,
        repo_root=root,
        source_commit=source_sha,
        run_description=str(args.run_description),
        seed=int(args.seed),
        runtime_image_ref=release.runtime_image_ref,
        base_materialized_recipe=base_contract_document,
        canonical_goal=resolved_documents.canonical_goal,
    )
    recipe_sha256 = canonical_json_sha256(portable_recipe)
    authority.put_recipe_document(
        portable_recipe,
        expected_sha256=recipe_sha256,
    )
    fault_fixture = str(getattr(args, "supervision_fault_fixture", "") or "").strip()
    manifest_compute = {
        "request": compute.as_manifest(),
        "selected": selected_compute.as_manifest(),
        "selected_offer": selected_offer,
        "dstack_project": dstack_backend.project,
        "dstack_task": dstack_task,
        "source_branch": branch,
        "runtime_workflow_run_id": release.workflow_run_id,
        "runtime_input_sha256": release.runtime_input_sha256,
        "runtime_build_source_sha": release.runtime_build_source_sha,
        "submission_key": str(args.submission_key or ""),
    }
    liveness = default_liveness_policy()
    if fault_fixture:
        if fault_fixture not in {
            "failed-result-live-process",
            "completed-result-hung-process",
        }:
            raise ValueError(f"unsupported supervision fault fixture: {fault_fixture}")
        manifest_compute["supervision_fault_fixture"] = fault_fixture
        liveness.update(
            {
                "startup_timeout_seconds": 30.0,
                "result_exit_grace_seconds": 2.0,
                "terminate_grace_seconds": 3.0,
                "kill_grace_seconds": 2.0,
                "failure_drain_timeout_seconds": 30.0,
            }
        )
    manifest = RunManifest(
        run_id=run_id,
        attempt_id=attempt_id,
        created_at=utc_now(),
        source_sha=source_sha,
        image_digest=release.runtime_image_ref,
        goal_slug=goal_slug,
        goal_sha256=str(document["train_config"]["effective_goal_contract_sha256"]),
        recipe_slug=recipe_slug,
        recipe_sha256=recipe_sha256,
        recipe_overrides=recipe_overrides,
        environment_sha256=str(document["environment_hash"]).removeprefix("sha256:"),
        seed=int(args.seed),
        run_description=str(args.run_description),
        compute=manifest_compute,
        wandb=wandb,
        modal={
            "enabled": checkpoint_eval_backend == "modal",
            "environment_name": modal_config.deployment.environment_name,
            "app_name": modal_app,
            "function_name": modal_config.deployment.function_name,
            "deployment_source_sha": source_sha,
            "rom_asset_manifest": asset,
            "vizdoom_iwad_binding": vizdoom_iwad,
        },
        storage=storage.manifest_locations(),
        goal_variant=goal_variant,
        liveness=liveness,
    )
    authority.create_manifest(manifest)
    manifest_uri = authority.control.uri(f"runs/{run_id}/manifest.json")
    task = dstack_backend.submit(_task_request(manifest, manifest_uri=manifest_uri))
    output = {
        "schema_version": 1,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "dstack": {"project": task.project, "task": task.name, "status": task.status},
        "compute": {
            "request": compute.as_manifest(),
            "selected": selected_compute.as_manifest(),
            "offer": selected_offer,
        },
        "source_sha": source_sha,
        "image_digest": release.runtime_image_ref,
        "runtime_input_sha256": release.runtime_input_sha256,
        "runtime_build_source_sha": release.runtime_build_source_sha,
        "goal_file": goal_path.relative_to(root).as_posix(),
        "recipe_file": recipe_path.relative_to(root).as_posix(),
        "goal_sha256": manifest.goal_sha256,
        "goal_variant_id": goal_variant["variant_id"],
        "goal_variant_label": goal_variant["label"],
        "recipe_sha256": manifest.recipe_sha256,
        "recipe_overrides": list(recipe_overrides),
        "seed": int(args.seed),
        "run_description": str(args.run_description),
        "submission_key": str(args.submission_key or ""),
        "checkpoint_eval_backend": checkpoint_eval_backend,
        "wandb_url": wandb["url"],
        "public_run_index_url": authority.models.public_url(f"runs/{run_id}/index.json"),
    }
    print(
        json.dumps(json_safe(output), sort_keys=True)
        if args.json
        else (
            f"run={run_id} task={task.name} compute={selected_compute.kind} "
            f"image={release.runtime_image_ref} wandb={wandb['url']} "
            f"index={output['public_run_index_url']}"
        )
    )
    return 0


def cmd_operator_preflight(args: argparse.Namespace) -> int:
    root = repository_root()
    _storage_config, _authority, _dstack_backend, report = _operator_preflight(
        root,
        checkpoint_eval_backend=str(args.checkpoint_eval_backend),
        local_target=args.target,
    )
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(
            "Operator preflight ready: "
            f"dstack={report['dstack']['server']} "
            f"R2={','.join(report['storage'])} "
            f"Modal={report['modal']['credentials']}"
        )
    return 0


def _catalog_rebuild_contract_failures(
    source_records: list[tuple[str, RunManifest, TerminalReceipt | None]],
) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    variant_contracts: dict[tuple[str, str], dict[str, Any]] = {}
    for key, manifest, _terminal in source_records:
        descriptor = dict(manifest.goal_variant or {})
        identity = (manifest.goal_slug, str(descriptor.get("variant_id") or ""))
        contract = goal_variant_catalog_contract(descriptor)
        existing = variant_contracts.get(identity)
        if existing is not None and existing != contract:
            failures.append(
                {
                    "key": key,
                    "error_type": "ValueError",
                    "error": "goal variant descriptor conflicts with another current run",
                }
            )
        else:
            variant_contracts[identity] = contract
    return failures


def cmd_catalog_rebuild(args: argparse.Namespace) -> int:
    root = repository_root()
    _load_environment(root)
    storage = RunStorageConfig.from_env()
    authority = RunAuthority(storage)
    discovered = 0
    source_records: list[tuple[str, RunManifest, TerminalReceipt | None]] = []
    failed: list[dict[str, str]] = []
    for key in sorted(authority.control.iter_keys("runs/")):
        if not re.fullmatch(r"runs/gradlab-[0-9a-f]{32}/manifest\.json", key):
            continue
        discovered += 1
        try:
            state = authority.semantic_state(key.split("/")[1])
            manifest = RunManifest.from_dict(_latest_attempt(state))
            terminal_document = _latest_attempt_terminal(state)
            terminal = None
            if terminal_document is not None:
                terminal = TerminalReceipt.from_dict(terminal_document)
                if terminal.run_id != manifest.run_id or terminal.attempt_id != manifest.attempt_id:
                    raise ValueError("latest attempt terminal does not match its manifest")
            source_records.append((key, manifest, terminal))
        except Exception as exc:
            failed.append(
                {
                    "key": key,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    failed.extend(_catalog_rebuild_contract_failures(source_records))
    cleared = {"catalog_objects": 0, "projection_receipts": 0}
    rebuilt = 0
    published: dict[str, Any] | None = None
    if not failed:
        try:
            if args.check_only:
                goals = sorted({manifest.goal_slug for _key, manifest, _terminal in source_records})
                published = {
                    "schema_version": 1,
                    "check_only": True,
                    "goals": {
                        goal_slug: authority._goal_catalog_projector()
                        .reconcile(
                            goal_slug,
                            publish=False,
                        )
                        .to_dict()
                        for goal_slug in goals
                    },
                }
            else:
                published = authority.replace_goal_variant_catalog(
                    [(manifest, terminal) for _key, manifest, terminal in source_records]
                )
        except Exception as exc:
            failed.append(
                {
                    "key": "goal-catalog/v1",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        if published is not None:
            rebuilt = len(source_records)
    report = {
        "schema_version": 1,
        "discovered": discovered,
        "rebuilt": rebuilt,
        "cleared": cleared,
        "published": published,
        "check_only": bool(args.check_only),
        "failed": failed,
    }
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(
            "Goal-variant catalog rebuild: "
            f"{rebuilt} current runs indexed, "
            f"{'catalog checked' if args.check_only else 'per-goal immutable generations published' if published else 'no generation published'}, "
            f"{len(failed)} current records failed"
        )
    return 1 if failed else 0


def _latest_attempt(state: dict[str, Any]) -> dict[str, Any]:
    attempts = list(state.get("attempts") or [])
    if not attempts:
        raise KeyError(f"current run has no attempt manifest: {state['run_id']}")
    return dict(attempts[-1])


def _latest_attempt_terminal(state: dict[str, Any]) -> dict[str, Any] | None:
    attempt_id = str(_latest_attempt(state).get("attempt_id") or "")
    terminals = [
        dict(row)
        for row in state.get("attempt_terminals") or []
        if str(row.get("attempt_id") or "") == attempt_id
    ]
    return terminals[-1] if terminals else None


def _record_pre_submit_failure(
    authority: RunAuthority,
    manifest: RunManifest,
) -> None:
    prefix = f"{authority.run_prefix(manifest.run_id)}/attempts/{manifest.attempt_id}"
    keys = sorted(authority.control.iter_keys(prefix))
    expected = [f"{prefix}/manifest.json"]
    if keys != expected:
        raise RuntimeError("not-found dstack task has attempt activity beyond its manifest")
    authority.create_attempt_terminal(
        TerminalReceipt(
            run_id=manifest.run_id,
            attempt_id=manifest.attempt_id,
            state="resumable_failure",
            acceptance_required=bool(manifest.modal["enabled"]),
            stop_reason="pre_submit_failure",
            final_step=0,
            checkpoint_inventory=(),
            eval_inventory=(),
            wandb_high_water_mark=0,
            drain={
                "complete": False,
                "phase": "pre-submit",
                "metric_segment_high_water": 0,
                "eval_terminal_count": 0,
                "journal_archive": None,
                "journal_expires_at": None,
                "wandb_remote_high_water_mark": 0,
                "publication_capacity_ratio": None,
                "failure": "dstack task was not created",
            },
            completed_at=utc_now(),
        )
    )


def _record_terminal_task_without_receipt(
    authority: RunAuthority,
    manifest: RunManifest,
    task: DstackTask,
    *,
    writer_lease: Lease,
    stop_reason: str = "supervisor_startup_failure",
    final_step: int = 0,
    checkpoint_inventory: tuple[Mapping[str, Any], ...] = (),
    evidence_sha256: tuple[str, ...] = (),
) -> TerminalReceipt:
    if not task.terminal:
        raise RuntimeError("cannot seal an orphan attempt while its dstack task is active")
    if writer_lease.run_id != manifest.run_id or writer_lease.attempt_id != manifest.attempt_id:
        raise RuntimeError("orphan-attempt reconciliation requires its exclusive writer lease")
    receipt = TerminalReceipt(
        run_id=manifest.run_id,
        attempt_id=manifest.attempt_id,
        state="resumable_failure",
        acceptance_required=bool(manifest.modal["enabled"]),
        stop_reason=stop_reason,
        final_step=int(final_step),
        checkpoint_inventory=tuple(dict(row) for row in checkpoint_inventory),
        eval_inventory=(),
        wandb_high_water_mark=0,
        drain={
            "complete": False,
            "phase": "startup/recovery",
            "metric_segment_high_water": 0,
            "eval_terminal_count": 0,
            "journal_archive": None,
            "journal_expires_at": None,
            "wandb_remote_high_water_mark": 0,
            "publication_capacity_ratio": None,
            "recovered_checkpoint_count": len(checkpoint_inventory),
            "failure": (
                "dstack task reached terminal status "
                f"{task.status!r} without an authoritative attempt receipt"
            ),
            "evidence_sha256": list(evidence_sha256),
            "dstack": _public_dstack_state(task),
        },
        completed_at=utc_now(),
    )
    authority.create_attempt_terminal(receipt)
    return receipt


def _require_retryable_attempt_terminal(
    attempt_terminal: Mapping[str, Any] | None,
) -> None:
    if attempt_terminal is None:
        return
    state = str(attempt_terminal.get("state") or "")
    stop_reason = str(attempt_terminal.get("stop_reason") or "")
    if state == "succeeded":
        raise RuntimeError("a successfully drained training-only run must not be retried")
    if state == "stopped":
        raise RuntimeError(
            "a neutral plateau stop is non-resumable; launch a new recipe/run or "
            "evaluate a published checkpoint explicitly"
        )
    if state == "failed" and stop_reason.startswith("early_stop_failure:"):
        raise RuntimeError(
            "a designed early-stop failure is non-resumable; launch a new recipe/run"
        )


def _public_dstack_state(task: DstackTask) -> dict[str, Any]:
    raw = dict(task.raw or {})
    fleet = raw.get("fleet")
    fleet_name = str(fleet.get("name") or "") if isinstance(fleet, Mapping) else ""
    return {
        "project": task.project,
        "task": task.name,
        "status": task.status,
        "terminal": task.terminal,
        "fleet": fleet_name or None,
        "submitted_at": str(raw.get("submitted_at") or "") or None,
        "termination_reason": str(raw.get("termination_reason") or "") or None,
    }


def _run_completed(
    *,
    semantic_terminal: Mapping[str, Any] | None,
    attempt_terminal: Mapping[str, Any] | None,
    dstack_terminal: bool,
) -> bool:
    if attempt_terminal is None or not dstack_terminal:
        return False
    canonical_terminal_expected = (
        str(attempt_terminal.get("state") or "") == "succeeded"
        and attempt_terminal.get("acceptance_required") is True
    )
    return semantic_terminal is not None or not canonical_terminal_expected


def _status(root: Path, run_id: str) -> dict[str, Any]:
    _storage_config, authority = _storage(root)
    semantic = authority.semantic_state(run_id)
    attempt = _latest_attempt(semantic)
    task_name = str(attempt["compute"]["dstack_task"])
    dstack_backend = _dstack_backend_for_compute(attempt["compute"])
    try:
        dstack = dstack_backend.status(task_name)
        dstack_value = _public_dstack_state(dstack)
    except KeyError:
        dstack_value = {
            "project": dstack_backend.project,
            "task": task_name,
            "status": "not-found",
            "terminal": False,
        }
    attempt_terminal = _latest_attempt_terminal(semantic)
    return {
        "schema_version": 1,
        "run_id": run_id,
        "attempt_id": attempt["attempt_id"],
        "dstack": dstack_value,
        "semantic": semantic,
        "attempt_terminal": attempt_terminal,
        "completed": _run_completed(
            semantic_terminal=semantic.get("terminal"),
            attempt_terminal=attempt_terminal,
            dstack_terminal=bool(dstack_value["terminal"]),
        ),
        "scientific_success": semantic.get("terminal") is not None,
    }


def cmd_status(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            json_safe(_status(repository_root(), args.run_id)),
            indent=None if args.json else 2,
            sort_keys=True,
        )
    )
    return 0


def _follow_fingerprint(value: Mapping[str, Any]) -> str:
    stable = dict(value)
    semantic = dict(stable.get("semantic") or {})
    semantic.pop("observed_at", None)
    stable["semantic"] = semantic
    return canonical_json_text(json_safe(stable), ensure_ascii=True)


def _poll_status(
    root: Path,
    run_id: str,
    *,
    timeout: float,
    poll_seconds: float,
) -> Iterator[tuple[dict[str, Any], bool]]:
    deadline = time.monotonic() + timeout
    while True:
        value = _status(root, run_id)
        timed_out = time.monotonic() >= deadline
        yield value, timed_out
        if timed_out:
            return
        time.sleep(poll_seconds)


def cmd_follow(args: argparse.Namespace) -> int:
    root = repository_root()
    previous = ""
    for value, timed_out in _poll_status(
        root,
        args.run_id,
        timeout=float(args.timeout),
        poll_seconds=float(args.poll_seconds),
    ):
        encoded = canonical_json_text(json_safe(value), ensure_ascii=True)
        fingerprint = _follow_fingerprint(value)
        if fingerprint != previous:
            print(encoded, flush=True)
            previous = fingerprint
        if value["completed"]:
            return 0
        if timed_out:
            return 1
    raise AssertionError("status poller ended without completion or timeout")


def cmd_wait(args: argparse.Namespace) -> int:
    root = repository_root()
    for value, timed_out in _poll_status(
        root,
        args.run_id,
        timeout=float(args.timeout),
        poll_seconds=2.0,
    ):
        reached = (
            value["completed"]
            if args.until == "terminal"
            else str(value["dstack"]["status"]).lower() in {"running", "pulling"}
        )
        if reached:
            print(json.dumps(json_safe(value), sort_keys=True))
            return 0
        if timed_out:
            print(json.dumps(json_safe(value), sort_keys=True))
            return 1
    raise AssertionError("status poller ended without completion or timeout")


def cmd_cancel(args: argparse.Namespace) -> int:
    root = repository_root()
    _storage_config, authority = _storage(root)
    state = authority.semantic_state(args.run_id)
    attempt = _latest_attempt(state)
    if _latest_attempt_terminal(state) is not None:
        raise RuntimeError("cannot cancel an attempt that already has a terminal receipt")
    task_name = str(attempt["compute"]["dstack_task"])
    request = authority.request_cancel(
        run_id=args.run_id,
        attempt_id=str(attempt["attempt_id"]),
    )
    if args.abort:
        _dstack_backend_for_compute(attempt["compute"]).cancel(task_name, abort=True)
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "attempt_id": attempt["attempt_id"],
                "dstack_task": task_name,
                "cancel_requested": True,
                "cancel_requested_at": request["requested_at"],
                "abort": bool(args.abort),
                "dstack_cancel_sent": bool(args.abort),
            },
            sort_keys=True,
        )
    )
    return 0


def _project_reconciled_terminal(
    manifest: RunManifest,
    receipt: TerminalReceipt,
) -> None:
    projector = WandbProjector.resume(
        {
            "wandb_run_id": manifest.run_id,
            "wandb_entity": manifest.wandb["entity"],
            "wandb_project": manifest.wandb["project"],
            "wandb_mode": "online",
            "run_name": manifest.wandb.get("display_name"),
            "wandb_group": manifest.wandb.get("group"),
        },
        update_finish_state=True,
    )
    try:
        publish_terminal_summary(projector.run, receipt)
    finally:
        projector.close(
            timeout_seconds=300,
            exit_code=0 if receipt.state in {"succeeded", "stopped"} else 1,
        )


def cmd_reconcile(args: argparse.Namespace) -> int:
    root = repository_root()
    _storage_config, authority = _storage(root)
    state = authority.semantic_state(args.run_id)
    manifest = RunManifest.from_dict(_latest_attempt(state))
    task = _dstack_backend_for_compute(manifest.compute).status(
        str(manifest.compute["dstack_task"])
    )
    if not task.terminal:
        raise RuntimeError("cannot reconcile while the dstack task is active")
    lease = authority.acquire_lease(
        run_id=manifest.run_id,
        attempt_id=manifest.attempt_id,
        holder_id=f"operator-reconcile-{os.getpid()}",
    )
    created = False
    try:
        state = authority.semantic_state(args.run_id)
        existing = _latest_attempt_terminal(state)
        if existing is None:
            checkpoints = [
                dict(row)
                for row in ((state.get("public_index") or {}).get("checkpoints") or [])
                if isinstance(row, Mapping)
            ]
            final_step = max(
                (int(row.get("step") or 0) for row in checkpoints),
                default=0,
            )
            receipt = _record_terminal_task_without_receipt(
                authority,
                manifest,
                task,
                writer_lease=lease,
                stop_reason=str(args.stop_reason),
                final_step=final_step,
                checkpoint_inventory=tuple(checkpoints),
                evidence_sha256=tuple(args.evidence_sha256 or ()),
            )
            created = True
        else:
            receipt = TerminalReceipt.from_dict(existing)
            if receipt.attempt_id != manifest.attempt_id:
                raise RuntimeError("latest attempt terminal belongs to another attempt")
        _project_reconciled_terminal(manifest, receipt)
    finally:
        authority.release_lease(lease)
    output = {
        "run_id": manifest.run_id,
        "attempt_id": manifest.attempt_id,
        "dstack_task": task.name,
        "dstack_status": task.status,
        "terminal_created": created,
        "state": receipt.state,
        "stop_reason": receipt.stop_reason,
        "final_step": receipt.final_step,
        "wandb_projected": True,
    }
    print(
        json.dumps(output, sort_keys=True)
        if args.json
        else (
            f"reconciled run={manifest.run_id} attempt={manifest.attempt_id} "
            f"state={receipt.state} reason={receipt.stop_reason}"
        )
    )
    return 0


def cmd_fault_test(args: argparse.Namespace) -> int:
    launch_args = argparse.Namespace(
        goal_file=Path("experiments/goals/VizdoomBasic-v1/_goal.yaml"),
        recipe_file=Path("experiments/goals/VizdoomBasic-v1/recipes/ppo.yaml"),
        seed=17,
        run_description=(
            "Non-production bounded local-fleet learner-supervision fault fixture; "
            f"mode={args.mode}."
        ),
        recipe_overrides=[],
        checkpoint_eval_backend="none",
        submission_key="supervision-fault-fixture-v1",
        compute="local",
        target=args.target,
        max_price=None,
        max_cost_usd=None,
        allow_on_demand=False,
        max_duration=120,
        rom_path=None,
        runtime_image_ref_file=args.runtime_image_ref_file,
        image_workflow=args.image_workflow,
        image_artifact=args.image_artifact,
        image_branch=args.image_branch,
        existing_runtime_only=bool(args.existing_runtime_only),
        runtime_readiness_timeout=args.runtime_readiness_timeout,
        supervision_fault_fixture=str(args.mode),
        json=bool(args.json),
    )
    return cmd_launch(launch_args)


def cmd_logs(args: argparse.Namespace) -> int:
    root = repository_root()
    _storage_config, authority = _storage(root)
    attempt = _latest_attempt(authority.semantic_state(args.run_id))
    text = _dstack_backend_for_compute(attempt["compute"]).logs(
        str(attempt["compute"]["dstack_task"]),
        since=args.since,
    )
    lines = text.splitlines()
    if args.tail > 0:
        lines = lines[-args.tail :]
    print("\n".join(lines))
    return 0


def _manifest_only_submission(
    authority: RunAuthority,
    run_id: str,
) -> RunManifest:
    state = authority.semantic_state(run_id)
    manifest_document = state.get("manifest")
    attempts = list(state.get("attempts") or [])
    if not isinstance(manifest_document, dict) or len(attempts) != 1:
        raise RuntimeError("resume-submit requires exactly one canonical and one attempt manifest")
    if dict(attempts[0]) != manifest_document:
        raise RuntimeError("canonical and attempt manifests do not match")
    if any(
        (
            state.get("terminal") is not None,
            state.get("promotion") is not None,
            state.get("public_index") is not None,
            bool(state.get("attempt_terminals")),
            int(state.get("eval_intents") or 0) > 0,
            int(state.get("eval_results") or 0) > 0,
            int(state.get("verified_eval_results") or 0) > 0,
        )
    ):
        raise RuntimeError("run has progressed beyond a manifest-only launch")
    manifest = RunManifest.from_dict(manifest_document)
    prefix = authority.run_prefix(run_id)
    allowed_control_keys = {
        f"{prefix}/manifest.json",
        f"{prefix}/attempts/{manifest.attempt_id}/manifest.json",
    }
    unexpected_control_keys = sorted(
        set(authority.control.iter_keys(prefix)) - allowed_control_keys
    )
    if unexpected_control_keys:
        raise RuntimeError(
            "run has control state beyond its manifests: " + ", ".join(unexpected_control_keys)
        )
    if next(authority.evaluation.iter_keys(prefix), None) is not None:
        raise RuntimeError("run has evaluation state and cannot resume submission")
    if next(authority.models.iter_keys(prefix), None) is not None:
        raise RuntimeError("run has public model state and cannot resume submission")
    created_at = parse_utc_datetime(str(manifest.created_at))
    quiet_for = (datetime.now(UTC) - created_at).total_seconds()
    if quiet_for < QUIESCENCE_SECONDS:
        raise RuntimeError("manifest-only launch has not reached the 30-second quiescence interval")
    return manifest


def cmd_resume_submit(args: argparse.Namespace) -> int:
    root = repository_root()
    storage, authority = _storage(root)
    manifest = _manifest_only_submission(authority, args.run_id)
    checkpoint_eval_backend = "modal" if bool(manifest.modal["enabled"]) else "none"
    _storage_config, _authority, dstack_backend, _report = _operator_preflight(
        root,
        checkpoint_eval_backend=checkpoint_eval_backend,
        dstack_project=_manifest_dstack_project(manifest.compute),
    )
    task_name = str(manifest.compute["dstack_task"])
    try:
        existing = dstack_backend.status(task_name)
    except KeyError:
        existing = None
    if existing is not None:
        raise RuntimeError(f"dstack task already exists with status {existing.status}: {task_name}")
    manifest_uri = authority.control.uri(f"runs/{args.run_id}/manifest.json")
    task = dstack_backend.submit(_task_request(manifest, manifest_uri=manifest_uri))
    output = {
        "schema_version": 1,
        "run_id": manifest.run_id,
        "attempt_id": manifest.attempt_id,
        "dstack": {
            "project": task.project,
            "task": task.name,
            "status": task.status,
        },
        "compute": {
            "selected": dict(manifest.compute["selected"]),
            "offer": manifest.compute.get("selected_offer"),
        },
        "source_sha": manifest.source_sha,
        "image_digest": manifest.image_digest,
        "runtime_input_sha256": manifest.compute["runtime_input_sha256"],
        "runtime_build_source_sha": manifest.compute["runtime_build_source_sha"],
        "wandb_url": manifest.wandb["url"],
        "public_run_index_url": authority.models.public_url(f"runs/{manifest.run_id}/index.json"),
        "resumed_submission": True,
    }
    print(
        json.dumps(json_safe(output), sort_keys=True)
        if args.json
        else (
            f"run={manifest.run_id} task={task.name} "
            f"compute={manifest.compute['selected']['kind']} "
            f"image={manifest.image_digest} wandb={manifest.wandb['url']} "
            f"index={output['public_run_index_url']}"
        )
    )
    return 0


def _lease_expiry(authority: RunAuthority, run_id: str) -> datetime | None:
    value = authority.control.get_json_optional(f"runs/{run_id}/writer-lease.json")
    if value is None:
        return None
    return parse_utc_datetime(str(value["expires_at"]))


def cmd_retry(args: argparse.Namespace) -> int:
    root = repository_root()
    storage, authority = _storage(root)
    state = authority.semantic_state(args.run_id)
    if state.get("terminal") is not None:
        raise RuntimeError("a scientifically successful run must not be retried")
    previous = _latest_attempt(state)
    previous_manifest = RunManifest.from_dict(previous)
    attempt_terminal = _latest_attempt_terminal(state)
    dstack_backend = _dstack_backend_for_compute(previous["compute"])
    try:
        previous_task = dstack_backend.status(str(previous["compute"]["dstack_task"]))
    except KeyError:
        previous_task = None
    expiry = _lease_expiry(authority, args.run_id)
    if expiry is not None and expiry > datetime.now(UTC):
        raise RuntimeError(f"the previous writer lease has not expired: {expiry.isoformat()}")
    if previous_task is None and attempt_terminal is None:
        created_at = parse_utc_datetime(str(previous_manifest.created_at))
        if (datetime.now(UTC) - created_at).total_seconds() < QUIESCENCE_SECONDS:
            raise RuntimeError(
                "not-found attempt has not reached the 30-second quiescence interval"
            )
        _record_pre_submit_failure(authority, previous_manifest)
        state = authority.semantic_state(args.run_id)
        attempt_terminal = _latest_attempt_terminal(state)
    if previous_task is None and (
        attempt_terminal is None
        or str(attempt_terminal.get("stop_reason") or "") != "pre_submit_failure"
    ):
        raise RuntimeError(
            "the previous dstack task is not found without typed pre-submit evidence"
        )
    if previous_task is not None and previous_task.terminal and attempt_terminal is None:
        reconcile_lease = authority.acquire_lease(
            run_id=previous_manifest.run_id,
            attempt_id=previous_manifest.attempt_id,
            holder_id=f"operator-reconcile-{os.getpid()}",
        )
        try:
            _record_terminal_task_without_receipt(
                authority,
                previous_manifest,
                previous_task,
                writer_lease=reconcile_lease,
            )
        finally:
            authority.release_lease(reconcile_lease)
        state = authority.semantic_state(args.run_id)
        attempt_terminal = _latest_attempt_terminal(state)
    _require_retryable_attempt_terminal(attempt_terminal)
    if previous_task is not None and (
        previous_task.status.lower().replace("_", "-") not in TERMINAL_DSTACK_STATUSES
    ):
        raise RuntimeError("the previous dstack attempt must be terminal before retry")
    time.sleep(QUIESCENCE_SECONDS)
    attempt_id = new_attempt_id()
    task_name = _task_name(args.run_id, attempt_id, initial=False)
    compute = dict(previous["compute"])
    compute["dstack_task"] = task_name
    request_compute = _retry_compute_request(compute)
    compute["request"] = request_compute
    compute["dstack_project"] = dstack_backend.project
    selected_compute, selected_offer = dstack_backend.select_compute(
        ComputeRequest(**request_compute)
    )
    compute["selected"] = selected_compute.as_manifest()
    compute["selected_offer"] = selected_offer
    public_checkpoints = list((state.get("public_index") or {}).get("checkpoints") or [])
    learner_finished = any(
        str(row.get("purpose") or "") == "final"
        for row in public_checkpoints
        if isinstance(row, dict)
    )
    if learner_finished:
        compute["recovery_mode"] = "drain-only"
    else:
        compute["recovery_mode"] = "resume-training"
    manifest = replace(
        RunManifest.from_dict(previous),
        attempt_id=attempt_id,
        created_at=utc_now(),
        compute=compute,
    )
    if bool(getattr(args, "repair_runtime", False)):
        checkpoint_eval_backend = "modal" if bool(previous_manifest.modal["enabled"]) else "none"
        source_sha = clean_git_source_sha(root)
        branch = current_git_branch(root)
        release = runtime_release_from_args(
            args,
            repo_root=root,
            wait_for_modal=checkpoint_eval_backend == "modal",
        )
        if release.source_sha != source_sha:
            raise RuntimeError("repair runtime source does not match committed HEAD")
        goal_path = root / "experiments" / "goals" / manifest.goal_slug / "_goal.yaml"
        recipe_path = goal_path.parent / "recipes" / f"{manifest.recipe_slug}.yaml"
        repaired_documents = compose_resolved_train_documents(
            goal_path,
            recipe_path,
            recipe_overrides=manifest.recipe_overrides,
            prepare_materialized=partial(
                prepare_checkpoint_eval_mode,
                checkpoint_eval_backend=checkpoint_eval_backend,
            ),
            source_sha=source_sha,
        )
        document = repaired_documents.effective
        if str(document["train_config"]["effective_goal_contract_sha256"]) != manifest.goal_sha256:
            raise RuntimeError("repair runtime changed the effective goal contract")
        if str(document["environment_hash"]).removeprefix("sha256:") != manifest.environment_sha256:
            raise RuntimeError("repair runtime changed the environment contract")
        repaired_goal_variant = (
            dict(document["goal_variant"]) if manifest.goal_variant is not None else None
        )
        asset = _manifest_rom_asset(manifest.modal)
        vizdoom_iwad = _manifest_vizdoom_iwad(manifest.modal)
        contract_document = _bind_launch_contract(
            document,
            asset=asset,
            vizdoom_iwad=vizdoom_iwad,
            checkpoint_eval_backend=checkpoint_eval_backend,
        )
        base_contract_document = _bind_launch_contract(
            repaired_documents.base,
            asset=asset,
            vizdoom_iwad=vizdoom_iwad,
            checkpoint_eval_backend=checkpoint_eval_backend,
        )
        portable_recipe = build_recipe_document(
            contract_document,
            repo_root=root,
            source_commit=source_sha,
            run_description=manifest.run_description,
            seed=manifest.seed,
            runtime_image_ref=release.runtime_image_ref,
            base_materialized_recipe=base_contract_document,
            canonical_goal=repaired_documents.canonical_goal,
        )
        repaired_recipe_sha256 = canonical_json_sha256(portable_recipe)
        authority.put_recipe_document(
            portable_recipe,
            expected_sha256=repaired_recipe_sha256,
        )
        compute.update(
            {
                "source_branch": branch,
                "runtime_workflow_run_id": release.workflow_run_id,
                "runtime_input_sha256": release.runtime_input_sha256,
                "runtime_build_source_sha": release.runtime_build_source_sha,
            }
        )
        modal = {
            **manifest.modal,
            "app_name": str(release.modal_app_name or ""),
            "deployment_source_sha": source_sha,
        }
        manifest = replace(
            manifest,
            source_sha=source_sha,
            image_digest=release.runtime_image_ref,
            recipe_sha256=repaired_recipe_sha256,
            compute=compute,
            modal=modal,
            goal_variant=repaired_goal_variant,
        )
    manifest.validate()
    manifest_key = f"runs/{args.run_id}/attempts/{attempt_id}/manifest.json"
    manifest_uri = authority.control.uri(manifest_key)
    task_request = _task_request(manifest, manifest_uri=manifest_uri)
    task_request.validate()
    authority.create_attempt_manifest(manifest)
    authority.project_goal_variant_best_effort(manifest)
    try:
        task = dstack_backend.submit(task_request)
    except Exception:
        try:
            dstack_backend.status(task_name)
        except KeyError:
            _record_pre_submit_failure(authority, manifest)
        raise
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "attempt_id": attempt_id,
                "retried_from_attempt_id": previous["attempt_id"],
                "dstack_task": task.name,
                "recovery_mode": compute.get("recovery_mode", "resume-training"),
            },
            sort_keys=True,
        )
    )
    return 0


def cmd_certify(args: argparse.Namespace) -> int:
    from gradlab.lifecycle_certification import (
        DEFAULT_SCENARIOS,
        SCENARIOS,
        preserve_failure_bundle,
        replay_simulated_certification,
        run_simulated_certification,
    )
    from gradlab.local_paths import default_runs_dir

    if args.list_scenarios:
        for name in SCENARIOS:
            print(name)
        return 0
    selected = tuple(args.scenario or DEFAULT_SCENARIOS)
    if args.artifacts_dir is not None:
        artifact_root = args.artifacts_dir.resolve()
        if artifact_root.exists() and any(artifact_root.iterdir()):
            raise ValueError(f"--artifacts-dir must be empty: {artifact_root}")
        artifact_root.mkdir(parents=True, exist_ok=True)
        if args.replay is not None:
            report = replay_simulated_certification(
                args.replay,
                artifact_root=artifact_root,
            )
        else:
            report = run_simulated_certification(
                scenarios=selected,
                artifact_root=artifact_root,
            )
        failure_bundle: Path | None = artifact_root if report["status"] == "failed" else None
    else:
        with tempfile.TemporaryDirectory(prefix="gradlab-tier1-cli-") as temporary:
            artifact_root = Path(temporary)
            if args.replay is not None:
                report = replay_simulated_certification(
                    args.replay,
                    artifact_root=artifact_root,
                )
            else:
                report = run_simulated_certification(
                    scenarios=selected,
                    artifact_root=artifact_root,
                )
            failure_bundle = None
            if report["status"] == "failed":
                destination = (
                    default_runs_dir()
                    / "certification"
                    / f"failure-{str(report['report_sha256'])[:16]}"
                )
                if destination.exists():
                    failure_bundle = destination
                else:
                    failure_bundle = preserve_failure_bundle(
                        artifact_root,
                        destination,
                    )
    output = {
        "report": report,
        "failure_bundle": str(failure_bundle) if failure_bundle is not None else None,
    }
    if args.json:
        print(json.dumps(output, sort_keys=True))
    else:
        print(
            f"Tier 1 lifecycle certification: {report['status']} "
            f"({len(report['scenarios'])} scenarios, "
            f"report {report['report_sha256']})"
        )
        for scenario in report["scenarios"]:
            print(f"  {scenario['status']}: {scenario['name']}")
        if failure_bundle is not None:
            print(f"Failure evidence and replay: {failure_bundle}")
    return 0 if report["status"] == "passed" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = ExactArgumentParser(
        prog="gradlab experiment",
        description="Launch and observe dstack-backed training experiments.",
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    launch = commands.add_parser(
        "launch",
        help="Launch a checked-in goal and recipe with an exact-source runtime image.",
        description=(
            "Launch one checked-in goal and recipe through dstack. The command requires "
            "a clean committed source revision and resolves its exact-source immutable "
            "runtime image before scheduling compute; it never falls back to an older image."
        ),
    )
    launch.add_argument("--goal-file", type=Path, required=True)
    launch.add_argument("--recipe-file", type=Path, required=True)
    launch.add_argument("--seed", type=int, required=True)
    launch.add_argument("--run-description", required=True)
    launch.add_argument(
        "--set",
        dest="recipe_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Hash-bound recipe override; repeat for independent keys.",
    )
    launch.add_argument(
        "--checkpoint-eval-backend",
        choices=("modal", "none"),
        default=None,
        help=(
            "Override the recipe's checkpoint-evaluation mode. Modal establishes "
            "acceptance; none publishes training-only evidence without promotion or "
            "goal acceptance."
        ),
    )
    launch.add_argument(
        "--submission-key",
        help="Optional research-wave identity recorded in launch output.",
    )
    launch.add_argument(
        "--compute",
        choices=("auto", "local", "spot", "on-demand"),
        default="auto",
    )
    launch.add_argument("--target")
    launch.add_argument("--max-price", type=float)
    launch.add_argument("--max-cost-usd", type=float)
    launch.add_argument("--allow-on-demand", action="store_true")
    launch.add_argument(
        "--max-duration",
        type=_parse_duration,
        default=DEFAULT_MAX_DURATION_SECONDS,
    )
    launch.add_argument(
        "--rom-path",
        type=Path,
        help="Use a verified external ROM or local-fleet ViZDoom IWAD for this run.",
    )
    launch.add_argument("--runtime-image-ref-file", type=Path)
    launch.add_argument("--image-workflow", default=DEFAULT_IMAGE_WORKFLOW)
    launch.add_argument("--image-artifact", default=DEFAULT_IMAGE_ARTIFACT)
    launch.add_argument("--image-branch")
    launch.add_argument("--existing-runtime-only", action="store_true")
    launch.add_argument(
        "--runtime-readiness-timeout",
        type=_parse_duration,
        default=DEFAULT_RUNTIME_READINESS_TIMEOUT_SECONDS,
    )
    launch.add_argument("--json", action="store_true")
    launch.set_defaults(func=cmd_launch)

    operator_preflight = commands.add_parser(
        "operator-preflight",
        help="Validate local credentials and live service access before launch.",
        description=(
            "Resolve the private operator environment, authenticate dstack, and "
            "read-check all three R2 scopes without launching or mutating a run."
        ),
    )
    operator_preflight.add_argument(
        "--checkpoint-eval-backend",
        choices=("modal", "none"),
        default="modal",
    )
    operator_preflight.add_argument(
        "--target",
        help="Optional local fleet to report instead of the configured default.",
    )
    operator_preflight.add_argument("--json", action="store_true")
    operator_preflight.set_defaults(func=cmd_operator_preflight)

    catalog_rebuild = commands.add_parser(
        "catalog-rebuild",
        help="Check or rebuild disposable per-goal activity catalogs from durable evidence.",
    )
    catalog_rebuild.add_argument(
        "--check-only",
        action="store_true",
        help="Validate and compare without advancing any catalog pointer.",
    )
    catalog_rebuild.add_argument("--json", action="store_true")
    catalog_rebuild.set_defaults(func=cmd_catalog_rebuild)

    status = commands.add_parser("status", help="Inspect dstack and R2 run state.")
    status.add_argument("--run", dest="run_id", type=_require_run_id, required=True)
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=cmd_status)

    follow = commands.add_parser("follow", help="Stream changes in semantic run state.")
    follow.add_argument("--run", dest="run_id", type=_require_run_id, required=True)
    follow.add_argument("--poll-seconds", type=float, default=2.0)
    follow.add_argument("--timeout", type=_parse_duration, default=12 * 60 * 60)
    follow.set_defaults(func=cmd_follow)

    wait = commands.add_parser("wait", help="Wait for one run state.")
    wait.add_argument("--run", dest="run_id", type=_require_run_id, required=True)
    wait.add_argument("--until", choices=("running", "terminal"), required=True)
    wait.add_argument("--timeout", type=_parse_duration, default=12 * 60 * 60)
    wait.set_defaults(func=cmd_wait)

    cancel = commands.add_parser(
        "cancel",
        help="Request cooperative cancellation and terminal drain for the current attempt.",
    )
    cancel.add_argument("--run", dest="run_id", type=_require_run_id, required=True)
    cancel.add_argument(
        "--abort",
        action="store_true",
        help="Also stop dstack immediately; final artifacts and drain are not guaranteed.",
    )
    cancel.set_defaults(func=cmd_cancel)

    reconcile = commands.add_parser(
        "reconcile",
        help="Seal a terminal dstack attempt that has no authoritative receipt.",
        description=(
            "Acquire the attempt writer lease, create an idempotent operational-failure "
            "receipt for a terminal dstack task, then project that receipt to W&B."
        ),
    )
    reconcile.add_argument("--run", dest="run_id", type=_require_run_id, required=True)
    reconcile.add_argument(
        "--stop-reason",
        choices=RECONCILE_STOP_REASONS,
        default="supervisor_startup_failure",
    )
    reconcile.add_argument(
        "--evidence-sha256",
        action="append",
        type=_require_sha256_arg,
        default=[],
        help="Bounded external failure-evidence hash; repeat to record multiple objects.",
    )
    reconcile.add_argument("--json", action="store_true")
    reconcile.set_defaults(func=cmd_reconcile)

    fault_test = commands.add_parser(
        "fault-test",
        help="Launch the bounded non-production local-fleet learner-supervision fixture.",
        description=(
            "Launch an exact-source, two-minute local-fleet task that bypasses training and "
            "intentionally exercises failed-result or hung-result process-group teardown."
        ),
    )
    fault_test.add_argument(
        "--mode",
        choices=(
            "failed-result-live-process",
            "completed-result-hung-process",
        ),
        default="failed-result-live-process",
    )
    fault_test.add_argument(
        "--target",
        help="Local dstack fleet; defaults to GRADLAB_LOCAL_FLEET.",
    )
    fault_test.add_argument("--runtime-image-ref-file", type=Path)
    fault_test.add_argument("--image-workflow", default=DEFAULT_IMAGE_WORKFLOW)
    fault_test.add_argument("--image-artifact", default=DEFAULT_IMAGE_ARTIFACT)
    fault_test.add_argument("--image-branch")
    fault_test.add_argument("--existing-runtime-only", action="store_true")
    fault_test.add_argument(
        "--runtime-readiness-timeout",
        type=_parse_duration,
        default=DEFAULT_RUNTIME_READINESS_TIMEOUT_SECONDS,
    )
    fault_test.add_argument("--json", action="store_true")
    fault_test.set_defaults(func=cmd_fault_test)

    retry = commands.add_parser("retry", help="Retry a terminal failed attempt.")
    retry.add_argument("--run", dest="run_id", type=_require_run_id, required=True)
    retry.add_argument(
        "--repair-runtime",
        action="store_true",
        help=(
            "Use the exact-source runtime for committed HEAD while preserving the "
            "logical run, goal, environment, seed, and recipe overrides."
        ),
    )
    retry.set_defaults(func=cmd_retry)

    resume_submit = commands.add_parser(
        "resume-submit",
        help="Submit an immutable manifest-only launch after a pre-submit failure.",
        description=(
            "Recover one launch whose immutable R2 manifest exists but whose dstack "
            "task was never created. The command fails closed if any run activity "
            "or task with the bound name exists."
        ),
    )
    resume_submit.add_argument(
        "--run",
        dest="run_id",
        type=_require_run_id,
        required=True,
    )
    resume_submit.add_argument("--json", action="store_true")
    resume_submit.set_defaults(func=cmd_resume_submit)

    logs = commands.add_parser("logs", help="Read dstack logs for the current attempt.")
    logs.add_argument("--run", dest="run_id", type=_require_run_id, required=True)
    logs.add_argument("--tail", type=int, default=100)
    logs.add_argument("--since")
    logs.set_defaults(func=cmd_logs)

    certify = commands.add_parser(
        "certify",
        help="Run the credential-free deterministic orchestration lifecycle gate.",
    )
    certify.add_argument(
        "--tier",
        choices=("simulated",),
        default="simulated",
    )
    certify.add_argument(
        "--scenario",
        action="append",
        help="Run one named scenario; repeat to select multiple scenarios.",
    )
    certify.add_argument(
        "--list",
        dest="list_scenarios",
        action="store_true",
        help="List deterministic Tier 1 scenarios and exit.",
    )
    certify.add_argument(
        "--replay",
        type=Path,
        help="Replay the exact scenario set from a preserved replay.json.",
    )
    certify.add_argument(
        "--artifacts-dir",
        type=Path,
        help="Keep raw buckets, SQLite ledgers, transcripts, report, and replay here.",
    )
    certify.add_argument("--json", action="store_true")
    certify.set_defaults(func=cmd_certify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        return int(args.func(args))
    except OperatorConfigurationError as exc:
        print(f"gradlab experiment: operator configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
