from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlab.compute_targets import (
    ensure_modal_target,
    instance_defaults,
    instance_label,
    load_instance_config,
    target_name,
)
from rlab.skypilot_launch import Check, manifest_game, training_options


@dataclass(frozen=True)
class ModalLaunchSummary:
    command: list[str]
    target: str
    label: str
    manifest_path: Path
    gpu: str
    cpu: float
    memory_mib: int


def modal_cpu(instance: dict[str, Any]) -> float:
    value = instance.get("cpu", instance.get("cpus", 8.0))
    return float(str(value).rstrip("+"))


def modal_memory_mib(instance: dict[str, Any]) -> int:
    value = instance.get("memory_mib", instance.get("memory", 16384))
    return int(float(str(value).rstrip("+")))


def modal_gpu(instance: dict[str, Any]) -> str:
    return str(instance.get("modal_gpu") or instance.get("accelerator") or "T4")


def build_modal_manifest_command(
    manifest_path: Path,
    *,
    repo_root: Path,
    instances_path: Path | None,
    target: str,
) -> list[str]:
    command = [
        "modal",
        "run",
        "src/rlab/modal_app.py::launch_manifest",
        "--manifest-path",
        str(manifest_path),
        "--repo-root",
        str(repo_root),
        "--target",
        target,
    ]
    if instances_path is not None:
        command.extend(["--instances-path", str(instances_path)])
    return command


def modal_launch_summary(
    manifest: dict[str, Any],
    manifest_path: Path,
    *,
    repo_root: Path,
    instances_path: Path | None,
    target_override: str | None = None,
) -> ModalLaunchSummary:
    instance_config = load_instance_config(repo_root, instances_path)
    target = target_name(manifest, target_override)
    instance = instance_defaults(instance_config, target)
    ensure_modal_target(instance)
    return ModalLaunchSummary(
        command=build_modal_manifest_command(
            manifest_path,
            repo_root=repo_root,
            instances_path=instances_path,
            target=target,
        ),
        target=target,
        label=instance_label(instance),
        manifest_path=manifest_path,
        gpu=modal_gpu(instance),
        cpu=modal_cpu(instance),
        memory_mib=modal_memory_mib(instance),
    )


def preflight_modal_manifest(
    manifest: dict[str, Any],
    instance_config: dict[str, Any],
    repo_root: Path,
    *,
    target_override: str | None = None,
) -> list[Check]:
    checks: list[Check] = []
    target = target_name(manifest, target_override)
    try:
        instance = instance_defaults(instance_config, target)
        ensure_modal_target(instance)
    except ValueError as exc:
        return [Check("error", str(exc))]

    try:
        manifest_game(manifest)
    except ValueError as exc:
        checks.append(Check("error", str(exc)))

    runs = manifest.get("runs", [])
    if not isinstance(runs, list) or not runs:
        checks.append(Check("error", "manifest must define at least one run"))
    else:
        max_children = int(instance.get("max_children", instance.get("children", 1)))
        if len(runs) > max_children:
            checks.append(
                Check("warning", f"{len(runs)} runs exceeds {instance_label(instance)} default {max_children}")
            )
        for index, run in enumerate(runs):
            if not isinstance(run, dict):
                checks.append(Check("error", f"runs[{index}] must be an object"))
                continue
            options = training_options(manifest, run)
            if not str(options.get("run_description", "")).strip():
                checks.append(Check("error", f"runs[{index}] has an empty run_description"))

    rom_source = repo_root / str(manifest.get("rom_source", ""))
    if manifest.get("rom_source") and not rom_source.exists():
        checks.append(
            Check(
                "warning",
                "ROM source path does not exist locally; Modal can still run if ROMs "
                f"were uploaded to its volume: {rom_source}",
            )
        )
    if not checks:
        checks.append(Check("ok", "Modal preflight passed with no blocking errors"))
    elif not any(check.level == "error" for check in checks):
        checks.append(Check("ok", "Modal preflight passed with warnings"))
    return checks
