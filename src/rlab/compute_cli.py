from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from rlab.compute_targets import (
    instance_defaults,
    instance_label,
    load_instance_config,
    target_kind,
    target_name,
)
from rlab.modal_launch import modal_launch_summary, preflight_modal_manifest
from rlab.skypilot_cli import cmd_launch as cmd_skypilot_launch
from rlab.skypilot_cli import cmd_preflight as cmd_skypilot_preflight
from rlab.skypilot_launch import load_manifest, shell_join


def repo_root_from_args(args: argparse.Namespace) -> Path:
    return Path(args.repo_root).expanduser().resolve()


def cmd_targets(args: argparse.Namespace) -> int:
    repo_root = repo_root_from_args(args)
    instance_config = load_instance_config(repo_root, args.instances)
    instances = instance_config.get("instances", {})
    if not isinstance(instances, dict):
        print("error: instances config must contain an instances object")
        return 1
    headers = [
        "target",
        "label",
        "available",
        "kind",
        "infra",
        "accelerator",
        "workers",
        "env_threads",
    ]
    rows = []
    for name, raw in sorted(instances.items()):
        if not isinstance(raw, dict):
            continue
        instance = instance_defaults(instance_config, str(name))
        rows.append(
            {
                "target": str(name),
                "label": instance_label(instance),
                "available": "no" if instance.get("available") is False else "yes",
                "kind": target_kind(instance),
                "infra": str(instance.get("infra", "")),
                "accelerator": str(instance.get("accelerator", "")),
                "workers": str(instance.get("children", "")),
                "env_threads": str(instance.get("env_threads", "")),
            }
        )
    if not rows:
        print("No compute targets configured.")
        return 0
    widths = {
        header: max(len(header), *(len(row.get(header, "")) for row in rows))
        for header in headers
    }
    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print(" | ".join("-" * widths[header] for header in headers))
    for row in rows:
        print(" | ".join(row.get(header, "").ljust(widths[header]) for header in headers))
    return 0


def _target_kind_for_manifest(args: argparse.Namespace) -> str:
    repo_root = repo_root_from_args(args)
    manifest = load_manifest(args.manifest)
    instance_config = load_instance_config(repo_root, args.instances)
    instance = instance_defaults(instance_config, target_name(manifest, args.target))
    return target_kind(instance)


def cmd_preflight(args: argparse.Namespace) -> int:
    kind = _target_kind_for_manifest(args)
    if kind in {"", "skypilot"}:
        return cmd_skypilot_preflight(args)
    if kind == "modal":
        repo_root = repo_root_from_args(args)
        manifest = load_manifest(args.manifest)
        instance_config = load_instance_config(repo_root, args.instances)
        checks = preflight_modal_manifest(
            manifest,
            instance_config,
            repo_root,
            target_override=args.target,
        )
        for check in checks:
            print(f"{check.level}: {check.message}")
        return 1 if any(check.level == "error" for check in checks) else 0
    print(f"error: compute target kind {kind!r} is not supported by preflight")
    return 1


def cmd_launch(args: argparse.Namespace) -> int:
    kind = _target_kind_for_manifest(args)
    if kind in {"", "skypilot"}:
        return cmd_skypilot_launch(args)
    if kind == "modal":
        repo_root = repo_root_from_args(args)
        manifest = load_manifest(args.manifest)
        summary = modal_launch_summary(
            manifest,
            args.manifest,
            repo_root=repo_root,
            instances_path=args.instances,
            target_override=args.target,
        )
        print(f"manifest: {summary.manifest_path}")
        print(f"target: {summary.target} ({summary.label})")
        print(f"modal_shape: gpu={summary.gpu} cpu={summary.cpu:g} memory_mib={summary.memory_mib}")
        print(shell_join(summary.command))
        if not args.execute:
            print("dry_run: pass --execute to run modal launch")
            return 0
        return subprocess.run(summary.command, cwd=repo_root, check=False).returncode
    print(f"error: compute target kind {kind!r} is not supported by launch")
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Provider-neutral rlab compute launcher.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    targets = subparsers.add_parser("targets", help="List configured compute targets")
    targets.add_argument("--repo-root", default=".")
    targets.add_argument("--instances", type=Path)
    targets.set_defaults(func=cmd_targets)

    preflight = subparsers.add_parser("preflight", help="Check manifest launch readiness")
    preflight.add_argument("manifest", type=Path)
    preflight.add_argument("--repo-root", default=".")
    preflight.add_argument("--instances", type=Path)
    preflight.add_argument("--target", help="Named compute target from experiments/instances.json")
    preflight.set_defaults(func=cmd_preflight)

    launch = subparsers.add_parser("launch", help="Launch a manifest on the selected provider")
    launch.add_argument("manifest", type=Path)
    launch.add_argument("--repo-root", default=".")
    launch.add_argument("--instances", type=Path)
    launch.add_argument("--target", help="Named compute target from experiments/instances.json")
    launch.add_argument(
        "--output",
        type=Path,
        default=Path("sky_rlab_generated_4090.yaml"),
        help="Rendered SkyPilot YAML path when the target kind is skypilot.",
    )
    launch.add_argument("--execute", action="store_true", help="Actually run the provider launch")
    launch.add_argument(
        "--detach-run",
        action="store_true",
        help="For SkyPilot targets, submit the job and return without streaming remote logs.",
    )
    launch.add_argument(
        "--sparse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For SkyPilot targets, print sparse launch milestones.",
    )
    launch.add_argument("--log-output", type=Path, help="For SkyPilot targets, full launch log path")
    launch.add_argument(
        "--down-on-complete",
        action="store_true",
        help="For SkyPilot targets, clean up the cluster after the launch exits.",
    )
    launch.set_defaults(func=cmd_launch)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
