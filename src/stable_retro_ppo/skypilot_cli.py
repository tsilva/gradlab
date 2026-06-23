from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from stable_retro_ppo.skypilot_launch import (
    build_launch_command,
    build_runner_launch_command,
    cleanup_command,
    collect_results,
    ensure_skypilot_api,
    execute_launch,
    fetch_wandb_run_config,
    format_results_table,
    LaunchSummary,
    launch_summary,
    load_instance_config,
    load_manifest,
    load_runner_profile,
    manifest_from_wandb_config,
    merged_env,
    preflight_checks,
    preflight_runner_profile,
    render_runner_task_yaml,
    render_task_yaml,
    shell_join,
    write_launch_report,
    write_rendered_task,
    write_rendered_runner_task,
)


def repo_root_from_args(args: argparse.Namespace) -> Path:
    return Path(args.repo_root).expanduser().resolve()


def add_common_manifest_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("manifest", type=Path, help="Experiment manifest JSON file")
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root used for relative paths; defaults to the current directory.",
    )
    parser.add_argument(
        "--instances",
        type=Path,
        help="Machine-readable instance config; defaults to experiments/instances.json.",
    )


def add_common_runner_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("profile", type=Path, help="Runner profile JSON file")
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root used for relative paths; defaults to the current directory.",
    )
    parser.add_argument(
        "--instances",
        type=Path,
        help="Machine-readable instance config; defaults to experiments/instances.json.",
    )


def cmd_render(args: argparse.Namespace) -> int:
    repo_root = repo_root_from_args(args)
    manifest = load_manifest(args.manifest)
    instance_config = load_instance_config(repo_root, args.instances)
    if args.output:
        path = write_rendered_task(manifest, instance_config, repo_root, args.output)
        print(path)
    else:
        print(render_task_yaml(manifest, instance_config, repo_root), end="")
    return 0


def cmd_render_runner(args: argparse.Namespace) -> int:
    repo_root = repo_root_from_args(args)
    profile = load_runner_profile(args.profile)
    instance_config = load_instance_config(repo_root, args.instances)
    if args.output:
        path = write_rendered_runner_task(profile, instance_config, repo_root, args.output)
        print(path)
    else:
        print(render_runner_task_yaml(profile, instance_config, repo_root), end="")
    return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    repo_root = repo_root_from_args(args)
    manifest = load_manifest(args.manifest)
    instance_config = load_instance_config(repo_root, args.instances)
    checks = preflight_checks(
        manifest,
        instance_config,
        repo_root,
        env=merged_env(repo_root / ".env"),
    )
    for check in checks:
        print(f"{check.level}: {check.message}")
    return 1 if any(check.level == "error" for check in checks) else 0


def cmd_preflight_runner(args: argparse.Namespace) -> int:
    repo_root = repo_root_from_args(args)
    profile = load_runner_profile(args.profile)
    instance_config = load_instance_config(repo_root, args.instances)
    checks = preflight_runner_profile(
        profile,
        instance_config,
        repo_root,
        env=merged_env(repo_root / ".env"),
    )
    for check in checks:
        print(f"{check.level}: {check.message}")
    return 1 if any(check.level == "error" for check in checks) else 0


def cmd_launch(args: argparse.Namespace) -> int:
    repo_root = repo_root_from_args(args)
    summary = launch_summary(
        args.manifest,
        args.output,
        repo_root,
        args.instances,
        detach_run=args.detach_run,
    )
    print(f"task: {summary.task_path}")
    print(f"cluster: {summary.cluster}")
    print(f"wandb_group_prefix: {summary.wandb_group_prefix}")
    print(shell_join(summary.command))
    if not args.execute:
        print("dry_run: pass --execute to run sky launch")
        return 0
    return execute_launch(
        summary,
        repo_root,
        repo_root / ".env",
        sparse=args.sparse,
        log_path=args.log_output,
        down_on_complete=args.down_on_complete,
    )


def cmd_launch_runner(args: argparse.Namespace) -> int:
    repo_root = repo_root_from_args(args)
    profile = load_runner_profile(args.profile)
    instance_config = load_instance_config(repo_root, args.instances)
    task_path = write_rendered_runner_task(profile, instance_config, repo_root, args.output)
    cluster = str(profile.get("cluster", profile.get("name", "stable-retro-ppo-runner-4090")))
    env = merged_env(repo_root / ".env")
    command = build_runner_launch_command(
        cluster,
        task_path,
        env=env,
        detach_run=args.detach_run,
    )
    summary = LaunchSummary(
        command=command,
        task_path=task_path,
        cluster=cluster,
        wandb_group_prefix=str(profile["profile_id"]),
    )
    print(f"task: {summary.task_path}")
    print(f"cluster: {summary.cluster}")
    print(f"profile: {profile['profile_id']}")
    print(shell_join(summary.command))
    if not args.execute:
        print("dry_run: pass --execute to run sky launch")
        return 0
    return execute_launch(
        summary,
        repo_root,
        repo_root / ".env",
        sparse=args.sparse,
        log_path=args.log_output,
        down_on_complete=args.down_on_complete,
    )


def cmd_command(args: argparse.Namespace) -> int:
    cmd = build_launch_command(args.cluster, args.task)
    print(shell_join(cmd))
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    rows = collect_results(args.log_dir, args.runs_dir)
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        print(format_results_table(rows))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    output = write_launch_report(args.log_path, args.output)
    print(output)
    return 0


def cmd_cleanup(args: argparse.Namespace) -> int:
    cmd = cleanup_command(args.cluster)
    print(shell_join(cmd))
    print(
        "known_cancel_workaround: if SkyPilot cancel/down fails with PermissionError, "
        "use sky exec to find the training process group, kill -TERM the group, "
        "verify no trainer remains, then rerun sky down."
    )
    if not args.execute:
        print("dry_run: pass --execute to run sky down")
        return 0
    return subprocess.run(cmd, cwd=repo_root_from_args(args), check=False).returncode


def cmd_doctor_api(args: argparse.Namespace) -> int:
    repo_root = repo_root_from_args(args)
    instance_config = load_instance_config(repo_root, args.instances)
    checks, command = ensure_skypilot_api(
        instance_config,
        repo_root=repo_root,
        instance_name=args.instance,
        execute=args.execute,
    )
    for check in checks:
        level = "ok" if check.ok else "error"
        print(f"{level}: {check.endpoint} {check.message}")
    if command is None:
        print("error: no healthy SkyPilot API endpoint found")
        return 1
    print(shell_join(command))
    if not args.execute:
        print("dry_run: pass --execute to run sky api login")
    return 0


def cmd_repro_from_wandb(args: argparse.Namespace) -> int:
    repo_root = repo_root_from_args(args)
    instance_config = load_instance_config(repo_root, args.instances)
    config = fetch_wandb_run_config(args.run_ref)
    manifest = manifest_from_wandb_config(
        args.run_ref,
        config,
        str(args.rom_source),
        name=args.name,
        cluster=args.cluster,
        artifact_storage_uri=args.artifact_storage_uri,
    )
    if args.manifest_output:
        args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"manifest: {args.manifest_output}")

    checks = preflight_checks(manifest, instance_config, repo_root, env=merged_env(repo_root / ".env"))
    for check in checks:
        print(f"{check.level}: {check.message}")
    if any(check.level == "error" for check in checks):
        return 1

    if args.ensure_api:
        api_checks, command = ensure_skypilot_api(
            instance_config,
            repo_root=repo_root,
            instance_name="rtx4090",
            execute=args.execute,
        )
        for check in api_checks:
            level = "ok" if check.ok else "error"
            print(f"{level}: {check.endpoint} {check.message}")
        if command is None:
            return 1
        print(shell_join(command))

    task_path = write_rendered_task(manifest, instance_config, repo_root, args.output)
    cluster = str(manifest.get("cluster", manifest["name"]))
    summary = LaunchSummary(
        command=build_launch_command(cluster, task_path, detach_run=args.detach_run),
        task_path=task_path,
        cluster=cluster,
        wandb_group_prefix=str(manifest.get("wandb_group_prefix", manifest["name"])),
    )
    print(f"task: {task_path}")
    print(f"cluster: {cluster}")
    print(shell_join(summary.command))
    if not args.execute:
        print("dry_run: pass --execute to run sky launch")
        return 0
    return execute_launch(
        summary,
        repo_root,
        repo_root / ".env",
        sparse=args.sparse,
        log_path=args.log_output,
        down_on_complete=args.down_on_complete,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render, launch, and summarize Stable Retro PPO SkyPilot RTX4090 batches."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    render = subparsers.add_parser("render", help="Render a manifest to SkyPilot YAML")
    add_common_manifest_args(render)
    render.add_argument("--output", type=Path, help="Write rendered task YAML to this path")
    render.set_defaults(func=cmd_render)

    render_runner = subparsers.add_parser(
        "render-runner",
        help="Render a long-lived train-runner profile to SkyPilot YAML",
    )
    add_common_runner_args(render_runner)
    render_runner.add_argument("--output", type=Path, help="Write rendered task YAML to this path")
    render_runner.set_defaults(func=cmd_render_runner)

    preflight = subparsers.add_parser("preflight", help="Check manifest/env launch readiness")
    add_common_manifest_args(preflight)
    preflight.set_defaults(func=cmd_preflight)

    preflight_runner = subparsers.add_parser(
        "preflight-runner",
        help="Check runner profile/env launch readiness",
    )
    add_common_runner_args(preflight_runner)
    preflight_runner.set_defaults(func=cmd_preflight_runner)

    launch = subparsers.add_parser("launch", help="Render a task and optionally run sky launch")
    add_common_manifest_args(launch)
    launch.add_argument(
        "--output",
        type=Path,
        default=Path("sky_stable_retro_generated_4090.yaml"),
        help="Rendered SkyPilot YAML path.",
    )
    launch.add_argument("--execute", action="store_true", help="Actually run sky launch")
    launch.add_argument(
        "--detach-run",
        action="store_true",
        help="Submit the SkyPilot job and return without streaming remote logs.",
    )
    launch.add_argument(
        "--sparse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep a full local launch log but print only milestone lines while SkyPilot runs.",
    )
    launch.add_argument("--log-output", type=Path, help="Full SkyPilot launch log path for --sparse")
    launch.add_argument(
        "--down-on-complete",
        action="store_true",
        help="Run the standard sky down cleanup after the launch command exits.",
    )
    launch.set_defaults(func=cmd_launch)

    launch_runner = subparsers.add_parser(
        "launch-runner",
        help="Render a train-runner task and optionally run sky launch",
    )
    add_common_runner_args(launch_runner)
    launch_runner.add_argument(
        "--output",
        type=Path,
        default=Path("sky_train_runner_4090.yaml"),
        help="Rendered SkyPilot YAML path.",
    )
    launch_runner.add_argument("--execute", action="store_true", help="Actually run sky launch")
    launch_runner.add_argument(
        "--detach-run",
        action="store_true",
        help="Submit the SkyPilot job and return without streaming remote logs.",
    )
    launch_runner.add_argument(
        "--sparse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep a full local launch log but print only milestone lines while SkyPilot runs.",
    )
    launch_runner.add_argument("--log-output", type=Path, help="Full SkyPilot launch log path for --sparse")
    launch_runner.add_argument(
        "--down-on-complete",
        action="store_true",
        help="Run the standard sky down cleanup after the launch command exits.",
    )
    launch_runner.set_defaults(func=cmd_launch_runner)

    command = subparsers.add_parser("command", help="Print the standard sky launch command")
    command.add_argument("cluster")
    command.add_argument("task", type=Path)
    command.set_defaults(func=cmd_command)

    collect = subparsers.add_parser("collect", help="Summarize child logs and run markers")
    collect.add_argument("log_dir", type=Path)
    collect.add_argument("--runs-dir", type=Path, default=Path("runs"))
    collect.add_argument("--json", action="store_true", help="Emit JSON instead of a table")
    collect.set_defaults(func=cmd_collect)

    report = subparsers.add_parser("report", help="Write a JSON report from a full SkyPilot launch log")
    report.add_argument("log_path", type=Path)
    report.add_argument("--output", type=Path, default=Path("reports/skypilot_launch_report.json"))
    report.set_defaults(func=cmd_report)

    repro = subparsers.add_parser(
        "repro-from-wandb",
        help="Clone a W&B run config into a ROM-agnostic RTX4090 SkyPilot launch.",
    )
    repro.add_argument("run_ref", help="W&B run ref in entity/project/run_id form")
    repro.add_argument("--rom-source", required=True, type=Path, help="Local ROM file to mount")
    repro.add_argument("--repo-root", default=".")
    repro.add_argument("--instances", type=Path)
    repro.add_argument("--name", help="Experiment name; defaults to repro-<run-id>-4090")
    repro.add_argument("--cluster", help="SkyPilot cluster name")
    repro.add_argument(
        "--artifact-storage-uri",
        default="${CHECKPOINT_BUCKET_URI}",
        help="s3:// bucket/prefix or ${CHECKPOINT_BUCKET_URI}; training appends the game id.",
    )
    repro.add_argument(
        "--output",
        type=Path,
        default=Path("sky_repro_wandb_4090.yaml"),
        help="Rendered SkyPilot YAML path.",
    )
    repro.add_argument("--manifest-output", type=Path, help="Optional generated manifest JSON path")
    repro.add_argument("--ensure-api", action="store_true", help="Select and login to a healthy API endpoint")
    repro.add_argument("--execute", action="store_true", help="Actually run sky launch")
    repro.add_argument(
        "--detach-run",
        action="store_true",
        help="Submit the SkyPilot job and return without streaming remote logs.",
    )
    repro.add_argument(
        "--sparse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep a full local launch log but print only milestone lines while SkyPilot runs.",
    )
    repro.add_argument("--log-output", type=Path, help="Full SkyPilot launch log path for --sparse")
    repro.add_argument(
        "--down-on-complete",
        action="store_true",
        help="Run the standard sky down cleanup after the launch command exits.",
    )
    repro.set_defaults(func=cmd_repro_from_wandb)

    cleanup = subparsers.add_parser("cleanup", help="Print or run standard cleanup")
    cleanup.add_argument("cluster")
    cleanup.add_argument(
        "--repo-root",
        default=".",
        help="Repository root used as cwd if --execute is passed.",
    )
    cleanup.add_argument("--execute", action="store_true", help="Actually run sky down")
    cleanup.set_defaults(func=cmd_cleanup)

    doctor = subparsers.add_parser("doctor-api", help="Find and optionally select a healthy SkyPilot API endpoint")
    doctor.add_argument("--repo-root", default=".")
    doctor.add_argument("--instances", type=Path)
    doctor.add_argument("--instance", default="rtx4090")
    doctor.add_argument("--execute", action="store_true", help="Actually run sky api login")
    doctor.set_defaults(func=cmd_doctor_api)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
