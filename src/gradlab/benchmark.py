from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import UTC, datetime
from importlib import metadata
import json
import math
import os
import platform
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any

from gradlab.benchmark_profiles import (
    DEFAULT_PROFILE_DIR,
    DEFAULT_RESULT_DIR,
    BenchmarkCommand,
    BenchmarkProfile,
    build_benchmark_commands,
    find_benchmark_profile,
    load_benchmark_profiles,
    ppo_backend_command_contract,
)
from gradlab.cli_parser import ExactArgumentParser
from gradlab.env import default_run_dir
from gradlab.metric_store import MetricStore, metric_store_path
from gradlab.policy_bundle import build_recipe_document, write_canonical_json
from gradlab.recipe_documents import compose_resolved_train_documents, repo_git_commit


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True)


def _command_plan(commands: list[BenchmarkCommand]) -> list[dict[str, Any]]:
    return [command.to_json() for command in commands]


def _execution_commands(
    profile: BenchmarkProfile,
    commands: list[BenchmarkCommand],
    *,
    execution_id: str,
    output_dir: Path,
) -> list[BenchmarkCommand]:
    if profile.kind not in {
        "ppo_backend_comparison",
        "train_loop_throughput",
        "train_loop_comparison",
    }:
        return commands
    prepared: list[BenchmarkCommand] = []
    runs_dir = str(output_dir / "runs" / f"{execution_id}-{profile.name}")
    for command in commands:
        config = json.loads(command.stdin or "{}")
        config["runs_dir"] = runs_dir
        if profile.kind == "ppo_backend_comparison":
            goal_file, recipe_file, recipe_overrides = ppo_backend_command_contract(
                profile.payload,
                command.label,
            )
        elif profile.kind == "train_loop_throughput":
            goal_file = str(profile.payload["goal_file"])
            recipe_file = str(profile.payload["recipe_file"])
            recipe_overrides = tuple(profile.payload.get("recipe_overrides", ()))
        else:
            goal_file = str(profile.payload["goal_file"])
            variant = "candidate" if command.label.startswith("candidate-") else "baseline"
            recipe_file = str(profile.payload[f"{variant}_recipe_file"])
            recipe_overrides = tuple(profile.payload.get("recipe_overrides", ()))
        source_commit = repo_git_commit() or "0" * 40
        resolved = compose_resolved_train_documents(
            Path(goal_file),
            Path(recipe_file),
            recipe_overrides=recipe_overrides,
            source_sha=source_commit,
        )
        recipe_document = build_recipe_document(
            resolved.effective,
            repo_root=Path.cwd(),
            source_commit=source_commit,
            run_description=str(config["run_description"]),
            seed=int(config["seed"]),
            runtime_image_ref="docker:gradlab-benchmark@sha256:" + "0" * 64,
            base_materialized_recipe=resolved.base,
            canonical_goal=resolved.canonical_goal,
        )
        recipe_path = (
            output_dir
            / "runs"
            / f"{execution_id}-{profile.name}"
            / "contracts"
            / str(config["run_name"])
            / "recipe.json"
        ).resolve()
        write_canonical_json(recipe_path, recipe_document)
        config["recipe_json_path"] = str(recipe_path)
        prepared.append(replace(command, stdin=json.dumps(config, sort_keys=True)))
    return prepared


def _git_dirty() -> bool | None:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def _load_average() -> list[float] | None:
    try:
        return [round(float(value), 3) for value in os.getloadavg()]
    except OSError:
        return None


def _environment_snapshot(profile: BenchmarkProfile) -> dict[str, Any]:
    versions = {"python": platform.python_version()}
    distributions = {"gradlab", "stable-retro-turbo", "supermariobrosnes-turbo"}
    provider = str(profile.payload.get("env_provider") or "").strip()
    if provider:
        distributions.add(provider)
    for distribution in sorted(distributions):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    cuda_devices: list[str] = []
    try:
        import torch

        cuda_devices = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
    except (ImportError, RuntimeError):
        pass
    return {
        "captured_at": datetime.now(UTC).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "load_average_1m_5m_15m": _load_average(),
        "cuda_devices": cuda_devices,
        "source_commit": repo_git_commit(),
        "source_dirty": _git_dirty(),
        "versions": versions,
    }


def _stdout_json(result: dict[str, Any], *, label: str, issues: list[str]) -> Any:
    try:
        return json.loads(str(result.get("stdout") or ""))
    except json.JSONDecodeError as exc:
        issues.append(f"{label} did not emit valid JSON: {exc}")
        return None


def validate_benchmark_results(
    profile: BenchmarkProfile,
    commands: list[BenchmarkCommand],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    issues = [
        f"{result['label']} exited with {result['returncode']}"
        for result in results
        if int(result["returncode"]) != 0
    ]
    evidence: dict[str, Any] = {}

    if len(results) != len(commands):
        issues.append(f"executed {len(results)} of {len(commands)} benchmark commands")

    successful = [result for result in results if int(result["returncode"]) == 0]
    if profile.kind == "local_smoke" and successful:
        payload = _stdout_json(successful[0], label="local smoke", issues=issues)
        if isinstance(payload, dict):
            wait = payload.get("wait")
            jobs = payload.get("jobs")
            if not isinstance(wait, dict) or not wait.get("reached"):
                issues.append("local smoke did not reach terminal state")
            if not isinstance(jobs, list) or not jobs:
                issues.append("local smoke returned no jobs")
            else:
                statuses = [str(job.get("status")) for job in jobs if isinstance(job, dict)]
                evidence["job_statuses"] = statuses
                if len(statuses) != len(jobs) or statuses != ["succeeded"] * len(statuses):
                    issues.append(f"local smoke jobs were not all successful: {statuses}")

    if profile.kind == "env_throughput":
        samples = []
        for result in successful:
            payload = _stdout_json(result, label=str(result["label"]), issues=issues)
            if not isinstance(payload, dict):
                continue
            sample = {
                "label": result["label"],
                "envs": payload.get("envs"),
                "results": payload.get("results"),
                "runtime_overhead_fraction": payload.get("runtime_overhead_fraction"),
                "overhead_gate_passed": payload.get("overhead_gate_passed"),
            }
            samples.append(sample)
            if payload.get("mode") == "compare" and payload.get("overhead_gate_passed") is not True:
                issues.append(f"{result['label']} exceeded the runtime-overhead gate")
        evidence["samples"] = samples

    if profile.kind == "train_loop_throughput" and successful:
        config = json.loads(commands[0].stdin or "{}")
        base_dir = commands[0].cwd or Path.cwd()
        run_dir = base_dir / default_run_dir(
            str(config["run_name"]), str(config.get("runs_dir") or "runs")
        )
        store_path = metric_store_path(run_dir)
        evidence["metric_store"] = str(store_path)
        if not store_path.is_file():
            issues.append(f"training-loop metric store is missing: {store_path}")
        else:
            store = MetricStore(store_path)
            metrics = {
                name: store.latest_metric(name)
                for name in profile.payload.get("required_metrics", ())
            }
            evidence["metrics"] = metrics
            missing = sorted(name for name, value in metrics.items() if value is None)
            if missing:
                issues.append(f"training-loop metrics are missing: {', '.join(missing)}")

    if profile.kind == "train_loop_comparison" and successful:
        common_metrics = tuple(profile.payload.get("required_metrics", ()))
        candidate_metrics = tuple(profile.payload.get("candidate_required_metrics", ()))
        samples: list[dict[str, Any]] = []
        loop_rates: dict[str, list[float]] = {"baseline": [], "candidate": []}
        for command, result in zip(commands, results, strict=False):
            if int(result["returncode"]) != 0:
                continue
            variant = "candidate" if command.label.startswith("candidate-") else "baseline"
            config = json.loads(command.stdin or "{}")
            base_dir = command.cwd or Path.cwd()
            run_dir = base_dir / default_run_dir(
                str(config["run_name"]), str(config.get("runs_dir") or "runs")
            )
            store_path = metric_store_path(run_dir)
            required = common_metrics + (candidate_metrics if variant == "candidate" else ())
            if not store_path.is_file():
                issues.append(f"{command.label} metric store is missing: {store_path}")
                continue
            store = MetricStore(store_path)
            metrics = {name: store.latest_metric(name) for name in required}
            missing = sorted(name for name, value in metrics.items() if value is None)
            if missing:
                issues.append(f"{command.label} metrics are missing: {', '.join(missing)}")
            loop_value = metrics.get("train/throughput/loop/rate")
            if loop_value is not None:
                loop_rates[variant].append(float(loop_value))
            samples.append(
                {
                    "label": command.label,
                    "variant": variant,
                    "metric_store": str(store_path),
                    "metrics": metrics,
                }
            )
        evidence["samples"] = samples
        if not loop_rates["baseline"] or not loop_rates["candidate"]:
            issues.append("throughput comparison lacks baseline or candidate loop-rate samples")
        else:
            baseline_mean = sum(loop_rates["baseline"]) / len(loop_rates["baseline"])
            candidate_mean = sum(loop_rates["candidate"]) / len(loop_rates["candidate"])
            slowdown = 1.0 - candidate_mean / baseline_mean if baseline_mean > 0.0 else 1.0
            maximum = float(profile.payload["max_candidate_slowdown"])
            evidence["comparison"] = {
                "baseline_loop_rate_mean": baseline_mean,
                "candidate_loop_rate_mean": candidate_mean,
                "candidate_slowdown_fraction": slowdown,
                "max_candidate_slowdown": maximum,
            }
            if slowdown > maximum:
                issues.append(
                    "candidate training-loop slowdown exceeded gate: "
                    f"{slowdown:.6f} > {maximum:.6f}"
                )

    if profile.kind == "ppo_backend_comparison" and successful:
        required_metrics = tuple(profile.payload.get("required_metrics", ()))
        warmup = int(profile.payload["warmup_iterations"])
        run_samples: list[dict[str, Any]] = []
        paired: dict[str, dict[int, dict[str, float]]] = {}
        for command, result in zip(commands, results, strict=False):
            if int(result["returncode"]) != 0:
                continue
            goal_file, _recipe_file, _overrides = ppo_backend_command_contract(
                profile.payload,
                command.label,
            )
            case_name = next(
                str(case["name"])
                for case in profile.payload["cases"]
                if str(case["goal_file"]) == goal_file
                and command.label.startswith(
                    re.sub(
                        r"[^A-Za-z0-9_.-]+",
                        "-",
                        str(case["name"]).strip().lower(),
                    ).strip("-")
                    + "-"
                )
            )
            remainder = command.label.rsplit("-", 2)
            if len(remainder) != 3:
                issues.append(f"invalid PPO comparison command label: {command.label}")
                continue
            _prefix, variant, repeat_text = remainder
            repeat = int(repeat_text)
            config = json.loads(command.stdin or "{}")
            base_dir = command.cwd or Path.cwd()
            run_dir = base_dir / default_run_dir(
                str(config["run_name"]),
                str(config.get("runs_dir") or "runs"),
            )
            store_path = metric_store_path(run_dir)
            if not store_path.is_file():
                issues.append(f"{command.label} metric store is missing: {store_path}")
                continue
            store = MetricStore(store_path)
            latest = {name: store.latest_metric(name) for name in required_metrics}
            missing = sorted(name for name, value in latest.items() if value is None)
            if missing:
                issues.append(f"{command.label} metrics are missing: {', '.join(missing)}")
            history = store.metric_history("train/throughput/loop/rate")
            measured = [float(sample["value"]) for sample in history[warmup:]]
            if not measured:
                issues.append(
                    f"{command.label} has no loop-rate samples after {warmup} warmups"
                )
                continue
            if any(value <= 0.0 or not math.isfinite(value) for value in measured):
                issues.append(f"{command.label} has invalid loop-rate history")
                continue
            run_median = float(statistics.median(measured))
            paired.setdefault(case_name, {}).setdefault(repeat, {})[variant] = run_median
            run_samples.append(
                {
                    "label": command.label,
                    "case": case_name,
                    "variant": variant,
                    "repeat": repeat,
                    "metric_store": str(store_path),
                    "warmup_iterations": warmup,
                    "loop_rate_history": history,
                    "loop_rate_measured_median": run_median,
                    "metrics": latest,
                }
            )
        evidence["samples"] = run_samples
        case_results: dict[str, Any] = {}
        case_speedups: list[float] = []
        expected_repeats = int(profile.payload["repeats"])
        minimum_case = float(profile.payload["min_case_speedup"])
        for case in profile.payload["cases"]:
            case_name = str(case["name"])
            ratios: list[float] = []
            for repeat in range(1, expected_repeats + 1):
                variants = paired.get(case_name, {}).get(repeat, {})
                if set(variants) != {"baseline", "candidate"}:
                    issues.append(f"{case_name} repeat {repeat} lacks a complete AB/BA pair")
                    continue
                ratios.append(variants["candidate"] / variants["baseline"])
            if not ratios:
                continue
            speedup = math.prod(ratios) ** (1.0 / len(ratios))
            case_speedups.append(speedup)
            case_results[case_name] = {
                "paired_speedups": ratios,
                "geometric_mean_speedup": speedup,
                "minimum_required_speedup": minimum_case,
            }
            if speedup < minimum_case:
                issues.append(
                    f"{case_name} candidate regressed: {speedup:.6f} < {minimum_case:.6f}"
                )
        minimum_geomean = float(profile.payload["min_geomean_speedup"])
        if len(case_speedups) != len(profile.payload["cases"]):
            issues.append("PPO backend comparison lacks a valid speedup for every case")
        else:
            geomean = math.prod(case_speedups) ** (1.0 / len(case_speedups))
            if geomean < minimum_geomean:
                issues.append(
                    "PPO backend geometric-mean speedup missed gate: "
                    f"{geomean:.6f} < {minimum_geomean:.6f}"
                )
            evidence["comparison"] = {
                "cases": case_results,
                "geometric_mean_speedup": geomean,
                "minimum_geometric_mean_speedup": minimum_geomean,
                "minimum_case_speedup": minimum_case,
            }

    return {"passed": not issues, "issues": issues, "evidence": evidence}


def list_profiles(args: argparse.Namespace) -> int:
    profiles = load_benchmark_profiles(args.profile_dir)
    rows = [
        {
            "name": profile.name,
            "kind": profile.kind,
            "description": profile.description,
            "path": str(profile.path),
        }
        for profile in profiles
    ]
    if args.json:
        print(_json(rows))
        return 0
    for row in rows:
        suffix = f" - {row['description']}" if row["description"] else ""
        print(f"{row['name']} ({row['kind']}){suffix}")
    return 0


def show_profile(args: argparse.Namespace) -> int:
    profile = find_benchmark_profile(args.profile, profile_dir=args.profile_dir)
    commands = build_benchmark_commands(profile)
    payload = {
        "profile": profile.payload,
        "path": str(profile.path),
        "commands": _command_plan(commands),
    }
    print(_json(payload))
    return 0


def run_command(command: BenchmarkCommand) -> dict[str, Any]:
    env = os.environ.copy()
    if command.env:
        env.update(command.env)
    started_at = datetime.now(UTC)
    load_before = _load_average()
    result = subprocess.run(
        command.argv,
        check=False,
        cwd=command.cwd,
        env=env,
        input=command.stdin,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    finished_at = datetime.now(UTC)
    load_after = _load_average()
    return {
        "label": command.label,
        "argv": list(command.argv),
        "cwd": str(command.cwd) if command.cwd else None,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "load_average_before": load_before,
        "load_average_after": load_after,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def run_profile(args: argparse.Namespace) -> int:
    profile = find_benchmark_profile(args.profile, profile_dir=args.profile_dir)
    commands = build_benchmark_commands(profile)
    if args.dry_run:
        print(
            _json({"profile": profile.name, "dry_run": True, "commands": _command_plan(commands)})
        )
        return 0

    execution_id = _timestamp()
    commands = _execution_commands(
        profile,
        commands,
        execution_id=execution_id,
        output_dir=args.output_dir,
    )
    plan = _command_plan(commands)
    environment_before = _environment_snapshot(profile)
    if profile.kind == "ppo_backend_comparison":
        preflight_issues: list[str] = []
        if profile.payload.get("require_clean_source") is True and environment_before.get(
            "source_dirty"
        ) is not False:
            preflight_issues.append("PPO backend comparison requires a clean source checkout")
        required_device = str(profile.payload["required_cuda_device_name"])
        if required_device not in environment_before.get("cuda_devices", []):
            preflight_issues.append(
                "PPO backend comparison requires CUDA device "
                f"{required_device!r}; found {environment_before.get('cuda_devices', [])}"
            )
        if preflight_issues:
            for issue in preflight_issues:
                print(f"benchmark preflight failed: {issue}")
            return 1
    results = []
    for command in commands:
        print(f"running benchmark command: {command.label}", flush=True)
        result = run_command(command)
        results.append(result)
        if result["returncode"] != 0 and not args.keep_going:
            break

    validation = validate_benchmark_results(profile, commands, results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{execution_id}-{profile.name}.json"
    output = {
        "profile": profile.payload,
        "profile_path": str(profile.path),
        "commands": plan,
        "environment_before": environment_before,
        "environment_after": _environment_snapshot(profile),
        "results": results,
        "validation": validation,
        "status": "passed" if validation["passed"] else "failed",
    }
    output_path.write_text(_json(output) + "\n", encoding="utf-8")
    print(f"wrote benchmark result: {output_path}")
    return 0 if validation["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = ExactArgumentParser(
        prog="gradlab benchmark",
        description="Run named gradlab benchmark profiles.",
    )
    parser.add_argument("--profile-dir", type=Path, default=DEFAULT_PROFILE_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List available benchmark profiles.")
    list_parser.add_argument("--json", action="store_true")
    list_parser.set_defaults(func=list_profiles)

    show_parser = subparsers.add_parser("show", help="Show a profile and its command plan.")
    show_parser.add_argument("profile")
    show_parser.set_defaults(func=show_profile)

    run_parser = subparsers.add_parser("run", help="Run a benchmark profile.")
    run_parser.add_argument("profile")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--keep-going", action="store_true")
    run_parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULT_DIR)
    run_parser.set_defaults(func=run_profile)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
