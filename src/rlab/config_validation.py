from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rlab.benchmark_profiles import load_benchmark_profiles
from rlab.compute_targets import load_instance_config
from rlab.fleet import load_capacity_policy, load_fleet_config, validate_capacity_policy
from rlab.job_queue import load_spec_document
from rlab.seeds import validate_training_seed


YAML_EXTENSIONS = {".yaml", ".yml"}
GOAL_SCHEMA_VERSION = 1
RECIPE_SCHEMA_VERSION = 1
BENCHMARK_BASELINES_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str

    def to_json(self) -> dict[str, str]:
        return {"path": self.path, "message": self.message}


@dataclass(frozen=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...]
    counts: dict[str, int]

    @property
    def ok(self) -> bool:
        return not self.issues

    def to_json(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "counts": dict(sorted(self.counts.items())),
            "issues": [issue.to_json() for issue in self.issues],
        }


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _resolve_repo_path(repo_root: Path, value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return repo_root / path


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("must contain a YAML object")
    return payload


def _label_path(label: str, key: str) -> str:
    return f"{label}.{key}" if label else key


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_key(document: Mapping[str, Any], key: str, *, label: str) -> Any:
    if key not in document:
        raise ValueError(f"{_label_path(label, key)} is required")
    return document[key]


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _require_non_empty_string(document: Mapping[str, Any], key: str, *, label: str) -> str:
    value = _require_key(document, key, label=label)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{_label_path(label, key)} must be a non-empty string")
    return value.strip()


def _require_int(
    document: Mapping[str, Any],
    key: str,
    *,
    label: str,
    minimum: int | None = None,
) -> int:
    value = _require_key(document, key, label=label)
    if not _is_int(value):
        raise ValueError(f"{_label_path(label, key)} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{_label_path(label, key)} must be >= {minimum}")
    return value


def _require_number(
    document: Mapping[str, Any],
    key: str,
    *,
    label: str,
    minimum: float | None = None,
) -> float:
    value = _require_key(document, key, label=label)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{_label_path(label, key)} must be a number")
    number = float(value)
    if minimum is not None and number < minimum:
        raise ValueError(f"{_label_path(label, key)} must be >= {minimum:g}")
    return number


def _require_string_list(document: Mapping[str, Any], key: str, *, label: str) -> list[str]:
    value = _require_key(document, key, label=label)
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{_label_path(label, key)} must be a list")
    if not value:
        raise ValueError(f"{_label_path(label, key)} must not be empty")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{_label_path(label, key)}[{index}] must be a non-empty string")
        result.append(item.strip())
    return result


def _require_int_list(document: Mapping[str, Any], key: str, *, label: str) -> list[int]:
    value = _require_key(document, key, label=label)
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{_label_path(label, key)} must be a list")
    if not value:
        raise ValueError(f"{_label_path(label, key)} must not be empty")
    result: list[int] = []
    for index, item in enumerate(value):
        if not _is_int(item):
            raise ValueError(f"{_label_path(label, key)}[{index}] must be an integer")
        result.append(item)
    return result


def _require_int_list_value(value: Any, *, label: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{label} must be a list")
    if not value:
        raise ValueError(f"{label} must not be empty")
    result: list[int] = []
    for index, item in enumerate(value):
        if not _is_int(item):
            raise ValueError(f"{label}[{index}] must be an integer")
        result.append(item)
    return result


def _require_schema_version(document: Mapping[str, Any], expected: int, *, label: str) -> None:
    schema_version = _require_int(document, "schema_version", label=label, minimum=1)
    if schema_version != expected:
        raise ValueError(f"{_label_path(label, 'schema_version')} must be {expected}")


def _require_existing_file(repo_root: Path, document: Mapping[str, Any], key: str, *, label: str) -> Path:
    value = _require_non_empty_string(document, key, label=label)
    path = _resolve_repo_path(repo_root, value)
    if not path.is_file():
        raise ValueError(f"{_label_path(label, key)} does not exist: {value}")
    return path


def validate_goal_contract(path: Path, repo_root: Path | None = None) -> None:
    repo_root = (repo_root or Path(".")).resolve()
    document = _load_yaml_mapping(path)
    label = f"goal file {_display_path(path, repo_root)}"
    _require_schema_version(document, GOAL_SCHEMA_VERSION, label=label)
    _require_non_empty_string(document, "goal_slug", label=label)
    _require_non_empty_string(document, "title", label=label)
    _require_non_empty_string(document, "status", label=label)
    goal_dir = _require_non_empty_string(document, "goal_dir", label=label)
    if not _resolve_repo_path(repo_root, goal_dir).is_dir():
        raise ValueError(f"{_label_path(label, 'goal_dir')} does not exist: {goal_dir}")

    objective = _require_mapping(_require_key(document, "objective", label=label), label=f"{label}.objective")
    _require_non_empty_string(objective, "game", label=f"{label}.objective")
    _require_string_list(objective, "states", label=f"{label}.objective")
    _require_non_empty_string(objective, "algorithm", label=f"{label}.objective")
    _require_non_empty_string(objective, "primary_metric", label=f"{label}.objective")
    _require_number(objective, "success_threshold", label=f"{label}.objective")
    _require_int(objective, "success_window_attempts", label=f"{label}.objective", minimum=1)
    _require_int(objective, "max_train_timesteps", label=f"{label}.objective", minimum=1)

    selection_policy = _require_mapping(
        _require_key(document, "selection_policy", label=label),
        label=f"{label}.selection_policy",
    )
    _require_string_list(selection_policy, "rank_order", label=f"{label}.selection_policy")

    seed_protocol = _require_mapping(
        _require_key(document, "seed_protocol", label=label),
        label=f"{label}.seed_protocol",
    )
    has_screen = "screen" in seed_protocol
    has_screen_pairs = "screen_pairs" in seed_protocol
    if not has_screen and not has_screen_pairs:
        raise ValueError(f"{label}.seed_protocol must define screen or screen_pairs")
    if has_screen:
        for index, seed in enumerate(_require_int_list(seed_protocol, "screen", label=f"{label}.seed_protocol")):
            validate_training_seed(seed, label=f"{label}.seed_protocol.screen[{index}]", seed_span=1)
    if has_screen_pairs:
        raw_pairs = _require_key(seed_protocol, "screen_pairs", label=f"{label}.seed_protocol")
        if not isinstance(raw_pairs, Sequence) or isinstance(raw_pairs, str | bytes) or not raw_pairs:
            raise ValueError(f"{label}.seed_protocol.screen_pairs must be a non-empty list")
        expected_pair_size = seed_protocol.get("screen_batch_size")
        if expected_pair_size is not None and not _is_int(expected_pair_size):
            raise ValueError(f"{label}.seed_protocol.screen_batch_size must be an integer")
        for pair_index, pair in enumerate(raw_pairs):
            seeds = _require_int_list_value(
                pair,
                label=f"{label}.seed_protocol.screen_pairs[{pair_index}]",
            )
            if expected_pair_size is not None and len(seeds) != expected_pair_size:
                raise ValueError(
                    f"{label}.seed_protocol.screen_pairs[{pair_index}] "
                    f"must contain {expected_pair_size} seed(s)"
                )
            for seed_index, seed in enumerate(seeds):
                validate_training_seed(
                    seed,
                    label=f"{label}.seed_protocol.screen_pairs[{pair_index}][{seed_index}]",
                    seed_span=1,
                )
    if "confirm" in seed_protocol:
        for index, seed in enumerate(_require_int_list(seed_protocol, "confirm", label=f"{label}.seed_protocol")):
            validate_training_seed(seed, label=f"{label}.seed_protocol.confirm[{index}]", seed_span=1)

    spec_path = _require_existing_file(repo_root, document, "default_train_spec_file", label=label)
    if spec_path.suffix.lower() not in YAML_EXTENSIONS:
        raise ValueError(f"{_label_path(label, 'default_train_spec_file')} must be YAML")
    load_spec_document(spec_path)

    capacity_path = _require_existing_file(repo_root, document, "capacity_policy_file", label=label)
    if capacity_path.suffix.lower() not in YAML_EXTENSIONS:
        raise ValueError(f"{_label_path(label, 'capacity_policy_file')} must be YAML")

    execution = _require_mapping(_require_key(document, "execution", label=label), label=f"{label}.execution")
    for key in ("hardware_config_file", "fleet_config_file"):
        config_path = _require_existing_file(repo_root, execution, key, label=f"{label}.execution")
        if config_path.suffix.lower() not in YAML_EXTENSIONS:
            raise ValueError(f"{_label_path(f'{label}.execution', key)} must be YAML")


def validate_train_recipe(path: Path) -> None:
    document = _load_yaml_mapping(path)
    label = f"recipe file {path}"
    _require_schema_version(document, RECIPE_SCHEMA_VERSION, label=label)
    kind = _require_non_empty_string(document, "kind", label=label)
    if kind != "train_recipe":
        raise ValueError(f"{label}.kind must be train_recipe")
    _require_non_empty_string(document, "slug", label=label)
    _require_non_empty_string(document, "algorithm", label=label)
    env = _require_mapping(_require_key(document, "env", label=label), label=f"{label}.env")
    train = _require_mapping(_require_key(document, "train", label=label), label=f"{label}.train")
    reward = _require_mapping(_require_key(document, "reward", label=label), label=f"{label}.reward")
    logging = _require_mapping(_require_key(document, "logging", label=label), label=f"{label}.logging")

    _require_non_empty_string(env, "game", label=f"{label}.env")
    _require_non_empty_string(env, "action_set", label=f"{label}.env")
    for key in ("n_steps", "batch_size", "n_epochs"):
        _require_int(train, key, label=f"{label}.train", minimum=1)
    if "reward_mode" in reward:
        _require_non_empty_string(reward, "reward_mode", label=f"{label}.reward")
    _require_int(logging, "timesteps", label=f"{label}.logging", minimum=1)
    wandb = _require_key(logging, "wandb", label=f"{label}.logging")
    if not isinstance(wandb, bool):
        raise ValueError(f"{label}.logging.wandb must be a boolean")
    if "wandb_mode" in logging:
        wandb_mode = _require_non_empty_string(logging, "wandb_mode", label=f"{label}.logging")
        if wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError(f"{label}.logging.wandb_mode must be one of online, offline, disabled")


def validate_instance_config(path: Path, repo_root: Path | None = None) -> None:
    repo_root = repo_root or Path(".")
    config = load_instance_config(repo_root, path)
    instances = _require_mapping(config.get("instances"), label=f"instance config {path}.instances")
    if not instances:
        raise ValueError(f"instance config {path}.instances must not be empty")
    for name, raw in instances.items():
        label = f"instance config {path}.instances.{name}"
        instance = _require_mapping(raw, label=label)
        _require_non_empty_string(instance, "kind", label=label)
        default_workers = _require_int(instance, "default_workers", label=label, minimum=1)
        max_workers = _require_int(instance, "hardware_max_workers", label=label, minimum=default_workers)
        if max_workers < default_workers:
            raise ValueError(f"{label}.hardware_max_workers must be >= default_workers")


def validate_fleet_and_capacity(repo_root: Path) -> None:
    config = load_fleet_config(repo_root)
    policy = load_capacity_policy(repo_root)
    _require_schema_version(policy, 1, label="capacity policy")
    validate_capacity_policy(policy, config)
    lanes = policy.get("lanes")
    if not isinstance(lanes, Sequence) or isinstance(lanes, str | bytes) or not lanes:
        raise ValueError("capacity policy lanes must be a non-empty list")
    for index, raw in enumerate(lanes):
        label = f"capacity policy lanes[{index}]"
        lane = _require_mapping(raw, label=label)
        _require_non_empty_string(lane, "name", label=label)
        _require_non_empty_string(lane, "target", label=label)
        _require_non_empty_string(lane, "manager", label=label)
        _require_int(lane, "max_runner_workers", label=label, minimum=1)
        _require_int(lane, "env_threads", label=label, minimum=1)
        _require_string_list(lane, "use_for", label=label)


def validate_benchmark_baselines(path: Path) -> None:
    document = _load_yaml_mapping(path)
    label = f"benchmark baselines file {path}"
    _require_schema_version(document, BENCHMARK_BASELINES_SCHEMA_VERSION, label=label)
    baselines = _require_mapping(_require_key(document, "baselines", label=label), label=f"{label}.baselines")
    if not baselines:
        raise ValueError(f"{label}.baselines must not be empty")
    for name, raw in baselines.items():
        baseline_label = f"{label}.baselines.{name}"
        baseline = _require_mapping(raw, label=baseline_label)
        _require_non_empty_string(baseline, "target", label=baseline_label)
        _require_non_empty_string(baseline, "host", label=baseline_label)
        _require_int(baseline, "workers", label=baseline_label, minimum=1)
        _require_int(baseline, "env_threads", label=baseline_label, minimum=1)


def _capture_issue(issues: list[ValidationIssue], path: Path, repo_root: Path, action: Any) -> None:
    try:
        action()
    except Exception as exc:  # noqa: BLE001 - validation should aggregate all schema failures.
        issues.append(ValidationIssue(path=_display_path(path, repo_root), message=str(exc)))


def validate_experiment_tree(repo_root: Path | str = Path(".")) -> ValidationReport:
    repo_root = Path(repo_root).resolve()
    experiments_dir = repo_root / "experiments"
    issues: list[ValidationIssue] = []
    counts: dict[str, int] = {}

    if not experiments_dir.is_dir():
        return ValidationReport(
            issues=(ValidationIssue(path="experiments", message="experiments directory does not exist"),),
            counts={},
        )

    yaml_files = sorted(experiments_dir.rglob("*.yaml")) + sorted(experiments_dir.rglob("*.yml"))
    json_files = sorted(experiments_dir.rglob("*.json"))
    counts["yaml_files"] = len(yaml_files)
    counts["json_files"] = len(json_files)
    for path in json_files:
        issues.append(ValidationIssue(path=_display_path(path, repo_root), message="experiments configs must be YAML"))

    goals = sorted((experiments_dir / "goals").glob("*/goal.yaml"))
    counts["goals"] = len(goals)
    for path in goals:
        _capture_issue(issues, path, repo_root, lambda path=path: validate_goal_contract(path, repo_root))

    specs = sorted((experiments_dir / "goals").glob("*/specs/*.yaml"))
    counts["train_specs"] = len(specs)
    for path in specs:
        _capture_issue(issues, path, repo_root, lambda path=path: load_spec_document(path))

    recipes_dir = experiments_dir / "recipes"
    recipes = sorted(recipes_dir.rglob("*.yaml")) if recipes_dir.is_dir() else []
    counts["recipes"] = len(recipes)
    for path in recipes:
        _capture_issue(issues, path, repo_root, lambda path=path: validate_train_recipe(path))

    instances_path = experiments_dir / "instances.yaml"
    counts["instance_configs"] = int(instances_path.is_file())
    if instances_path.is_file():
        _capture_issue(
            issues,
            instances_path,
            repo_root,
            lambda: validate_instance_config(instances_path, repo_root),
        )

    fleet_path = experiments_dir / "fleet.yaml"
    capacity_path = experiments_dir / "policies" / "capacity_policy.yaml"
    counts["fleet_configs"] = int(fleet_path.is_file())
    counts["capacity_policies"] = int(capacity_path.is_file())
    if fleet_path.is_file() and capacity_path.is_file():
        _capture_issue(issues, capacity_path, repo_root, lambda: validate_fleet_and_capacity(repo_root))
    else:
        if not fleet_path.is_file():
            issues.append(ValidationIssue(path="experiments/fleet.yaml", message="file is required"))
        if not capacity_path.is_file():
            issues.append(ValidationIssue(path="experiments/policies/capacity_policy.yaml", message="file is required"))

    benchmark_dir = experiments_dir / "benchmarks"
    benchmark_baselines = benchmark_dir / "baselines.yaml"
    counts["benchmark_baselines"] = int(benchmark_baselines.is_file())
    if benchmark_baselines.is_file():
        _capture_issue(
            issues,
            benchmark_baselines,
            repo_root,
            lambda: validate_benchmark_baselines(benchmark_baselines),
        )

    profile_dir = benchmark_dir / "profiles"
    if profile_dir.is_dir():
        _capture_issue(issues, profile_dir, repo_root, lambda: load_benchmark_profiles(profile_dir))
        counts["benchmark_profiles"] = len(sorted(profile_dir.glob("*.yaml")))
    else:
        counts["benchmark_profiles"] = 0

    return ValidationReport(issues=tuple(issues), counts=counts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlab validate",
        description="Validate checked-in YAML experiment, goal, spec, recipe, and ops configs.",
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--json", action="store_true", help="Emit machine-readable validation output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = validate_experiment_tree(args.repo_root)
    if args.json:
        print(json.dumps(report.to_json(), indent=2, sort_keys=True))
    elif report.ok:
        counts = ", ".join(f"{name}={value}" for name, value in sorted(report.counts.items()))
        print(f"YAML config validation passed ({counts}).")
    else:
        print("YAML config validation failed:", file=sys.stderr)
        for issue in report.issues:
            print(f"- {issue.path}: {issue.message}", file=sys.stderr)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
