from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from gradlab.benchmark_profiles import load_benchmark_profiles
from gradlab.cli_parser import ExactArgumentParser
from gradlab.experiment_contracts import validate_env_config_file
from gradlab.modal_eval_config import load_modal_eval_config
from gradlab.recipe_documents import compose_train_document, load_goal_contract
from gradlab.validation import display_path


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


def _capture_issue(
    issues: list[ValidationIssue],
    path: Path,
    repo_root: Path,
    action: Any,
) -> None:
    try:
        action()
    except Exception as exc:  # noqa: BLE001 - validation aggregates schema failures.
        issues.append(ValidationIssue(path=display_path(path, repo_root), message=str(exc)))


def validate_experiment_tree(repo_root: Path | str = Path(".")) -> ValidationReport:
    repo_root = Path(repo_root).resolve()
    experiments_dir = repo_root / "experiments"
    issues: list[ValidationIssue] = []
    counts: dict[str, int] = {}

    if not experiments_dir.is_dir():
        return ValidationReport(
            issues=(
                ValidationIssue(path="experiments", message="experiments directory does not exist"),
            ),
            counts={},
        )

    yaml_files = sorted(experiments_dir.rglob("*.yaml")) + sorted(
        experiments_dir.rglob("*.yml")
    )
    json_files = sorted(experiments_dir.rglob("*.json"))
    counts["yaml_files"] = len(yaml_files)
    counts["json_files"] = len(json_files)
    for path in json_files:
        issues.append(
            ValidationIssue(
                path=display_path(path, repo_root),
                message="experiments configs must be YAML",
            )
        )

    goals_dir = experiments_dir / "goals"
    goals = sorted(goals_dir.rglob("_goal.yaml"))
    counts["goals"] = len(goals)
    for path in goals:
        _capture_issue(
            issues,
            path,
            repo_root,
            lambda path=path: load_goal_contract(path, repo_root),
        )

    report_manifests = sorted(goals_dir.rglob("_reports.yaml"))
    counts["report_manifests"] = len(report_manifests)
    if report_manifests:
        from gradlab.wandb_reports import validate_report_declarations

        _capture_issue(
            issues,
            report_manifests[0],
            repo_root,
            lambda: validate_report_declarations(repo_root),
        )

    recipes = sorted(goals_dir.rglob("recipes/*.yaml"))
    counts["train_recipes"] = len(recipes)
    recipes_by_goal = {path.parent.parent.resolve() for path in recipes}
    for goal_path in goals:
        if goal_path.parent.resolve() not in recipes_by_goal:
            issues.append(
                ValidationIssue(
                    path=display_path(goal_path, repo_root),
                    message="active goal has no launchable recipe under its recipes directory",
                )
            )
    for path in recipes:
        goal_path = path.parent.parent / "_goal.yaml"
        if not goal_path.is_file():
            issues.append(
                ValidationIssue(
                    path=display_path(path, repo_root),
                    message="goal-local recipe has no sibling _goal.yaml owner",
                )
            )
            continue
        _capture_issue(
            issues,
            path,
            repo_root,
            lambda path=path, goal_path=goal_path: compose_train_document(goal_path, path),
        )

    recipes_root = experiments_dir / "recipes"
    shared_recipe_leaves = sorted(
        path
        for path in recipes_root.rglob("*.yaml")
        if not path.is_relative_to(recipes_root / "_presets")
    )
    for path in shared_recipe_leaves:
        issues.append(
            ValidationIssue(
                path=display_path(path, repo_root),
                message=(
                    "shared recipe directories may contain only reusable _presets; "
                    "launchable recipes belong under their goal"
                ),
            )
        )

    env_configs = sorted(goals_dir.glob("*/_env-*.yaml"))
    counts["env_configs"] = len(env_configs)
    for path in env_configs:
        _capture_issue(
            issues,
            path,
            repo_root,
            lambda path=path: validate_env_config_file(path),
        )

    modal_eval_path = experiments_dir / "modal_eval.yaml"
    counts["modal_eval_configs"] = int(modal_eval_path.is_file())
    if modal_eval_path.is_file():
        _capture_issue(
            issues,
            modal_eval_path,
            repo_root,
            lambda: load_modal_eval_config(modal_eval_path),
        )
    else:
        issues.append(
            ValidationIssue(path="experiments/modal_eval.yaml", message="file is required")
        )

    profile_dir = experiments_dir / "benchmarks" / "profiles"
    if profile_dir.is_dir():
        _capture_issue(
            issues,
            profile_dir,
            repo_root,
            lambda: load_benchmark_profiles(profile_dir),
        )
        counts["benchmark_profiles"] = len(sorted(profile_dir.glob("*.yaml")))
    else:
        counts["benchmark_profiles"] = 0

    return ValidationReport(issues=tuple(issues), counts=counts)


def build_parser() -> argparse.ArgumentParser:
    parser = ExactArgumentParser(
        prog="gradlab validate",
        description="Validate checked-in YAML experiment, goal, recipe, benchmark, and ops configs.",
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")
    parser.add_argument(
        "--load-goal",
        type=Path,
        help="Print the final composed goal contract for a _goal.yaml path.",
    )
    parser.add_argument(
        "--format",
        choices=("yaml", "json"),
        default="yaml",
        help="Output format for --load-goal.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.load_goal is not None:
        document = load_goal_contract(args.load_goal, args.repo_root)
        output_format = "json" if args.json else args.format
        if output_format == "json":
            print(json.dumps(document, indent=2, sort_keys=True))
        else:
            print(yaml.safe_dump(document, sort_keys=False), end="")
        return 0

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
