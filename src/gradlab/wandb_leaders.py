from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

from gradlab.json_utils import json_safe
from gradlab.metric_names import (
    LEADER_CHECKPOINT_ARTIFACT_REF,
    LEADER_CHECKPOINT_EVALUATION_SOURCE,
    LEADER_CHECKPOINT_RETURN_SHAPED_MEAN,
    LEADER_CHECKPOINT_STEP,
    LEADER_CHECKPOINT_SUCCESS_ACROSS_STARTS_RATE_MEAN,
    LEADER_CHECKPOINT_SUCCESS_ACROSS_STARTS_RATE_MIN,
    LEGACY_METRICS_SCHEMA_VERSION,
    TRAIN_EPISODE_RETURN_SHAPED_ACROSS_ORIGINS_ROLLING_UP_TO_100_MEAN,
    TRAIN_GLOBAL_STEP,
    TRAIN_OUTCOME_SUCCESS_ACROSS_STARTS_WINDOW_100_RATE_MIN,
    V13_LEADER_CHECKPOINT_ARTIFACT_REF,
    V13_LEADER_CHECKPOINT_EVALUATION_SOURCE,
    V13_LEADER_CHECKPOINT_RETURN_SHAPED_MEAN,
    V13_LEADER_CHECKPOINT_SUCCESS_ACROSS_STARTS_RATE_MEAN,
    V13_LEADER_CHECKPOINT_SUCCESS_ACROSS_STARTS_RATE_MIN,
    evaluation_metric_schema,
    leader_metric_for_rank_metric,
)
from gradlab.ranking import parse_objective_rank, rank_score
from gradlab.wandb_utils import DEFAULT_WANDB_PROJECT_PATH, load_wandb_env


RUN_OBJECTIVE_KEYS = (
    TRAIN_OUTCOME_SUCCESS_ACROSS_STARTS_WINDOW_100_RATE_MIN,
    TRAIN_EPISODE_RETURN_SHAPED_ACROSS_ORIGINS_ROLLING_UP_TO_100_MEAN,
)
RUN_PRIMARY_ORDER = "-created_at"
CHECKPOINT_SUCCESS_KEYS = (
    LEADER_CHECKPOINT_SUCCESS_ACROSS_STARTS_RATE_MIN,
    V13_LEADER_CHECKPOINT_SUCCESS_ACROSS_STARTS_RATE_MIN,
)
CHECKPOINT_SUCCESS_MEAN_KEYS = (
    LEADER_CHECKPOINT_SUCCESS_ACROSS_STARTS_RATE_MEAN,
    V13_LEADER_CHECKPOINT_SUCCESS_ACROSS_STARTS_RATE_MEAN,
)
CHECKPOINT_RETURN_KEYS = (
    LEADER_CHECKPOINT_RETURN_SHAPED_MEAN,
    V13_LEADER_CHECKPOINT_RETURN_SHAPED_MEAN,
)
CHECKPOINT_STEP_KEYS = (LEADER_CHECKPOINT_STEP,)
# API ordering is only a retrieval hint. Goal-specific ranking happens in Python,
# because the primary objective may be either minimized or maximized.
CHECKPOINT_PRIMARY_ORDER = "-created_at"
WANDB_RUNS_PER_PAGE = 200


@dataclass(frozen=True)
class RunScore:
    goal_slug: str
    recipe_slug: str
    reward_shape: str
    reward_shape_sha256: str
    effective_goal_contract_sha256: str
    reward_shape_is_default: bool
    run_id: str
    run_name: str
    url: str
    seed: int | None
    objective: float
    steps: int | None = None


@dataclass(frozen=True)
class RunLeader:
    goal_slug: str
    recipe_slug: str
    reward_shape: str
    reward_shape_sha256: str
    effective_goal_contract_sha256: str
    reward_shape_is_default: bool
    seeds: int
    worst_seed: float
    mean_seed: float
    best_seed: float
    runs: tuple[RunScore, ...]
    mean_steps: float | None = None


@dataclass(frozen=True)
class CheckpointLeader:
    goal_slug: str
    recipe_slug: str
    reward_shape: str
    reward_shape_sha256: str
    effective_goal_contract_sha256: str
    reward_shape_is_default: bool
    run_id: str
    run_name: str
    url: str
    objective: float
    objective_name: str
    success_rate_min: float | None
    success_rate_mean: float | None
    progress_max: float | None
    return_mean: float
    checkpoint_step: int | None
    artifact_ref: str
    eval_source: str
    rank_score: tuple[float, ...]


def _mapping_value(mapping: Mapping[str, Any], key: str) -> Any:
    try:
        return mapping.get(key)
    except AttributeError:
        return None


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _first_float(mapping: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        value = _mapping_value(mapping, key)
        if value is None:
            continue
        try:
            return float(value)
        except TypeError, ValueError:
            continue
    return None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except TypeError, ValueError:
        return None


def _summary_metric_key(metric: str) -> str:
    return f"summary_metrics.{metric}"


def _exists_filter(metric: str) -> dict[str, Any]:
    return {_summary_metric_key(metric): {"$exists": True}}


def _and_filters(*filters: Mapping[str, Any] | None) -> dict[str, Any]:
    parts = [dict(item) for item in filters if item]
    if not parts:
        return {}
    if len(parts) == 1:
        return parts[0]
    return {"$and": parts}


def goal_run_filter(goal: str | None) -> dict[str, Any]:
    if not goal:
        return {}
    return {"config.goal_slug": goal}


def run_objective_filter(objective_keys: Sequence[str]) -> dict[str, Any]:
    return {"$or": [_exists_filter(key) for key in objective_keys]}


def run_query_objective_keys(args: argparse.Namespace) -> tuple[str, ...]:
    explicit_keys = tuple(args.objective_key or ())
    if explicit_keys:
        return explicit_keys
    return RUN_OBJECTIVE_KEYS


def checkpoint_summary_filter() -> dict[str, Any]:
    return {
        "$and": [
            {
                "$or": [
                    _exists_filter(LEADER_CHECKPOINT_ARTIFACT_REF),
                    _exists_filter(V13_LEADER_CHECKPOINT_ARTIFACT_REF),
                ]
            },
            {
                "$or": [
                    _exists_filter(LEADER_CHECKPOINT_RETURN_SHAPED_MEAN),
                    _exists_filter(V13_LEADER_CHECKPOINT_RETURN_SHAPED_MEAN),
                ]
            },
        ]
    }


def run_score(run: Any, *, objective_keys: Sequence[str]) -> RunScore | None:
    config = dict(getattr(run, "config", {}) or {})
    summary = getattr(run, "summary", {}) or {}
    configured_rank = parse_objective_rank(config.get("selection_rank"))
    configured_primary = (
        configured_rank[0].metric
        if configured_rank and configured_rank[0].metric.startswith("train/")
        else ""
    )
    candidate_keys = tuple(
        dict.fromkeys((configured_primary, *objective_keys) if configured_primary else objective_keys)
    )
    objective = _first_float(summary, candidate_keys)
    if objective is None:
        return None
    goal_slug = _first_text(config.get("goal_slug"))
    recipe_slug = _first_text(config.get("recipe_slug"))
    if not goal_slug or not recipe_slug:
        return None
    return RunScore(
        goal_slug=goal_slug,
        recipe_slug=recipe_slug,
        reward_shape=_first_text(config.get("reward_shape")),
        reward_shape_sha256=_first_text(config.get("reward_shape_sha256")),
        effective_goal_contract_sha256=_first_text(
            config.get("effective_goal_contract_sha256")
        ),
        reward_shape_is_default=bool(config.get("reward_shape_is_default", False)),
        run_id=str(getattr(run, "id", "") or ""),
        run_name=str(getattr(run, "name", "") or ""),
        url=str(getattr(run, "url", "") or ""),
        seed=_optional_int(config.get("seed")),
        objective=float(objective),
        steps=_optional_int(_mapping_value(summary, TRAIN_GLOBAL_STEP)),
    )


def rank_run_leaders(scores: Iterable[RunScore], *, min_seeds: int = 1) -> list[RunLeader]:
    grouped: dict[tuple[str, str, str, str], list[RunScore]] = defaultdict(list)
    for score in scores:
        shape_identity = score.reward_shape_sha256 or score.reward_shape
        grouped[
            (
                score.goal_slug,
                score.recipe_slug,
                shape_identity,
                score.effective_goal_contract_sha256,
            )
        ].append(score)

    leaders: list[RunLeader] = []
    for (
        goal_slug,
        recipe_slug,
        _shape_identity,
        effective_goal_hash,
    ), group_scores in grouped.items():
        if len(group_scores) < min_seeds:
            continue
        ordered_runs = tuple(
            sorted(
                group_scores,
                key=lambda item: (
                    item.objective,
                    -(item.steps if item.steps is not None else float("inf")),
                ),
                reverse=True,
            )
        )
        values = [item.objective for item in ordered_runs]
        step_values = [item.steps for item in ordered_runs if item.steps is not None]
        leaders.append(
            RunLeader(
                goal_slug=goal_slug,
                recipe_slug=recipe_slug,
                reward_shape=group_scores[0].reward_shape,
                reward_shape_sha256=group_scores[0].reward_shape_sha256,
                effective_goal_contract_sha256=effective_goal_hash,
                reward_shape_is_default=group_scores[0].reward_shape_is_default,
                seeds=len(ordered_runs),
                worst_seed=min(values),
                mean_seed=mean(values),
                best_seed=max(values),
                runs=ordered_runs,
                mean_steps=mean(step_values) if step_values else None,
            )
        )
    return sorted(
        leaders,
        key=lambda item: (
            item.reward_shape_is_default,
            item.reward_shape,
            item.worst_seed,
            item.mean_seed,
            item.best_seed,
            -(item.mean_steps if item.mean_steps is not None else float("inf")),
            item.seeds,
        ),
        reverse=True,
    )


def checkpoint_leader(run: Any) -> CheckpointLeader | None:
    config = dict(getattr(run, "config", {}) or {})
    summary = getattr(run, "summary", {}) or {}
    try:
        metrics_schema_version = int(config.get("metrics_schema_version"))
        evaluation_metric_schema(metrics_schema_version)
    except (TypeError, ValueError):
        return None
    legacy = metrics_schema_version == LEGACY_METRICS_SCHEMA_VERSION
    success = _first_float(
        summary,
        (
            V13_LEADER_CHECKPOINT_SUCCESS_ACROSS_STARTS_RATE_MIN
            if legacy
            else LEADER_CHECKPOINT_SUCCESS_ACROSS_STARTS_RATE_MIN,
        ),
    )
    success_mean = _first_float(
        summary,
        (
            V13_LEADER_CHECKPOINT_SUCCESS_ACROSS_STARTS_RATE_MEAN
            if legacy
            else LEADER_CHECKPOINT_SUCCESS_ACROSS_STARTS_RATE_MEAN,
        ),
    )
    episode_return = _first_float(
        summary,
        (
            V13_LEADER_CHECKPOINT_RETURN_SHAPED_MEAN
            if legacy
            else LEADER_CHECKPOINT_RETURN_SHAPED_MEAN,
        ),
    )
    checkpoint_step = _optional_int(_first_float(summary, CHECKPOINT_STEP_KEYS))
    artifact_ref = _first_text(
        _mapping_value(
            summary,
            (
                V13_LEADER_CHECKPOINT_ARTIFACT_REF
                if legacy
                else LEADER_CHECKPOINT_ARTIFACT_REF
            ),
        ),
    )
    if episode_return is None or not artifact_ref:
        return None
    rank = parse_objective_rank(
        config.get("selection_rank"),
        metrics_schema_version=metrics_schema_version,
    )
    if not rank:
        return None
    rank_metrics: dict[str, Any] = {}
    progress: float | None = None
    for criterion in rank:
        leader_metric = leader_metric_for_rank_metric(
            criterion.metric,
            schema_version=metrics_schema_version,
        )
        value = (
            checkpoint_step
            if leader_metric == LEADER_CHECKPOINT_STEP
            else _first_float(summary, (leader_metric,))
        )
        rank_metrics[criterion.metric] = value
        if "/progress/" in criterion.metric and criterion.metric.endswith("/max"):
            progress = value
    objective = rank_metrics.get(rank[0].metric)
    if objective is None:
        return None
    return CheckpointLeader(
        goal_slug=_first_text(config.get("goal_slug")),
        recipe_slug=_first_text(config.get("recipe_slug")),
        reward_shape=_first_text(config.get("reward_shape")),
        reward_shape_sha256=_first_text(config.get("reward_shape_sha256")),
        effective_goal_contract_sha256=_first_text(
            config.get("effective_goal_contract_sha256")
        ),
        reward_shape_is_default=bool(config.get("reward_shape_is_default", False)),
        run_id=str(getattr(run, "id", "") or ""),
        run_name=str(getattr(run, "name", "") or ""),
        url=str(getattr(run, "url", "") or ""),
        objective=float(objective),
        objective_name=rank[0].metric,
        success_rate_min=success,
        success_rate_mean=success_mean,
        progress_max=progress,
        return_mean=episode_return,
        checkpoint_step=checkpoint_step,
        artifact_ref=artifact_ref,
        eval_source=_first_text(
            _mapping_value(
                summary,
                (
                    V13_LEADER_CHECKPOINT_EVALUATION_SOURCE
                    if legacy
                    else LEADER_CHECKPOINT_EVALUATION_SOURCE
                ),
            )
        ),
        rank_score=rank_score(rank_metrics, rank),
    )


def rank_checkpoint_leaders(leaders: Iterable[CheckpointLeader]) -> list[CheckpointLeader]:
    grouped: dict[tuple[str, str, str], list[CheckpointLeader]] = defaultdict(list)
    for leader in leaders:
        shape_identity = leader.reward_shape_sha256 or leader.reward_shape
        grouped[(leader.goal_slug, shape_identity, leader.effective_goal_contract_sha256)].append(
            leader
        )
    ordered_groups = sorted(
        grouped.values(),
        key=lambda rows: (
            not rows[0].reward_shape_is_default,
            rows[0].reward_shape,
        ),
    )
    return [
        leader
        for rows in ordered_groups
        for leader in sorted(rows, key=lambda item: item.rank_score, reverse=True)
    ]


def wandb_runs(
    *,
    project: str,
    goal: str | None = None,
    extra_filter: Mapping[str, Any] | None = None,
    order: str = "+created_at",
    lazy: bool = True,
):
    load_wandb_env()
    import wandb

    api = wandb.Api()
    filters = _and_filters(goal_run_filter(goal), extra_filter)
    return api.runs(
        project,
        filters=filters or None,
        order=order,
        per_page=WANDB_RUNS_PER_PAGE,
        lazy=lazy,
    )


def print_json(rows: Sequence[Any]) -> None:
    print(json.dumps(json_safe([asdict(row) for row in rows]), indent=2, sort_keys=True))


def print_run_leaders(rows: Sequence[RunLeader]) -> None:
    print(
        "goal_slug\trecipe_slug\treward_shape\tseeds\tworst_seed\tmean_seed\tbest_seed"
        "\tmean_steps"
    )
    for row in rows:
        mean_steps = f"{row.mean_steps:.6g}" if row.mean_steps is not None else ""
        print(
            f"{row.goal_slug}\t{row.recipe_slug}\t{row.reward_shape}\t{row.seeds}\t"
            f"{row.worst_seed:.6g}\t{row.mean_seed:.6g}\t{row.best_seed:.6g}\t{mean_steps}"
        )


def print_checkpoint_leaders(rows: Sequence[CheckpointLeader]) -> None:
    print(
        "goal_slug\trecipe_slug\treward_shape\tobjective\tobjective_name\tsuccess_min\t"
        "success_mean\treturn\tprogress\tstep\trun\tartifact_ref"
    )
    for row in rows:
        success_rate = f"{row.success_rate_min:.6g}" if row.success_rate_min is not None else ""
        success_rate_mean = (
            f"{row.success_rate_mean:.6g}" if row.success_rate_mean is not None else ""
        )
        progress = f"{row.progress_max:.6g}" if row.progress_max is not None else ""
        print(
            f"{row.goal_slug}\t{row.recipe_slug}\t{row.reward_shape}\t"
            f"{row.objective:.6g}\t"
            f"{row.objective_name}\t{success_rate}\t"
            f"{success_rate_mean}\t{row.return_mean:.6g}\t"
            f"{progress}\t"
            f"{row.checkpoint_step or ''}\t{row.run_name}\t{row.artifact_ref}"
        )


def add_common_args(parser: argparse.ArgumentParser, *, suppress_defaults: bool = False) -> None:
    default = argparse.SUPPRESS if suppress_defaults else None
    parser.add_argument(
        "--project",
        default=argparse.SUPPRESS if suppress_defaults else DEFAULT_WANDB_PROJECT_PATH,
    )
    parser.add_argument(
        "--goal",
        default=default,
        help="Limit to one W&B config.goal_slug.",
    )
    parser.add_argument(
        "--reward-shape",
        default=default,
        help="Limit to one W&B config.reward_shape; required for an unambiguous winner query.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=argparse.SUPPRESS if suppress_defaults else 20,
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS if suppress_defaults else False,
        help="Print JSON instead of TSV.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gradlab leaders",
        description="Query W&B summaries projected from verified R2 run receipts.",
    )
    add_common_args(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    runs = subparsers.add_parser(
        "runs", help="Rank completed training runs from W&B scientific metrics."
    )
    add_common_args(runs, suppress_defaults=True)
    runs.add_argument("--min-seeds", type=int, default=1)
    runs.add_argument(
        "--objective-key",
        action="append",
        default=[],
        help=(
            "Final W&B metric to rank; may be repeated."
        ),
    )
    runs.set_defaults(func=cmd_runs)

    checkpoints = subparsers.add_parser(
        "checkpoints",
        help="Rank exact checkpoint evaluation evidence.",
    )
    add_common_args(checkpoints, suppress_defaults=True)
    checkpoints.set_defaults(func=cmd_checkpoints)
    return parser


def cmd_runs(args: argparse.Namespace) -> int:
    scores = (
        score
        for run in wandb_runs(
            project=args.project,
            goal=args.goal,
            extra_filter=run_objective_filter(run_query_objective_keys(args)),
            order=RUN_PRIMARY_ORDER,
        )
        if (score := run_score(run, objective_keys=run_query_objective_keys(args))) is not None
    )
    if args.reward_shape:
        scores = (score for score in scores if score.reward_shape == args.reward_shape)
    ranked_rows = rank_run_leaders(scores, min_seeds=max(1, int(args.min_seeds)))[
        : max(0, int(args.limit))
    ]
    if args.json:
        print_json(ranked_rows)
    else:
        print_run_leaders(ranked_rows)
    return 0


def cmd_checkpoints(args: argparse.Namespace) -> int:
    candidates = (
        leader
        for run in wandb_runs(
            project=args.project,
            goal=args.goal,
            extra_filter=checkpoint_summary_filter(),
            order=CHECKPOINT_PRIMARY_ORDER,
        )
        if (leader := checkpoint_leader(run)) is not None
    )
    if args.reward_shape:
        candidates = (
            leader for leader in candidates if leader.reward_shape == args.reward_shape
        )
    ranked = rank_checkpoint_leaders(candidates)[: max(0, int(args.limit))]
    if args.json:
        print_json(ranked)
    else:
        print_checkpoint_leaders(ranked)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
