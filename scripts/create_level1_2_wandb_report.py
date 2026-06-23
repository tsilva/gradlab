#!/usr/bin/env python3
"""Create a W&B report for the Level1-1 + Level1-2 completion goal."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

from dotenv import load_dotenv
from stable_retro_ppo.metric_names import (
    EVAL_OUTCOME_RATE,
    EVAL_PROGRESS_X_MAX,
    EVAL_REWARD_MEAN,
    EVAL_STATE_MEAN_RATE,
    EVAL_STATE_MIN_RATE,
    TRAIN_OUTCOME_RATE,
    TRAIN_OUTCOME_STATE_MEAN_RATE,
    TRAIN_OUTCOME_STATE_MIN_RATE,
    eval_state_prefix,
    train_outcome_state_prefix,
)

try:
    from wandb_workspaces.reports import v2 as wr
except ImportError as exc:  # pragma: no cover - operator-facing dependency hint
    raise SystemExit(
        "Missing W&B report support. Install it with: "
        "UV_CACHE_DIR=.uv-cache uv pip install --python .venv/bin/python "
        "'wandb[workspaces]==0.22.3'"
    ) from exc


DEFAULT_ENTITY = "tsilva"
DEFAULT_PROJECT = "SuperMarioBros-NES"
DEFAULT_QUERY = r"(level1[-_]1[-_]2|level1[-_]1[-_]1[-_]2|mario-level1-1-1-2-100of100)"
DEFAULT_TITLE = "Mario Level1-1 + Level1-2 100/100 Completion Search"


def normalize_report_url(url: str) -> str:
    parts = urlsplit(url)
    marker = "/reports/"
    if marker not in parts.path:
        return url
    prefix, report_slug = parts.path.split(marker, 1)
    path = f"{prefix}{marker}{quote(report_slug, safe='-_.~+=')}"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


def line(
    title: str,
    metrics: list[str],
    *,
    x: int,
    y: int,
    w: int = 12,
    h: int = 6,
    ymin: float | None = None,
    ymax: float | None = None,
) -> wr.LinePlot:
    return wr.LinePlot(
        title=title,
        x="global_step",
        y=metrics,
        range_y=(ymin, ymax),
        smoothing_type="none",
        max_runs_to_show=50,
        legend_position="east",
        legend_template="${run:displayName}",
        layout=wr.Layout(x=x, y=y, w=w, h=h),
    )


def scalar(
    title: str,
    metric: str,
    *,
    x: int,
    y: int,
    w: int = 6,
    h: int = 4,
) -> wr.ScalarChart:
    return wr.ScalarChart(
        title=title,
        metric=metric,
        groupby_aggfunc="max",
        layout=wr.Layout(x=x, y=y, w=w, h=h),
    )


def build_report(
    *,
    entity: str,
    project: str,
    title: str,
    query: str,
) -> wr.Report:
    runset = wr.Runset(
        entity=entity,
        project=project,
        name="Level1+2 policy candidates",
        query=query,
        order=[wr.OrderBy(wr.SummaryMetric(EVAL_STATE_MIN_RATE), ascending=False)],
        pinned_columns=[
            "Name",
            "State",
            "group",
            f"summary.{EVAL_STATE_MIN_RATE}",
            f"summary.{EVAL_STATE_MEAN_RATE}",
            f"summary.{eval_state_prefix('Level1-1')}/outcome/rate",
            f"summary.{eval_state_prefix('Level1-2')}/outcome/rate",
            f"summary.{EVAL_OUTCOME_RATE}",
            f"summary.{TRAIN_OUTCOME_STATE_MIN_RATE}",
            f"summary.{TRAIN_OUTCOME_STATE_MEAN_RATE}",
            f"summary.{TRAIN_OUTCOME_RATE}",
            f"summary.{train_outcome_state_prefix('Level1-1')}/rate",
            f"summary.{train_outcome_state_prefix('Level1-2')}/rate",
            "summary.global_step",
            "config.learning_rate",
            "config.ent_coef_final",
            "config.clip_range",
            "config.target_kl",
            "config.n_steps",
            "config.states",
            "config.state_probs",
        ],
    )

    return wr.Report(
        entity=entity,
        project=project,
        title=title,
        description=(
            "Tracks the search for one SuperMarioBros-NES policy that completes both "
            "Level1-1 and Level1-2 with 100/100 eval completions. The north-star metric "
            f"is {EVAL_STATE_MIN_RATE}, the minimum of the per-level eval completion rates. "
            "Mean completion rate and the matching training rolling-window aggregates are "
            "tracked as secondary balance metrics."
        ),
        width="fluid",
        blocks=[
            wr.H1("Goal"),
            wr.MarkdownBlock(
                """
                Find a single policy whose robust eval reaches `eval/state/Level1-1/outcome/rate = 1.0`
                and `eval/state/Level1-2/outcome/rate = 1.0`. Use `eval/state/min_rate` as the
                promotion gate: when the eval batch is 100 episodes per level, `1.0` means 100/100
                completions on both levels, not just a high pooled or mean average. Track
                `eval/state/mean_rate` beside it to see average two-level performance.
                """,
            ),
            wr.PanelGrid(
                runsets=[runset],
                panels=[
                    scalar("Best min completion rate", EVAL_STATE_MIN_RATE, x=0, y=0),
                    scalar("Best mean completion rate", EVAL_STATE_MEAN_RATE, x=6, y=0),
                    scalar(
                        "Best Level1-1 completion",
                        f"{eval_state_prefix('Level1-1')}/outcome/rate",
                        x=12,
                        y=0,
                    ),
                    scalar(
                        "Best Level1-2 completion",
                        f"{eval_state_prefix('Level1-2')}/outcome/rate",
                        x=18,
                        y=0,
                    ),
                    line(
                        "North star: minimum per-level eval completion rate",
                        [EVAL_STATE_MIN_RATE],
                        x=0,
                        y=4,
                        w=24,
                        h=7,
                        ymin=0,
                        ymax=1.05,
                    ),
                    line(
                        "Mean per-level eval completion rate",
                        [EVAL_STATE_MEAN_RATE],
                        x=0,
                        y=11,
                        w=24,
                        h=7,
                        ymin=0,
                        ymax=1.05,
                    ),
                    line(
                        "Per-level eval completion rates",
                        [
                            f"{eval_state_prefix('Level1-1')}/outcome/rate",
                            f"{eval_state_prefix('Level1-2')}/outcome/rate",
                        ],
                        x=0,
                        y=18,
                        w=24,
                        h=7,
                        ymin=0,
                        ymax=1.05,
                    ),
                    line(
                        "Training completion windows",
                        [
                            TRAIN_OUTCOME_STATE_MIN_RATE,
                            TRAIN_OUTCOME_STATE_MEAN_RATE,
                            TRAIN_OUTCOME_RATE,
                            f"{train_outcome_state_prefix('Level1-1')}/rate",
                            f"{train_outcome_state_prefix('Level1-2')}/rate",
                        ],
                        x=0,
                        y=25,
                        w=24,
                        h=7,
                        ymin=0,
                        ymax=1.05,
                    ),
                    line("Max x-position by eval", [EVAL_PROGRESS_X_MAX], x=0, y=32),
                    line("Eval reward mean", [EVAL_REWARD_MEAN], x=12, y=32),
                    line("Approx KL", ["train/approx_kl"], x=0, y=38),
                    line("Clip fraction", ["train/clip_fraction"], x=12, y=38, ymin=0),
                    line(
                        "Explained variance",
                        ["train/explained_variance"],
                        x=0,
                        y=44,
                        ymin=-0.1,
                        ymax=1.05,
                    ),
                    line("Value loss", ["train/value_loss"], x=12, y=44, ymin=0),
                ],
            ),
            wr.H2("Decision Rule"),
            wr.MarkdownBlock(
                """
                Promote only when `eval/state/min_rate` reaches `1.0` on the intended robust
                eval profile. Use `eval/state/mean_rate` to summarize average two-level
                performance. Use `train/outcome/state/min_rate` and
                `train/outcome/state/mean_rate` as live training indicators only; if the mean
                or pooled metric is high while the min metric is lower, the policy is still
                specializing or regressing on one level.
                """,
            ),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--url", help="Existing W&B report URL to update in place")
    parser.add_argument("--draft", action="store_true")
    args = parser.parse_args()

    load_dotenv(Path(".env"))
    report = build_report(
        entity=args.entity,
        project=args.project,
        title=args.title,
        query=args.query,
    )
    if args.url:
        existing = wr.Report.from_url(normalize_report_url(args.url))
        existing.title = report.title
        existing.description = report.description
        existing.width = report.width
        existing.blocks = report.blocks
        report = existing
    report = report.save(draft=args.draft)
    print(report.url)


if __name__ == "__main__":
    main()
