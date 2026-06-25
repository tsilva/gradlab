#!/usr/bin/env python3
"""Create a W&B report for the Level1-1 + Level1-2 completion goal."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

import wandb
from dotenv import load_dotenv
from rlab.metric_names import (
    THROUGHPUT_LOOP_FPS,
    THROUGHPUT_ROLLOUT_FPS,
    TRAIN_DONE_ALL,
    TRAIN_DONE_MAX_STEPS,
    TRAIN_DONE_UNCLASSIFIED,
    TRAIN_REWARD_COMPONENT_ROOT,
    TRAIN_REWARD_SHARE_ROOT,
    train_done_value_metric,
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
DEFAULT_QUERY = ""
DEFAULT_RUN_STATE = "running"
DEFAULT_TITLE = "Mario Active Training Monitor"
RUN_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#005f73",
    "#ca6702",
)


TRAIN_DONE_LEVEL_CHANGE = "train/done/level_change"
TRAIN_DONE_LIFE_LOSS = "train/done/life_loss"
LEVEL1_1_VALUE = (0, 0)
LEVEL1_2_VALUE = (0, 1)
TRAIN_DONE_LEVEL1_1_CLEAR = train_done_value_metric("level_change", "from", LEVEL1_1_VALUE)
TRAIN_DONE_LEVEL1_2_CLEAR = train_done_value_metric("level_change", "from", LEVEL1_2_VALUE)


def normalize_report_url(url: str) -> str:
    parts = urlsplit(url)
    marker = "/reports/"
    if marker not in parts.path:
        return url
    prefix, report_slug = parts.path.split(marker, 1)
    path = f"{prefix}{marker}{quote(report_slug, safe='-_.~+=')}"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


def runset_query(query: str, run_state: str | None) -> str:
    return query.strip()


def runset_filters(run_state: str | None) -> str:
    if run_state is None:
        return ""
    return f"State = '{run_state}'"


def filter_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def wandb_run_filters(query: str, run_state: str | None = None) -> dict[str, object]:
    parts: list[dict[str, object]] = []
    if run_state is not None:
        parts.append({"state": run_state})
    query = query.strip()
    if query:
        parts.append({"display_name": {"$regex": query}})
    if not parts:
        return {}
    if len(parts) == 1:
        return parts[0]
    return {"$and": parts}


def report_scope(
    *,
    entity: str,
    project: str,
    query: str,
    run_state: str | None,
) -> tuple[str, str, list[object]]:
    api = wandb.Api()
    project_path = f"{entity}/{project}"

    if run_state is None:
        runs = list(
            api.runs(
                project_path,
                filters=wandb_run_filters(query),
                order="-created_at",
                per_page=200,
            )
        )
        return query.strip(), "", runs

    runs = list(
        api.runs(
            project_path,
            filters=wandb_run_filters(query, run_state),
            order="-created_at",
            per_page=200,
        )
    )
    if runs:
        return query.strip(), runset_filters(run_state), runs

    latest_runs = list(
        api.runs(
            project_path,
            filters=wandb_run_filters(query),
            order="-created_at",
            per_page=50,
        )
    )
    if not latest_runs:
        return query.strip(), runset_filters(run_state), []

    latest_group = getattr(latest_runs[0], "group", None)
    if not latest_group:
        return query.strip(), "", latest_runs[:1]

    group_filter = f"Group = '{filter_value(str(latest_group))}'"
    group_runs = list(
        api.runs(
            project_path,
            filters={"group": latest_group},
            order="-created_at",
            per_page=200,
        )
    )
    return query.strip(), group_filter, group_runs


def run_colors(runs: list[object]) -> dict[str, str]:
    runs.sort(key=lambda run: (run.name or "", run.id))
    return {run.id: RUN_COLORS[index % len(RUN_COLORS)] for index, run in enumerate(runs)}


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


def section_panel(title: str, body: str, *, y: int) -> wr.MarkdownPanel:
    return wr.MarkdownPanel(
        markdown=f"### {title}\n{body.strip()}",
        layout=wr.Layout(x=0, y=y, w=24, h=3),
    )


def build_report(
    *,
    entity: str,
    project: str,
    title: str,
    query: str,
    filters: str,
    run_colors: dict[str, str] | None = None,
) -> wr.Report:
    runset = wr.Runset(
        entity=entity,
        project=project,
        name="Level1+2 policy candidates",
        query=query.strip(),
        filters=filters,
        order=[
            wr.OrderBy(
                wr.SummaryMetric(
                    TRAIN_DONE_LEVEL1_2_CLEAR,
                ),
                ascending=False,
            ),
        ],
        pinned_columns=[
            "Name",
            "State",
            "group",
            f"summary.{TRAIN_DONE_ALL}",
            f"summary.{TRAIN_DONE_LEVEL_CHANGE}",
            f"summary.{TRAIN_DONE_LEVEL1_2_CLEAR}",
            f"summary.{TRAIN_DONE_LEVEL1_1_CLEAR}",
            f"summary.{TRAIN_DONE_LIFE_LOSS}",
            f"summary.{TRAIN_DONE_UNCLASSIFIED}",
            f"summary.{TRAIN_DONE_MAX_STEPS}",
            f"summary.{TRAIN_REWARD_COMPONENT_ROOT}/shaped/mean",
            f"summary.{TRAIN_REWARD_COMPONENT_ROOT}/prog_x/max",
            f"summary.{TRAIN_REWARD_COMPONENT_ROOT}/score/max",
            f"summary.{TRAIN_REWARD_COMPONENT_ROOT}/death/nonzero_rate",
            "summary.global_step",
            "summary.rollout/ep_rew_mean",
            "summary.rollout/ep_len_mean",
            "summary.train/approx_kl",
            "summary.train/clip_fraction",
            "summary.train/entropy_loss",
            "summary.train/explained_variance",
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
            "Training-focused view for active SuperMarioBros-NES runs. It prioritizes "
            "Level1-2 learning, per-level balance, "
            "done volume, done-reason mix, sampling denominators, "
            "and PPO health."
        ),
        width="fluid",
        blocks=[
            wr.H1("Goal"),
            wr.MarkdownBlock(
                """
                Training monitor for active SuperMarioBros-NES runs. During training,
                read the per-level done counts and configured done reasons first. Eval panels
                are intentionally omitted from this live view; use a separate eval report once
                checkpoint eval jobs exist. Check max-step and unclassified done counts before comparing
                runs. For the Reward component share panel, select one run
                in the run table before reading the component fractions.
                """,
            ),
            wr.PanelGrid(
                runsets=[runset],
                custom_run_colors=run_colors or {},
                panels=[
                    section_panel(
                        "1. Policy selection",
                        (
                            "Start here. Prefer runs that improve Level1-2 without sacrificing "
                            "Level1-1, then confirm candidates with separate robust eval."
                        ),
                        y=0,
                    ),
                    scalar(
                        "Level1-2 clears",
                        TRAIN_DONE_LEVEL1_2_CLEAR,
                        x=0,
                        y=3,
                    ),
                    scalar(
                        "All level changes",
                        TRAIN_DONE_LEVEL_CHANGE,
                        x=6,
                        y=3,
                    ),
                    scalar(
                        "Level1-1 clears",
                        TRAIN_DONE_LEVEL1_1_CLEAR,
                        x=12,
                        y=3,
                    ),
                    scalar(
                        "All done",
                        TRAIN_DONE_ALL,
                        x=18,
                        y=3,
                    ),
                    line(
                        "Level1-2 clears from native vars",
                        [
                            TRAIN_DONE_LEVEL1_2_CLEAR,
                        ],
                        x=0,
                        y=7,
                        w=12,
                        h=7,
                        ymin=0,
                    ),
                    line(
                        "Level1-1 clears from native vars",
                        [
                            TRAIN_DONE_LEVEL1_1_CLEAR,
                        ],
                        x=0,
                        y=14,
                        w=12,
                        h=7,
                        ymin=0,
                    ),
                    line(
                        "All done events",
                        [
                            TRAIN_DONE_ALL,
                        ],
                        x=0,
                        y=21,
                        w=12,
                        h=7,
                        ymin=0,
                    ),
                    line(
                        "Level-change from value counts",
                        [
                            TRAIN_DONE_LEVEL1_2_CLEAR,
                            TRAIN_DONE_LEVEL1_1_CLEAR,
                        ],
                        x=12,
                        y=21,
                        w=12,
                        h=7,
                        ymin=0,
                    ),
                    line(
                        "Done reason counts",
                        [
                            TRAIN_DONE_LEVEL_CHANGE,
                            TRAIN_DONE_LIFE_LOSS,
                            TRAIN_DONE_MAX_STEPS,
                            TRAIN_DONE_UNCLASSIFIED,
                        ],
                        x=12,
                        y=28,
                        w=12,
                        h=7,
                        ymin=0,
                    ),
                    line(
                        "Max-step and unclassified done",
                        [
                            TRAIN_DONE_MAX_STEPS,
                            TRAIN_DONE_UNCLASSIFIED,
                        ],
                        x=0,
                        y=42,
                        w=12,
                        h=7,
                        ymin=0,
                    ),
                    section_panel(
                        "2. Reward and behavior diagnosis",
                        (
                            "Use these panels to see whether reward is coming from real progress "
                            "or from side effects. Select one run before reading reward shares."
                        ),
                        y=49,
                    ),
                    line(
                        "Episode reward mean",
                        ["rollout/ep_rew_mean"],
                        x=0,
                        y=52,
                        w=12,
                        h=7,
                    ),
                    line(
                        "Episode length mean",
                        ["rollout/ep_len_mean"],
                        x=12,
                        y=52,
                        w=12,
                        h=7,
                        ymin=0,
                    ),
                    line(
                        "Reward component share",
                        [
                            f"{TRAIN_REWARD_SHARE_ROOT}/prog_x",
                            f"{TRAIN_REWARD_SHARE_ROOT}/score",
                            f"{TRAIN_REWARD_SHARE_ROOT}/death",
                            f"{TRAIN_REWARD_SHARE_ROOT}/done",
                            f"{TRAIN_REWARD_SHARE_ROOT}/time",
                            f"{TRAIN_REWARD_SHARE_ROOT}/native",
                        ],
                        x=0,
                        y=59,
                        w=24,
                        h=8,
                        ymin=0,
                        ymax=1.05,
                    ),
                    line(
                        "Shaped reward mean",
                        [
                            f"{TRAIN_REWARD_COMPONENT_ROOT}/shaped/mean",
                        ],
                        x=0,
                        y=67,
                        w=12,
                        h=7,
                    ),
                    line(
                        "Progress reward mean",
                        [
                            f"{TRAIN_REWARD_COMPONENT_ROOT}/prog_x/mean",
                        ],
                        x=12,
                        y=67,
                        w=12,
                        h=7,
                    ),
                    line(
                        "Death event rate",
                        [
                            f"{TRAIN_REWARD_COMPONENT_ROOT}/death/nonzero_rate",
                        ],
                        x=0,
                        y=74,
                        w=12,
                        h=7,
                        ymin=0,
                        ymax=1.05,
                    ),
                    line(
                        "Progress reward spike",
                        [
                            f"{TRAIN_REWARD_COMPONENT_ROOT}/prog_x/max",
                        ],
                        x=12,
                        y=74,
                        w=12,
                        h=7,
                    ),
                    section_panel(
                        "3. PPO update health",
                        (
                            "Use this section to catch destructive updates: KL/clip spikes, "
                            "value-function collapse, and entropy loss trends."
                        ),
                        y=81,
                    ),
                    line("Approx KL", ["train/approx_kl"], x=0, y=84),
                    line("Clip fraction", ["train/clip_fraction"], x=12, y=84, ymin=0),
                    line("Entropy loss", ["train/entropy_loss"], x=0, y=91),
                    line(
                        "Explained variance",
                        ["train/explained_variance"],
                        x=12,
                        y=91,
                        ymin=-0.1,
                        ymax=1.05,
                    ),
                    line("Value loss", ["train/value_loss"], x=0, y=98, ymin=0),
                    section_panel(
                        "4. Throughput",
                        (
                            "Use throughput after policy quality checks. Low throughput can explain "
                            "slow learning, but it is not itself a policy-selection metric."
                        ),
                        y=105,
                    ),
                    line("Loop FPS", [THROUGHPUT_LOOP_FPS], x=0, y=108, ymin=0),
                    line("Rollout FPS", [THROUGHPUT_ROLLOUT_FPS], x=12, y=108, ymin=0),
                ],
            ),
            wr.H2("Decision Rule"),
            wr.MarkdownBlock(
                """
                During training, compare Level1-2 completion volume and window rate across active runs.
                Treat pooled completion as secondary: a high pooled rate still fails this goal if
                Level1-2 remains weak. Promotion still requires separate robust eval later.
                """,
            ),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help="Optional W&B report search query. Defaults to all runs matching --run-state.",
    )
    parser.add_argument(
        "--run-state",
        default=DEFAULT_RUN_STATE,
        help="Default W&B run state filter for the report runset. Use --all-states for history.",
    )
    parser.add_argument(
        "--all-states",
        action="store_true",
        help="Include finished, failed, crashed, and running runs instead of only active runs.",
    )
    parser.add_argument("--url", help="Existing W&B report URL to update in place")
    parser.add_argument("--draft", action="store_true")
    parser.add_argument(
        "--no-run-colors",
        action="store_true",
        help="Do not assign deterministic report colors to the active runs.",
    )
    args = parser.parse_args()

    load_dotenv(Path(".env"))
    run_state = None if args.all_states else args.run_state
    query, filters, scoped_runs = report_scope(
        entity=args.entity,
        project=args.project,
        query=args.query,
        run_state=run_state,
    )
    colors = {} if args.no_run_colors else run_colors(scoped_runs)
    report = build_report(
        entity=args.entity,
        project=args.project,
        title=args.title,
        query=query,
        filters=filters,
        run_colors=colors,
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
