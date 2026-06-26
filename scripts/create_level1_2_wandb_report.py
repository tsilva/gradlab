#!/usr/bin/env python3
"""Create a W&B report for the Level1-1 + Level1-2 completion goal."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

import wandb
from dotenv import load_dotenv
from rlab.metric_names import (
    EVAL_DONE_LEVEL_CHANGE_FROM_RATE_MIN,
    ROLLOUT_ADVANTAGE,
    ROLLOUT_VALUE_PRED,
    THROUGHPUT_LOOP_FPS,
    THROUGHPUT_ROLLOUT_FPS,
    TRAIN_DONE_ALL,
    TRAIN_DONE_MAX_STEPS,
    TRAIN_DONE_UNCLASSIFIED,
    TRAIN_OUTCOME_LEVEL_CHANGE_FROM_RATE_MEAN,
    TRAIN_OUTCOME_LEVEL_CHANGE_FROM_RATE_MIN,
    TRAIN_REWARD_COMPONENT_ROOT,
    TRAIN_REWARD_SHARE_ROOT,
    eval_done_value_metric,
    stat_metric,
    train_outcome_value_metric,
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
DEFAULT_LEVEL_SPECS = ("Level1-2=0-1", "Level1-1=0-0")
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


@dataclass(frozen=True)
class LevelSpec:
    label: str
    train_from_value: object

    @property
    def count_metric(self) -> str:
        return train_outcome_value_metric("level_change", "from", self.train_from_value)

    @property
    def window_rate_metric(self) -> str:
        return f"{self.count_metric}/attempt_window/rate"

    @property
    def eval_rate_metric(self) -> str:
        return f"{eval_done_value_metric('level_change', 'from', self.label)}/rate"


def parse_train_from_value(raw_value: str) -> object:
    value = raw_value.strip()
    if not value:
        raise argparse.ArgumentTypeError("level train-from value cannot be empty")
    if "," in value:
        parts = tuple(part.strip() for part in value.split(",") if part.strip())
        if not parts:
            raise argparse.ArgumentTypeError("level train-from value cannot be empty")
        return parts
    if "-" in value:
        parts = tuple(part.strip() for part in value.split("-") if part.strip())
        if len(parts) > 1:
            return parts
    return value


def parse_level_spec(raw_spec: str) -> LevelSpec:
    label, separator, raw_value = raw_spec.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError(
            f"invalid level spec {raw_spec!r}; expected Label=done_from_value",
        )
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("level label cannot be empty")
    return LevelSpec(label=label, train_from_value=parse_train_from_value(raw_value))


def parse_level_specs(raw_specs: list[str] | None) -> list[LevelSpec]:
    specs = [parse_level_spec(raw_spec) for raw_spec in (raw_specs or DEFAULT_LEVEL_SPECS)]
    if len(specs) < 2:
        raise argparse.ArgumentTypeError("at least two --level specs are required")
    return specs


def normalize_report_url(url: str) -> str:
    parts = urlsplit(url)
    marker = "/reports/"
    if marker not in parts.path:
        return url
    prefix, report_slug = parts.path.split(marker, 1)
    path = f"{prefix}{marker}{quote(report_slug, safe='-_.~+=')}"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


def runset_filters(run_state: str | None) -> str:
    if run_state is None:
        return ""
    return f"State = '{run_state}'"


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
    runs = list(
        api.runs(
            project_path,
            filters=wandb_run_filters(query, run_state),
            order="-created_at",
            per_page=200,
        )
    )
    return query.strip(), runset_filters(run_state), runs


def run_colors(runs: list[object]) -> dict[str, str]:
    runs.sort(key=lambda run: (run.name or "", run.id))
    return {run.id: RUN_COLORS[index % len(RUN_COLORS)] for index, run in enumerate(runs)}


def numeric_summary_value(run: object, metric: str) -> float | None:
    value = dict(run.summary).get(metric)
    if not isinstance(value, (int, float)):
        return None
    return float(value)


def format_rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def format_step(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{int(value):,}"


def markdown_cell(value: object) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


def run_url(entity: str, project: str, run: object) -> str:
    url = getattr(run, "url", None)
    if isinstance(url, str) and url:
        return url
    return f"https://wandb.ai/{entity}/{project}/runs/{run.id}"


def policy_selection_markdown(
    *,
    entity: str,
    project: str,
    runs: list[object],
    level_specs: list[LevelSpec],
) -> str:
    published_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    rows: list[dict[str, object]] = []
    for run in runs:
        rates = [numeric_summary_value(run, spec.window_rate_metric) for spec in level_specs]
        numeric_rates = [rate for rate in rates if rate is not None]
        min_rate = min(numeric_rates) if len(numeric_rates) == len(level_specs) else None
        mean_rate = (
            sum(numeric_rates) / len(numeric_rates)
            if len(numeric_rates) == len(level_specs)
            else None
        )
        rows.append(
            {
                "run": run,
                "rates": rates,
                "min": min_rate,
                "mean": mean_rate,
                "step": numeric_summary_value(run, "global_step"),
            }
        )

    rows.sort(
        key=lambda row: (
            row["min"] is not None,
            float(row["min"] or -1.0),
            float(row["step"] or -1.0),
        ),
        reverse=True,
    )

    headers = [
        "Rank",
        "Run",
        "Min",
        *[spec.label for spec in level_specs],
        "Mean",
        "Step",
    ]
    lines = [
        "### Current minimum clearance leaderboard",
        "",
        (
            f"Computed at publish time from active W&B run summaries ({published_at}). "
            "Rank by the `Min` column; it is the lower of the per-level clearance rates."
        ),
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for index, row in enumerate(rows, start=1):
        run = row["run"]
        name = markdown_cell(run.name or run.id)
        rates = row["rates"]
        values = [
            str(index),
            f"[{name}]({run_url(entity, project, run)})",
            format_rate(row["min"] if isinstance(row["min"], float) else None),
            *[format_rate(rate) for rate in rates],
            format_rate(row["mean"] if isinstance(row["mean"], float) else None),
            format_step(row["step"] if isinstance(row["step"], float) else None),
        ]
        lines.append("| " + " | ".join(values) + " |")

    if not rows:
        empty_values = ["n/a", "No active runs matched the report filters."]
        empty_values.extend(["n/a"] * (len(headers) - 2))
        lines.append("| " + " | ".join(empty_values) + " |")
    return "\n".join(lines)


def backfill_min_rate_summary(runs: list[object], level_specs: list[LevelSpec]) -> int:
    updated = 0
    for run in runs:
        rates = [numeric_summary_value(run, spec.window_rate_metric) for spec in level_specs]
        if any(rate is None for rate in rates):
            continue
        numeric_rates = [rate for rate in rates if rate is not None]
        run.summary[TRAIN_OUTCOME_LEVEL_CHANGE_FROM_RATE_MIN] = min(numeric_rates)
        run.summary[TRAIN_OUTCOME_LEVEL_CHANGE_FROM_RATE_MEAN] = sum(numeric_rates) / len(
            numeric_rates,
        )
        run.summary.update()
        run.update()
        updated += 1
    return updated


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
    level_specs: list[LevelSpec],
    scoped_runs: list[object],
    run_colors: dict[str, str] | None = None,
) -> wr.Report:
    primary_level = level_specs[0]
    secondary_level = level_specs[1]
    level_count_metrics = [spec.count_metric for spec in level_specs]
    level_window_rate_metrics = [spec.window_rate_metric for spec in level_specs]
    eval_level_rate_metrics = [
        EVAL_DONE_LEVEL_CHANGE_FROM_RATE_MIN,
        *[spec.eval_rate_metric for spec in level_specs],
    ]

    runset = wr.Runset(
        entity=entity,
        project=project,
        name="Level1+2 policy candidates",
        query=query.strip(),
        filters=filters,
        order=[
            wr.OrderBy(
                wr.SummaryMetric(
                    TRAIN_OUTCOME_LEVEL_CHANGE_FROM_RATE_MIN,
                ),
                ascending=False,
            ),
        ],
        pinned_columns=[
            "Name",
            "State",
            "group",
            *[f"summary.{metric}" for metric in level_window_rate_metrics],
            f"summary.{TRAIN_OUTCOME_LEVEL_CHANGE_FROM_RATE_MIN}",
            f"summary.{TRAIN_OUTCOME_LEVEL_CHANGE_FROM_RATE_MEAN}",
            *[f"summary.{metric}" for metric in level_count_metrics],
            f"summary.{EVAL_DONE_LEVEL_CHANGE_FROM_RATE_MIN}",
            f"summary.{TRAIN_DONE_ALL}",
            f"summary.{TRAIN_DONE_LEVEL_CHANGE}",
            f"summary.{TRAIN_DONE_LIFE_LOSS}",
            f"summary.{TRAIN_DONE_UNCLASSIFIED}",
            f"summary.{TRAIN_DONE_MAX_STEPS}",
            f"summary.{TRAIN_REWARD_COMPONENT_ROOT}/shaped/mean",
            f"summary.{TRAIN_REWARD_COMPONENT_ROOT}/prog_x/max",
            f"summary.{TRAIN_REWARD_COMPONENT_ROOT}/score/max",
            f"summary.{TRAIN_REWARD_COMPONENT_ROOT}/death/nonzero_rate",
            f"summary.{TRAIN_REWARD_SHARE_ROOT}/prog_x",
            f"summary.{TRAIN_REWARD_SHARE_ROOT}/death",
            "summary.train/adv_norm/mode",
            "summary.global_step",
            "summary.rollout/ep_rew_mean",
            "summary.rollout/ep_len_mean",
            f"summary.{stat_metric(ROLLOUT_VALUE_PRED, 'abs_mean')}",
            f"summary.{stat_metric(ROLLOUT_ADVANTAGE, 'abs_mean')}",
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
            "the current running W&B runset, minimum per-level clearance rate, "
            "per-level 100-attempt completion windows, reward attribution, "
            "rollout diagnostics, and PPO health."
        ),
        width="fluid",
        blocks=[
            wr.H1("Goal"),
            wr.MarkdownBlock(
                f"""
                Training monitor for active SuperMarioBros-NES runs. During training,
                start with the per-level 100-attempt window rates:
                `{primary_level.window_rate_metric}` and `{secondary_level.window_rate_metric}`.
                The top metric is the current minimum per-level clearance rate: the lower of the
                per-level rates for each active run. New training code also logs
                `{TRAIN_OUTCOME_LEVEL_CHANGE_FROM_RATE_MIN}` directly. The runset is softcoded to W&B
                `State = 'running'` by default, so it follows current active runs instead of a fixed
                batch. For the Reward component share panel, select one run in the run table before
                reading the component fractions.
                """,
            ),
            wr.PanelGrid(
                runsets=[runset],
                custom_run_colors=run_colors or {},
                panels=[
                    section_panel(
                        "1. Policy selection",
                        (
                            "Start here. The winner is the run with the highest minimum "
                            "per-level clearance rate, not the highest average."
                        ),
                        y=0,
                    ),
                    wr.MarkdownPanel(
                        markdown=policy_selection_markdown(
                            entity=entity,
                            project=project,
                            runs=scoped_runs,
                            level_specs=level_specs,
                        ),
                        layout=wr.Layout(x=0, y=3, w=24, h=5),
                    ),
                    scalar(
                        f"{primary_level.label} window rate",
                        primary_level.window_rate_metric,
                        x=0,
                        y=8,
                        w=12,
                    ),
                    scalar(
                        f"{secondary_level.label} window rate",
                        secondary_level.window_rate_metric,
                        x=12,
                        y=8,
                        w=12,
                    ),
                    line(
                        "Per-level clearance rates (bottleneck is the lower trace)",
                        [
                            *level_window_rate_metrics,
                            TRAIN_OUTCOME_LEVEL_CHANGE_FROM_RATE_MIN,
                        ],
                        x=0,
                        y=12,
                        w=24,
                        h=8,
                        ymin=0,
                        ymax=1.05,
                    ),
                    line(
                        "Per-level clear counts",
                        level_count_metrics,
                        x=0,
                        y=20,
                        w=12,
                        h=7,
                        ymin=0,
                    ),
                    line(
                        "Eval balanced completion if present",
                        eval_level_rate_metrics,
                        x=12,
                        y=20,
                        w=12,
                        h=7,
                        ymin=0,
                        ymax=1.05,
                    ),
                    line(
                        "Done reason counts",
                        [
                            TRAIN_DONE_LEVEL_CHANGE,
                            TRAIN_DONE_LIFE_LOSS,
                            TRAIN_DONE_MAX_STEPS,
                            TRAIN_DONE_UNCLASSIFIED,
                        ],
                        x=0,
                        y=27,
                        w=24,
                        h=7,
                        ymin=0,
                    ),
                    section_panel(
                        "2. Reward and behavior diagnosis",
                        (
                            "Use these panels to see whether reward is coming from real progress "
                            "or from side effects. Reward shares use absolute magnitude, so "
                            "negative penalties are visible instead of canceling out."
                        ),
                        y=35,
                    ),
                    line(
                        "Episode reward mean",
                        ["rollout/ep_rew_mean"],
                        x=0,
                        y=38,
                        w=12,
                        h=7,
                    ),
                    line(
                        "Episode length mean",
                        ["rollout/ep_len_mean"],
                        x=12,
                        y=38,
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
                        y=45,
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
                        y=53,
                        w=12,
                        h=7,
                    ),
                    line(
                        "Progress reward mean",
                        [
                            f"{TRAIN_REWARD_COMPONENT_ROOT}/prog_x/mean",
                        ],
                        x=12,
                        y=53,
                        w=12,
                        h=7,
                    ),
                    line(
                        "Death event rate",
                        [
                            f"{TRAIN_REWARD_COMPONENT_ROOT}/death/nonzero_rate",
                        ],
                        x=0,
                        y=60,
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
                        y=60,
                        w=12,
                        h=7,
                    ),
                    section_panel(
                        "3. PPO update health",
                        (
                            "Use this section to catch destructive updates: KL/clip spikes, "
                            "value-function collapse, entropy trends, and rollout-buffer drift."
                        ),
                        y=67,
                    ),
                    line("Approx KL", ["train/approx_kl"], x=0, y=70),
                    line("Clip fraction", ["train/clip_fraction"], x=12, y=70, ymin=0),
                    line("Entropy loss", ["train/entropy_loss"], x=0, y=77),
                    line(
                        "Explained variance",
                        ["train/explained_variance"],
                        x=12,
                        y=77,
                        ymin=-0.1,
                        ymax=1.05,
                    ),
                    line("Value loss", ["train/value_loss"], x=0, y=84, ymin=0),
                    line(
                        "Rollout value and advantage magnitude",
                        [
                            stat_metric(ROLLOUT_VALUE_PRED, "abs_mean"),
                            stat_metric(ROLLOUT_ADVANTAGE, "abs_mean"),
                        ],
                        x=12,
                        y=84,
                        w=12,
                        h=7,
                        ymin=0,
                    ),
                    line(
                        "Per-task advantage std before normalization",
                        [
                            "train/adv/task0/std_pre",
                            "train/adv/task1/std_pre",
                        ],
                        x=0,
                        y=91,
                        w=12,
                        h=7,
                        ymin=0,
                    ),
                    line(
                        "Per-task advantage std after normalization",
                        [
                            "train/adv/task0/std_post",
                            "train/adv/task1/std_post",
                        ],
                        x=12,
                        y=91,
                        w=12,
                        h=7,
                        ymin=0,
                    ),
                    section_panel(
                        "4. Throughput",
                        (
                            "Use throughput after policy quality checks. Low throughput can explain "
                            "slow learning, but it is not itself a policy-selection metric."
                        ),
                        y=98,
                    ),
                    line("Loop FPS", [THROUGHPUT_LOOP_FPS], x=0, y=101, ymin=0),
                    line("Rollout FPS", [THROUGHPUT_ROLLOUT_FPS], x=12, y=101, ymin=0),
                ],
            ),
            wr.H2("Decision Rule"),
            wr.MarkdownBlock(
                f"""
                During active training, rank runs by the current minimum per-level clearance rate.
                That value is the lower of the per-level 100-attempt clearance rates;
                the mean can look good while one level is failing, so it is secondary. New runs log
                `{TRAIN_OUTCOME_LEVEL_CHANGE_FROM_RATE_MIN}` directly for history panels. Once
                out-of-process checkpoint eval exists, rank by
                `{EVAL_DONE_LEVEL_CHANGE_FROM_RATE_MIN}`, then mean reward, then max x-position.
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
    parser.add_argument(
        "--level",
        action="append",
        dest="levels",
        metavar="LABEL=FROM_VALUE",
        help=(
            "Per-level training metric mapping. Repeat for each level. "
            "Defaults to Level1-2=0-1 and Level1-1=0-0."
        ),
    )
    parser.add_argument("--url", help="Existing W&B report URL to update in place")
    parser.add_argument("--draft", action="store_true")
    parser.add_argument(
        "--backfill-min-summary",
        action="store_true",
        help=(
            "Best-effort update of scoped run summaries with the current min/mean per-level "
            "window rates. Active W&B clients may overwrite this before training logs the metric."
        ),
    )
    parser.add_argument(
        "--no-run-colors",
        action="store_true",
        help="Do not assign deterministic report colors to the active runs.",
    )
    args = parser.parse_args()

    load_dotenv(Path(".env"))
    level_specs = parse_level_specs(args.levels)
    run_state = None if args.all_states else args.run_state
    query, filters, scoped_runs = report_scope(
        entity=args.entity,
        project=args.project,
        query=args.query,
        run_state=run_state,
    )
    if args.backfill_min_summary:
        updated = backfill_min_rate_summary(scoped_runs, level_specs)
        print(f"attempted_min_summary_backfill={updated}")
    colors = {} if args.no_run_colors else run_colors(scoped_runs)
    report = build_report(
        entity=args.entity,
        project=args.project,
        title=args.title,
        query=query,
        filters=filters,
        level_specs=level_specs,
        scoped_runs=scoped_runs,
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
