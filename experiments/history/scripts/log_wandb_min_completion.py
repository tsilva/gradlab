#!/usr/bin/env python3
"""Log derived min per-level completion metrics to existing W&B runs."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import wandb
from dotenv import load_dotenv
from rlab.metric_names import (
    GLOBAL_STEP,
    TRAIN_DONE_LEVEL_CHANGE_FROM_RATE_MEAN,
    TRAIN_DONE_LEVEL_CHANGE_FROM_RATE_MIN,
)


DEFAULT_ENTITY = "tsilva"
DEFAULT_PROJECT = "SuperMarioBros-NES"
DEFAULT_RUN_STATE = "running"
DEFAULT_LEVEL_RATE_METRICS = (
    "train/done/level_change/from/0-1/ep_window/rate",
    "train/done/level_change/from/0-0/ep_window/rate",
)


def wandb_run_filters(query: str, run_state: str | None) -> dict[str, object]:
    parts: list[dict[str, object]] = []
    if run_state is not None:
        parts.append({"state": run_state})
    if query.strip():
        parts.append({"display_name": {"$regex": query.strip()}})
    if not parts:
        return {}
    if len(parts) == 1:
        return parts[0]
    return {"$and": parts}


def numeric_summary(summary: dict[str, object], key: str) -> float | None:
    value = summary.get(key)
    if not isinstance(value, (int, float)):
        return None
    return float(value)


def current_payload(run: object, level_metrics: tuple[str, ...]) -> dict[str, float] | None:
    summary = dict(run.summary)
    rates = [numeric_summary(summary, metric) for metric in level_metrics]
    if any(rate is None for rate in rates):
        return None

    step = numeric_summary(summary, GLOBAL_STEP)
    if step is None:
        step = numeric_summary(summary, "_step")
    if step is None:
        return None

    numeric_rates = [rate for rate in rates if rate is not None]
    return {
        GLOBAL_STEP: step,
        TRAIN_DONE_LEVEL_CHANGE_FROM_RATE_MIN: min(numeric_rates),
        TRAIN_DONE_LEVEL_CHANGE_FROM_RATE_MEAN: sum(numeric_rates) / len(numeric_rates),
    }


def log_payload(
    *,
    entity: str,
    project: str,
    run_id: str,
    payload: dict[str, float],
    dry_run: bool,
) -> None:
    if dry_run:
        print(f"{run_id}: {payload}")
        return

    run = wandb.init(
        entity=entity,
        project=project,
        id=run_id,
        resume="allow",
        reinit=True,
        settings=wandb.Settings(_disable_stats=True, console="off"),
    )
    try:
        run.log(payload, step=int(payload[GLOBAL_STEP]))
    finally:
        run.finish()


def update_once(args: argparse.Namespace, level_metrics: tuple[str, ...]) -> int:
    api = wandb.Api()
    runs = list(
        api.runs(
            f"{args.entity}/{args.project}",
            filters=wandb_run_filters(args.query, None if args.all_states else args.run_state),
            order="-created_at",
            per_page=args.limit,
        ),
    )
    updated = 0
    skipped = 0
    for run in runs:
        payload = current_payload(run, level_metrics)
        if payload is None:
            skipped += 1
            continue
        log_payload(
            entity=args.entity,
            project=args.project,
            run_id=run.id,
            payload=payload,
            dry_run=args.dry_run,
        )
        updated += 1
    print(f"updated={updated} skipped={skipped}")
    return updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--query", default="")
    parser.add_argument("--run-state", default=DEFAULT_RUN_STATE)
    parser.add_argument("--all-states", action="store_true")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument(
        "--level-rate-metric",
        action="append",
        dest="level_rate_metrics",
        help="Per-level rate metric to include in the min. Repeat at least twice.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=60.0)
    args = parser.parse_args()

    load_dotenv(Path(".env"))
    level_metrics = tuple(args.level_rate_metrics or DEFAULT_LEVEL_RATE_METRICS)
    if len(level_metrics) < 2:
        raise SystemExit("at least two --level-rate-metric values are required")

    while True:
        update_once(args, level_metrics)
        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
