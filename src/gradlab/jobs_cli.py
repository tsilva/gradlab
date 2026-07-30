from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gradlab.cli_parser import ExactArgumentParser
from gradlab.job_queue import (
    DEFAULT_IDLE_SECONDS,
    JobStore,
    ensure_flusher,
    run_flusher,
)


def _parser() -> argparse.ArgumentParser:
    parser = ExactArgumentParser(
        prog="gradlab jobs",
        description="Inspect and flush GradLab's durable per-user background jobs.",
    )
    parser.add_argument(
        "--queue-dir",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    list_parser = subparsers.add_parser("list", help="list recent jobs")
    list_parser.add_argument(
        "--state",
        action="append",
        default=[],
        help="filter by a job state; may be repeated",
    )
    list_parser.add_argument("--limit", type=int, default=100)

    show_parser = subparsers.add_parser("show", help="show one job and its history")
    show_parser.add_argument("job_id")

    flush_parser = subparsers.add_parser("flush", help="flush jobs in the foreground")
    flush_parser.add_argument("--idle-seconds", type=float, default=DEFAULT_IDLE_SECONDS)

    retry_parser = subparsers.add_parser(
        "retry",
        help="retry a failed, blocked, or canceled job",
    )
    retry_parser.add_argument("job_id")

    cancel_parser = subparsers.add_parser("cancel", help="cancel a queued or running job")
    cancel_parser.add_argument("job_id")

    worker_parser = subparsers.add_parser("_worker", help=argparse.SUPPRESS)
    worker_parser.add_argument("--generation", required=True)
    worker_parser.add_argument("--idle-seconds", type=float, default=DEFAULT_IDLE_SECONDS)
    return parser


def _print(value: Any) -> None:
    print(json.dumps(value, sort_keys=True, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    store = JobStore(args.queue_dir)
    store.init()

    if args.action == "list":
        jobs = store.jobs(states=args.state or None, limit=args.limit)
        _print(
            {
                "items": jobs,
                "worker": store.worker_state(),
            }
        )
        return 0

    if args.action == "show":
        job = store.job(args.job_id)
        if job is None:
            raise SystemExit(f"unknown job: {args.job_id}")
        _print(
            {
                "job": job,
                "subjects": store.subjects(args.job_id),
                "events": store.events(args.job_id),
                "worker": store.worker_state(),
            }
        )
        return 0

    if args.action == "flush":
        return run_flusher(store, idle_seconds=args.idle_seconds)

    if args.action == "_worker":
        return run_flusher(
            store,
            generation=args.generation,
            idle_seconds=args.idle_seconds,
        )

    if args.action == "retry":
        job = store.retry(args.job_id)
        worker = ensure_flusher(store)
        _print({"job": job, "worker": worker.to_dict()})
        return 0

    if args.action == "cancel":
        job = store.request_cancel(args.job_id)
        worker = ensure_flusher(store) if store.has_unfinished() else None
        _print(
            {
                "job": job,
                "worker": None if worker is None else worker.to_dict(),
            }
        )
        return 0

    raise AssertionError(f"unhandled jobs action: {args.action}")


if __name__ == "__main__":
    raise SystemExit(main())
