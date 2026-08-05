from __future__ import annotations

import argparse
import importlib
import sys

from gradlab.cli_parser import ExactArgumentParser


COMMANDS: dict[str, tuple[str, str]] = {
    "train": ("train a checked-in recipe locally", "gradlab.local_train"),
    "experiment": ("launch and observe dstack training experiments", "gradlab.experiment_cli"),
    "eval": ("run a direct local evaluation", "gradlab.eval"),
    "play": ("browse and inspect W&B, public-run, local, or Hugging Face models", "gradlab.play"),
    "benchmark": ("run gated local-smoke and throughput profiles", "gradlab.benchmark"),
    "validate": (
        "validate checked-in YAML experiments, recipes, benchmarks, and ops configs",
        "gradlab.config_validation",
    ),
    "env": ("list, inspect, and preflight environment providers", "gradlab.env_cli"),
    "rom": ("import, provision, verify, and warm ROM assets", "gradlab.rom_cli"),
    "dataset": (
        "record, inspect, verify, and publish gameplay datasets",
        "gradlab.dataset_cli",
    ),
    "jobs": ("inspect and flush durable local background jobs", "gradlab.jobs_cli"),
    "leaders": ("query accepted runs and promoted checkpoints", "gradlab.wandb_leaders"),
    "reports": ("plan, synchronize, and verify declarative W&B reports", "gradlab.wandb_reports"),
    "workspaces": (
        "plan, synchronize, and verify declarative W&B workspace views",
        "gradlab.wandb_workspaces",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = ExactArgumentParser(
        prog="gradlab",
        description="Unified command surface for gradlab training, eval, playback, and ops.",
        epilog=(
            "Research: train, experiment, eval, play, validate.  Environments: env, rom, "
            "benchmark.  Datasets: dataset.  Operations: jobs.  Results: leaders, reports, "
            "workspaces."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    for name in COMMANDS:
        help_text, _module_name = COMMANDS[name]
        subparser = subparsers.add_parser(name, help=help_text, add_help=False)
        subparser.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    argv_list = list(sys.argv[1:] if argv is None else argv)
    if not argv_list or argv_list[0] in {"-h", "--help"}:
        parser.print_help()
        return 0 if argv_list else 2
    command = argv_list[0]
    route = COMMANDS.get(command)
    if route is None:
        parser.error(f"unknown command: {command}")
    _help, module_name = route
    result = importlib.import_module(module_name).main(list(argv_list[1:]))
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
