from __future__ import annotations

import argparse
import importlib
import sys


COMMANDS: dict[str, tuple[str, str]] = {
    "train": ("train a checked-in recipe locally", "rlab.local_train"),
    "experiment": ("launch and observe dstack training experiments", "rlab.experiment_cli"),
    "eval": ("run a direct local evaluation", "rlab.eval"),
    "play": ("browse and inspect W&B, public-run, local, or Hugging Face models", "rlab.play"),
    "benchmark": ("run gated local-smoke and throughput profiles", "rlab.benchmark"),
    "validate": (
        "validate checked-in YAML experiments, recipes, benchmarks, and ops configs",
        "rlab.config_validation",
    ),
    "env": ("list, inspect, and preflight environment providers", "rlab.env_cli"),
    "rom": ("import, provision, verify, and warm ROM assets", "rlab.rom_cli"),
    "dataset": (
        "record, inspect, verify, migrate, and publish gameplay datasets",
        "rlab.dataset_cli",
    ),
    "leaders": ("query accepted runs and promoted checkpoints", "rlab.wandb_leaders"),
    "reports": ("plan, synchronize, and verify declarative W&B reports", "rlab.wandb_reports"),
}

COMPATIBILITY_COMMANDS: dict[str, tuple[str, str]] = {
    "import-roms": ("import ROMs into the installed rlab runtime", "rlab.import_roms"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlab",
        description="Unified command surface for rlab training, eval, playback, and ops.",
        epilog=(
            "Research: train, experiment, eval, play, validate.  Environments: env, rom, "
            "benchmark.  Datasets: dataset.  Results: leaders, reports."
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
    route = COMMANDS.get(command) or COMPATIBILITY_COMMANDS.get(command)
    if route is None:
        parser.error(f"unknown command: {command}")
    _help, module_name = route
    result = importlib.import_module(module_name).main(list(argv_list[1:]))
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
