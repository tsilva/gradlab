from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence


def _eval_argv(argv: Sequence[str]) -> list[str] | None:
    if argv and argv[0] == "run":
        return list(argv[1:])
    parser = argparse.ArgumentParser(
        prog="rlab eval",
        description="Run an ad-hoc evaluation locally.",
    )
    commands = parser.add_subparsers(dest="command", metavar="<command>")
    commands.add_parser("run", help="Evaluate one model against an environment contract.")
    if argv and argv[0] in {"-h", "--help"}:
        parser.parse_args(["--help"])
    if not argv:
        parser.print_help()
        return None
    parser.error(f"unknown eval command: {argv[0]}")


COMMANDS: dict[str, tuple[str, str]] = {
    "train": ("train a checked-in recipe locally", "rlab.local_train"),
    "experiment": ("launch and observe dstack training experiments", "rlab.experiment_cli"),
    "eval": ("run a direct local evaluation", "rlab.eval"),
    "play": ("browse and inspect W&B, public-run, local, or Hugging Face models", "rlab.play"),
    "import-roms": ("import ROMs into the installed rlab runtime", "rlab.import_roms"),
    "benchmark": ("run gated local-smoke and throughput profiles", "rlab.benchmark"),
    "validate": (
        "validate checked-in YAML experiments, recipes, benchmarks, and ops configs",
        "rlab.config_validation",
    ),
    "env": ("list, inspect, and preflight environment providers", "rlab.env_cli"),
    "rom": ("provision, verify, and warm immutable ROM assets", "rlab.rom_cli"),
    "dataset": (
        "record, inspect, verify, migrate, and publish gameplay datasets",
        "rlab.dataset_cli",
    ),
    "leaders": ("query accepted runs and promoted checkpoints", "rlab.wandb_leaders"),
    "reports": ("plan, synchronize, and verify declarative W&B reports", "rlab.wandb_reports"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlab",
        description="Unified command surface for rlab training, eval, playback, and ops.",
        epilog=(
            "Research: train, experiment, eval, play, validate.  Environments: env, rom, "
            "import-roms, benchmark.  Datasets: dataset.  Results: leaders, reports."
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
    if command not in COMMANDS:
        parser.error(f"unknown command: {command}")
    _help, module_name = COMMANDS[command]
    command_argv = _eval_argv(argv_list[1:]) if command == "eval" else argv_list[1:]
    if command_argv is None:
        return 2
    result = importlib.import_module(module_name).main(list(command_argv))
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
