from __future__ import annotations

import argparse
from pathlib import Path

from gradlab.cli_parser import ExactArgumentParser
from gradlab.local_paths import default_runs_dir
from gradlab.model_sources import DEFAULT_PUBLIC_MODELS_BASE_URL, positional_model_source_arg
from gradlab.seeds import DEFAULT_EVAL_SEED, EVAL_SEED_START

PLAYBACK_DEVICE = "cpu"


def add_play_source_args(parser: argparse.ArgumentParser) -> None:
    def positional_play_source_arg(value: str) -> str:
        from gradlab.play_catalog import is_wandb_url

        if is_wandb_url(value):
            return value
        return positional_model_source_arg(value)

    parser.add_argument(
        "artifact_ref",
        nargs="?",
        type=positional_play_source_arg,
        help=(
            "W&B run URL, immutable public checkpoint manifest, or Hugging Face model ref. "
            "Use --model for a local checkpoint or --run for an gradlab public run."
        ),
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Open the highest-step published checkpoint of the newest catalog run with published checkpoints, paused.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Local gradlab policy path. The artifact must have model.json and recipe.json sidecars.",
    )
    parser.add_argument(
        "--recording",
        type=Path,
        help="Open a local .trj episode archive without loading a Policy or environment.",
    )
    parser.add_argument(
        "--recipe",
        help=(
            "Play the newest completed local run for a built-in <goal-path>/<recipe> "
            "reference or recipe YAML."
        ),
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=default_runs_dir(),
        help="Local run root searched by --recipe; defaults to ~/.config/gradlab/runs.",
    )
    parser.add_argument(
        "--rom-path",
        type=Path,
        help=(
            "Use a provider-compatible raw .nes ROM, or a local ViZDoom IWAD as a "
            "visible counterfactual when it differs from training."
        ),
    )
    parser.add_argument(
        "--run",
        help=(
            "Immutable gradlab run ID or exact public checkpoint manifest URL. A run ID "
            "resolves its promoted checkpoint, or its highest-step final checkpoint when "
            "no promotion exists, without W&B or private R2 credentials."
        ),
    )
    parser.add_argument(
        "--public-models-base-url",
        default=DEFAULT_PUBLIC_MODELS_BASE_URL,
        help="Public models bucket URL. Defaults to gradlab's checked-in public endpoint.",
    )
    parser.add_argument(
        "--public-model-root",
        default=str(default_runs_dir() / "public_models"),
        help="Local cache for public run checkpoints.",
    )
    parser.set_defaults(
        hf_revision=None,
        hf_model_root=str(default_runs_dir() / "hf_models"),
    )


def nonnegative_int_arg(value: str) -> int:
    parsed = int(value)
    if parsed < 0 or parsed > 65535:
        raise argparse.ArgumentTypeError("must be in [0, 65535]")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = ExactArgumentParser(
        prog="gradlab play",
        description=(
            "Browse repository goals and control-plane runs, then inspect a public "
            "checkpoint in the interactive web player"
        ),
    )
    add_play_source_args(parser)
    parser.add_argument(
        "--episodes", type=int, default=0, help="Number of episodes; use 0 to run forever"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_EVAL_SEED,
        help=(
            "Base playback seed. The default lives in the eval/play-reserved seed "
            f"range >= {EVAL_SEED_START}; overrides must stay in that range."
        ),
    )
    parser.add_argument(
        "--device",
        default=PLAYBACK_DEVICE,
        choices=[PLAYBACK_DEVICE],
        help="Inference device; playback currently runs on CPU only.",
    )
    parser.add_argument(
        "--env-provider",
        help=(
            "Run the artifact's unchanged evaluation contract through an equivalent provider. "
            "The provider must support the recorded game and constructor arguments."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Playback display FPS (default: 30; 0: unlimited). Policy inference is unthrottled.",
    )
    parser.add_argument(
        "--port",
        type=nonnegative_int_arg,
        default=0,
        help="Loopback dashboard port; use 0 to select an available port automatically.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Print the play and stats dashboard URLs without opening a dedicated browser.",
    )
    parser.add_argument(
        "--hotreload",
        action="store_true",
        help="Use live frontend assets in a source checkout instead of compiled player assets.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Start paused for interactive transition debugging (now the default).",
    )
    parser.add_argument(
        "--continuous-play",
        action="store_true",
        help=(
            "Ignore the recipe's task success, failure, stall, and step-limit boundaries. "
            "This is a semantic deviation intended only for continuous interactive play."
        ),
    )
    parser.add_argument(
        "--resume-cell",
        help=(
            "Resume a cell-graph representative by node ID. The model must have "
            "been exported with state_archive.export.snapshots=retained."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable model-download and player-startup progress bars.",
    )
    return parser


