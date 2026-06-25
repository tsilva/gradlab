from __future__ import annotations

import argparse

from rlab.env import EnvConfig


REWARD_MODE_CHOICES = ("auto", "baseline", "bounded", "additive", "score", "native")


def add_env_config_args(
    parser: argparse.ArgumentParser,
    *,
    max_steps_default: int,
    defaults: EnvConfig | None = None,
) -> None:
    defaults = defaults or EnvConfig()
    parser.add_argument("--game", default=defaults.game)
    parser.add_argument("--state", default=defaults.state)
    parser.add_argument("--states", default=",".join(defaults.states))
    parser.add_argument("--state-probs", default=",".join(str(prob) for prob in defaults.state_probs))
    parser.add_argument(
        "--task-conditioning",
        action=argparse.BooleanOptionalAction,
        default=defaults.task_conditioning,
    )
    parser.add_argument(
        "--task-conditioning-info-vars",
        default=",".join(defaults.task_conditioning_info_vars),
        help="Comma-separated info keys used to map task-conditioned one-hot vectors.",
    )
    parser.add_argument(
        "--task-conditioning-info-values",
        default=";".join(
            ",".join(str(item) for item in row)
            for row in defaults.task_conditioning_info_values
        ),
        help=(
            "Semicolon-separated info-value rows for task conditioning, "
            "for example '0,0;0,1'. Omit when values can be derived from states."
        ),
    )
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--max-pool-frames", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--sticky-action-prob",
        type=float,
        default=defaults.sticky_action_prob,
        help="Probability of replaying the previous high-level action; 0 disables sticky actions.",
    )
    parser.add_argument("--max-steps", type=int, default=max_steps_default)
    parser.add_argument("--observation-size", type=int, default=defaults.observation_size)
    parser.add_argument(
        "--hud-crop-top",
        type=int,
        default=defaults.hud_crop_top,
        help="Crop this many pixels from the top of raw frames before grayscale resize.",
    )
    parser.add_argument("--obs-resize-algorithm", default=defaults.obs_resize_algorithm)
    parser.add_argument("--use-retro-reward", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--clip-rewards", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--reward-mode",
        choices=REWARD_MODE_CHOICES,
        default=defaults.reward_mode,
    )
    parser.add_argument("--progress-reward-cap", type=float, default=30.0)
    parser.add_argument("--progress-reward-scale", type=float, default=1.0)
    parser.add_argument("--terminal-reward", type=float, default=50.0)
    parser.add_argument("--reward-scale", type=float, default=10.0)
    parser.add_argument("--time-penalty", type=float, default=0.0)
    parser.add_argument("--death-penalty", type=float, default=25.0)
    parser.add_argument("--completion-reward", type=float, default=0.0)
    parser.add_argument("--score-progress-clipped", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--no-progress-timeout-steps", type=int, default=0)
    parser.add_argument("--no-progress-min-delta", type=int, default=0)
    parser.add_argument(
        "--completion-x-threshold",
        type=int,
        default=defaults.completion_x_threshold,
        help="Deprecated no-op; level completion is detected from stable-retro level changes.",
    )
    parser.add_argument(
        "--done-on-info-json",
        default="",
        help=(
            "JSON object mapping done-reason names to [key_or_keys, op], for example "
            '\'{"life_loss":["lives","decrease"],"level_change":[["levelHi","levelLo"],"change"]}\''
        ),
    )
    parser.add_argument("--action-set", default=defaults.action_set)
