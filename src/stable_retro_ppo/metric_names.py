from __future__ import annotations

import re

GLOBAL_STEP = "global_step"

THROUGHPUT_ROLLOUT_FPS = "throughput/rollout_fps"
THROUGHPUT_LOOP_FPS = "throughput/loop_fps"

ROLLOUT_VALUE_PRED = "rollout/value_pred"
ROLLOUT_VALUE_PRED_HIST = "rollout/value_pred/hist"
ROLLOUT_ADVANTAGE = "rollout/advantage"
ROLLOUT_ADVANTAGE_HIST = "rollout/advantage/hist"

TRAIN_COMPLETION_EVENTS_ROLLOUT = "train/events/completion/rollout"
TRAIN_COMPLETION_EVENTS_ROLLING_MEAN = "train/events/completion/rolling_mean"
TRAIN_COMPLETION_EVENTS_TOTAL = "train/events/completion/total"

TRAIN_OUTCOME_RATE = "train/outcome/rate"
TRAIN_OUTCOME_WINDOW = "train/outcome/window"
TRAIN_OUTCOME_COMPLETIONS = "train/outcome/completions"
TRAIN_OUTCOME_TERMINALS = "train/outcome/terminals"
TRAIN_OUTCOME_STATE_MIN_RATE = "train/outcome/state/min_rate"
TRAIN_OUTCOME_STATE_MEAN_RATE = "train/outcome/state/mean_rate"

EVAL_STATE_ROOT = "eval/state"
EVAL_STATE_MIN_RATE = "eval/state/min_rate"
EVAL_STATE_MEAN_RATE = "eval/state/mean_rate"

EVAL_REWARD_MEAN = "eval/reward/mean"
EVAL_REWARD_STD = "eval/reward/std"
EVAL_REWARD_MAX = "eval/reward/max"
EVAL_PROGRESS_X_MEAN = "eval/progress/x/mean"
EVAL_PROGRESS_X_MAX = "eval/progress/x/max"
EVAL_PROGRESS_LEVEL_X_MEAN = "eval/progress/level_x/mean"
EVAL_PROGRESS_LEVEL_X_MAX = "eval/progress/level_x/max"
EVAL_OUTCOME_COMPLETIONS = "eval/outcome/completions"
EVAL_OUTCOME_RATE = "eval/outcome/rate"
EVAL_DEATH_COUNT = "eval/death/count"
EVAL_DEATH_RATE = "eval/death/rate"
EVAL_DEATH_X_HIST = "eval/death/x_hist"
EVAL_BEST_REWARD = "eval/best/reward"
EVAL_BEST_X = "eval/best/x"
EVAL_BEST_VIDEO = "eval/best/video"
EVAL_CHECKPOINT_STEP = "eval/checkpoint/step"
EVAL_CHECKPOINT_ARTIFACT = "eval/checkpoint/artifact"
EVAL_CONFIG_HUD_CROP_TOP = "eval/config/hud_crop_top"


def metric_path_segment(value: object) -> str:
    segment = str(value).strip()
    segment = re.sub(r"[^A-Za-z0-9_.-]+", "_", segment)
    return segment.strip("_") or "unknown"


def stat_metric(prefix: str, stat: str) -> str:
    return f"{prefix}/{stat}"


def train_outcome_state_prefix(state: object) -> str:
    return f"train/outcome/state/{metric_path_segment(state)}"


def eval_state_prefix(state: object) -> str:
    return f"{EVAL_STATE_ROOT}/{metric_path_segment(state)}"
