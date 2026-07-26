from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources
from numbers import Real
from pathlib import Path
from typing import Any, Mapping


METRICS_SCHEMA_VERSION = 7
TRAIN_GLOBAL_STEP = "train/global_step"
EVAL_CHECKPOINT_STEP = "eval/checkpoint_step"
ORCHESTRATION_EVENT_SEQ = "orchestration/event_seq"
ORCHESTRATION_EVENT_ID = "orchestration/event_id"
ORCHESTRATION_QUEUE_DEPTH = "orchestration/outbox/queue_depth"
ORCHESTRATION_OLDEST_UNPUBLISHED_SECONDS = "orchestration/outbox/oldest_unpublished_seconds"
ORCHESTRATION_INGRESS_RATE = "orchestration/outbox/ingress_rate"
ORCHESTRATION_PUBLISH_RATE = "orchestration/outbox/publish_rate"
ORCHESTRATION_PUBLICATION_CAPACITY_RATIO = "orchestration/outbox/publication_capacity_ratio"
ORCHESTRATION_LOCAL_HIGH_WATER = "orchestration/outbox/local_high_water"
ORCHESTRATION_R2_HIGH_WATER = "orchestration/outbox/r2_high_water"
ORCHESTRATION_WANDB_HIGH_WATER = "orchestration/outbox/wandb_high_water"
ORCHESTRATION_WANDB_REMOTE_HIGH_WATER = "orchestration/outbox/wandb_remote_high_water"
ORCHESTRATION_WANDB_REMOTE_VISIBLE_LAG_SECONDS = (
    "orchestration/outbox/wandb_remote_visible_lag_seconds"
)
ORCHESTRATION_CHECKPOINT_BACKLOG = "orchestration/checkpoint/backlog"
ORCHESTRATION_PENDING_EVALS = "orchestration/eval/pending"
ORCHESTRATION_RESULT_TO_STOP_SECONDS = "orchestration/eval/result_to_stop_seconds"
ORCHESTRATION_IDLE_GPU_TAIL_SECONDS = "orchestration/drain/idle_gpu_tail_seconds"
ORCHESTRATION_SCRATCH_USED_FRACTION = "orchestration/scratch/used_fraction"

TRAIN_EPISODE_RETURN_SHAPED_MEAN = "train/episode/return/shaped/mean"
TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN = "train/episode/return/shaped/from/target/mean"
TRAIN_EPISODE_LENGTH_MEAN = "train/episode/length/mean"

TRAIN_SNAPSHOT_CURRICULUM_ROOT = "train/curriculum/snapshot"
TRAIN_SNAPSHOT_ARCHIVE_CELL_COUNT = f"{TRAIN_SNAPSHOT_CURRICULUM_ROOT}/archive/cell/count"
TRAIN_SNAPSHOT_ARCHIVE_SNAPSHOT_COUNT = f"{TRAIN_SNAPSHOT_CURRICULUM_ROOT}/archive/snapshot/count"
TRAIN_SNAPSHOT_ADMISSION_CANDIDATE_COUNT = (
    f"{TRAIN_SNAPSHOT_CURRICULUM_ROOT}/admission/candidate/count"
)
TRAIN_SNAPSHOT_ADMISSION_ACCEPTED_COUNT = (
    f"{TRAIN_SNAPSHOT_CURRICULUM_ROOT}/admission/accepted/count"
)
TRAIN_SNAPSHOT_ARCHIVE_EVICTED_COUNT = f"{TRAIN_SNAPSHOT_CURRICULUM_ROOT}/archive/evicted/count"
TRAIN_SNAPSHOT_CAPTURE_CALL_COUNT = f"{TRAIN_SNAPSHOT_CURRICULUM_ROOT}/capture/call/count"
TRAIN_SNAPSHOT_RESET_EPISODE_COUNT = f"{TRAIN_SNAPSHOT_CURRICULUM_ROOT}/reset/episode/count"
TRAIN_SNAPSHOT_RESET_FORCED_BOUNDARY_COUNT = (
    f"{TRAIN_SNAPSHOT_CURRICULUM_ROOT}/reset/forced_boundary/count"
)
TRAIN_SNAPSHOT_FEEDBACK_TRAJECTORY_COUNT = (
    f"{TRAIN_SNAPSHOT_CURRICULUM_ROOT}/feedback/trajectory/count"
)
TRAIN_SNAPSHOT_TRANSITION_SHARE = f"{TRAIN_SNAPSHOT_CURRICULUM_ROOT}/transition/share"
TRAIN_SNAPSHOT_SAMPLING_PROBABILITY_MAX = (
    f"{TRAIN_SNAPSHOT_CURRICULUM_ROOT}/sampling/probability/max"
)
TRAIN_SNAPSHOT_SAMPLING_EFFECTIVE_CELL_COUNT = (
    f"{TRAIN_SNAPSHOT_CURRICULUM_ROOT}/sampling/effective_cell/count"
)
TRAIN_SNAPSHOT_CAPTURE_SECONDS = f"{TRAIN_SNAPSHOT_CURRICULUM_ROOT}/capture/seconds"
TRAIN_SNAPSHOT_RESET_SECONDS = f"{TRAIN_SNAPSHOT_CURRICULUM_ROOT}/reset/seconds"

TRAIN_OUTCOME_TERMINAL_COUNT = "train/outcome/terminal/count"
TRAIN_OUTCOME_SUCCESS_ROOT = "train/outcome/success"
TRAIN_OUTCOME_SUCCESS_CURRENT_RATE_MIN = f"{TRAIN_OUTCOME_SUCCESS_ROOT}/current/rate/min"
TRAIN_OUTCOME_SUCCESS_CURRENT_RATE_MEAN = f"{TRAIN_OUTCOME_SUCCESS_ROOT}/current/rate/mean"
TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MIN = f"{TRAIN_OUTCOME_SUCCESS_ROOT}/window_100/rate/min"
TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MEAN = f"{TRAIN_OUTCOME_SUCCESS_ROOT}/window_100/rate/mean"
TRAIN_OUTCOME_SUCCESS_START_COVERAGE_RATE = f"{TRAIN_OUTCOME_SUCCESS_ROOT}/start_coverage/rate"

TRAIN_EARLY_STOP_ROOT = "train/early_stop"
TRAIN_REWARD_ROOT = "train/reward"

TRAIN_ALGORITHM_ROOT = "train/algorithm"
TRAIN_ACTOR_CRITIC_ALGORITHMS = ("ppo", "a2c")
TRAIN_ALGORITHM_JERK_ROOT = f"{TRAIN_ALGORITHM_ROOT}/jerk"
TRAIN_ALGORITHM_JERK_RETAINED_COUNT = f"{TRAIN_ALGORITHM_JERK_ROOT}/retained/count"
TRAIN_ALGORITHM_JERK_BEST_RETURN_MEAN = f"{TRAIN_ALGORITHM_JERK_ROOT}/best/return_mean"
TRAIN_ALGORITHM_JERK_BEST_SEQUENCE_LENGTH = f"{TRAIN_ALGORITHM_JERK_ROOT}/best/sequence_length"
TRAIN_ALGORITHM_JERK_ARCHIVE_SELECTED_PREFIX_RETURN_MEAN = (
    f"{TRAIN_ALGORITHM_JERK_ROOT}/archive/selected_prefix_return_mean"
)
TRAIN_ALGORITHM_JERK_EXPLOIT_PROBABILITY = f"{TRAIN_ALGORITHM_JERK_ROOT}/exploit/probability"


def train_algorithm_root(algorithm_id: str) -> str:
    if algorithm_id not in TRAIN_ACTOR_CRITIC_ALGORITHMS:
        raise ValueError(f"unsupported actor-critic algorithm id: {algorithm_id}")
    return f"{TRAIN_ALGORITHM_ROOT}/{algorithm_id}"


def train_algorithm_metric(algorithm_id: str, suffix: str) -> str:
    return f"{train_algorithm_root(algorithm_id)}/{suffix}"


TRAIN_ALGORITHM_PPO_ROOT = train_algorithm_root("ppo")
TRAIN_ALGORITHM_A2C_ROOT = train_algorithm_root("a2c")
TRAIN_PPO_APPROX_KL = f"{TRAIN_ALGORITHM_PPO_ROOT}/update/approx_kl"
TRAIN_PPO_CLIP_FRACTION = f"{TRAIN_ALGORITHM_PPO_ROOT}/update/clip_fraction"
TRAIN_PPO_EXPLAINED_VARIANCE = f"{TRAIN_ALGORITHM_PPO_ROOT}/value/explained_variance"
TRAIN_PPO_VALUE_LOSS = f"{TRAIN_ALGORITHM_PPO_ROOT}/update/value_loss"
TRAIN_PPO_LEARNING_RATE = f"{TRAIN_ALGORITHM_PPO_ROOT}/update/learning_rate"
TRAIN_PPO_POLICY_ENTROPY = f"{TRAIN_ALGORITHM_PPO_ROOT}/policy/entropy"

TRAIN_A2C_EXPLAINED_VARIANCE = f"{TRAIN_ALGORITHM_A2C_ROOT}/value/explained_variance"
TRAIN_A2C_VALUE_LOSS = f"{TRAIN_ALGORITHM_A2C_ROOT}/update/value_loss"
TRAIN_A2C_LEARNING_RATE = f"{TRAIN_ALGORITHM_A2C_ROOT}/update/learning_rate"
TRAIN_A2C_POLICY_ENTROPY = f"{TRAIN_ALGORITHM_A2C_ROOT}/policy/entropy"

TRAIN_THROUGHPUT_ROOT = "train/throughput"
TRAIN_THROUGHPUT_LOOP_FPS = f"{TRAIN_THROUGHPUT_ROOT}/loop_fps"
TRAIN_THROUGHPUT_ROLLOUT_FPS = f"{TRAIN_THROUGHPUT_ROOT}/rollout_fps"
TRAIN_THROUGHPUT_ENV_STEP_FPS = f"{TRAIN_THROUGHPUT_ROOT}/env_step_fps"
TRAIN_THROUGHPUT_ROLLOUT_SECONDS = f"{TRAIN_THROUGHPUT_ROOT}/rollout_seconds"
TRAIN_THROUGHPUT_ENV_STEP_SECONDS = f"{TRAIN_THROUGHPUT_ROOT}/env_step_seconds"
TRAIN_THROUGHPUT_ROLLOUT_OVERHEAD_SECONDS = f"{TRAIN_THROUGHPUT_ROOT}/rollout_overhead_seconds"
TRAIN_THROUGHPUT_BETWEEN_ROLLOUTS_SECONDS = f"{TRAIN_THROUGHPUT_ROOT}/between_rollouts_seconds"

TRAIN_ARTIFACT_SAVE_SECONDS = "train/artifact/save/seconds"
TRAIN_ARTIFACT_UPLOAD_SECONDS = "train/artifact/upload/seconds"

EVAL_ROOT = "eval"
EVAL_PROTOCOLS = ("full",)
EVAL_FULL_ROOT = f"{EVAL_ROOT}/full"
EVAL_FULL_EPISODE_RETURN_MEAN = f"{EVAL_FULL_ROOT}/episode/return/mean"
EVAL_FULL_EPISODE_RETURN_BEST = f"{EVAL_FULL_ROOT}/episode/return/best"
EVAL_FULL_EPISODE_COUNT = f"{EVAL_FULL_ROOT}/episode/count"
EVAL_FULL_PROGRESS_X_MAX = f"{EVAL_FULL_ROOT}/progress/x/max"
EVAL_FULL_SUCCESS_RATE_MIN = f"{EVAL_FULL_ROOT}/outcome/success/rate/min"
EVAL_FULL_SUCCESS_RATE_MEAN = f"{EVAL_FULL_ROOT}/outcome/success/rate/mean"
EVAL_FULL_BY_START = f"{EVAL_FULL_ROOT}/by_start"
EVAL_FULL_CHECKPOINT_STEP = f"{EVAL_FULL_ROOT}/checkpoint/step"
EVAL_FULL_CHECKPOINT_ARTIFACT = f"{EVAL_FULL_ROOT}/checkpoint/artifact"
EVAL_FULL_DURATION_SECONDS = f"{EVAL_FULL_ROOT}/duration/seconds"
EVAL_ACCEPTANCE_PASS = "eval/acceptance/pass"
EVAL_ACCEPTANCE_EPISODES_PLANNED = "eval/acceptance/episodes/planned"
EVAL_ACCEPTANCE_EPISODES_COMPLETED = "eval/acceptance/episodes/completed"
EVAL_ACCEPTANCE_FAILURE_COUNT = "eval/acceptance/failure/count"
EVAL_ACCEPTANCE_DURATION_SECONDS = "eval/acceptance/duration/seconds"

LEADER_CHECKPOINT_SUCCESS_RATE_MIN = "leader/checkpoint/success_rate_min"
LEADER_CHECKPOINT_SUCCESS_RATE_MEAN = "leader/checkpoint/success_rate_mean"
LEADER_CHECKPOINT_ACCEPTANCE_PASS = "leader/checkpoint/acceptance_pass"
LEADER_CHECKPOINT_OBJECTIVE = "leader/checkpoint/objective"
LEADER_CHECKPOINT_RETURN_MEAN = "leader/checkpoint/return_mean"
LEADER_CHECKPOINT_BEST_RETURN = "leader/checkpoint/best_return"
LEADER_CHECKPOINT_RANK_VALUES = "leader/checkpoint/rank_values"
LEADER_CHECKPOINT_PROGRESS_MAX = "leader/checkpoint/progress_max"
LEADER_CHECKPOINT_STEP = "leader/checkpoint/step"
LEADER_CHECKPOINT_ARTIFACT_REF = "leader/checkpoint/artifact_ref"
LEADER_CHECKPOINT_EVAL_SOURCE = "leader/checkpoint/eval_source"
LEADER_CHECKPOINT_UPDATED_AT = "leader/checkpoint/updated_at"


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    description: str
    unit: str
    cadence: str
    storage: str


_METRIC_REGISTRY_START = "<!-- METRIC_REGISTRY_START -->"
_METRIC_REGISTRY_END = "<!-- METRIC_REGISTRY_END -->"
_METRIC_REGISTRY_HEADER = "| Metric or template | Meaning | Unit | Cadence | Surface |"
_METRIC_REGISTRY_SEPARATOR = "|---|---|---|---|---|"


def _metrics_markdown() -> str:
    source_document = Path(__file__).resolve().parents[2] / "METRICS.md"
    if source_document.is_file():
        return source_document.read_text(encoding="utf-8")
    return resources.files("rlab").joinpath("METRICS.md").read_text(encoding="utf-8")


def _load_metric_definitions() -> tuple[MetricDefinition, ...]:
    document = _metrics_markdown()
    try:
        registry = document.split(f"{_METRIC_REGISTRY_START}\n", 1)[1].split(
            f"\n{_METRIC_REGISTRY_END}", 1
        )[0]
    except IndexError as exc:
        raise RuntimeError("METRICS.md is missing its metric registry markers") from exc
    lines = registry.splitlines()
    if lines[:2] != [_METRIC_REGISTRY_HEADER, _METRIC_REGISTRY_SEPARATOR]:
        raise RuntimeError("METRICS.md has an invalid metric registry header")
    definitions: list[MetricDefinition] = []
    for line_number, line in enumerate(lines[2:], start=3):
        columns = line.removeprefix("| ").removesuffix(" |").split(" | ")
        if len(columns) != 5:
            raise RuntimeError(
                f"METRICS.md metric registry row {line_number} must have five columns"
            )
        name = columns[0]
        if len(name) < 3 or not name.startswith("`") or not name.endswith("`"):
            raise RuntimeError(
                f"METRICS.md metric registry row {line_number} must use a code metric name"
            )
        definitions.append(MetricDefinition(name[1:-1], *columns[1:]))
    names = [definition.name for definition in definitions]
    if not names or len(names) != len(set(names)):
        raise RuntimeError("METRICS.md metric registry must be non-empty and unique")
    return tuple(definitions)


METRIC_DEFINITIONS = _load_metric_definitions()


_SAFE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_PLACEHOLDER_PATTERNS = {
    "algorithm": "(?:ppo|a2c)",
    "protocol": "(?:full)",
    "reason": "[A-Za-z0-9_.-]+",
    "start": "[A-Za-z0-9_.-]+",
    "component": "[A-Za-z0-9_.-]+",
    "condition": "[A-Za-z0-9_.-]+",
    "signal": "[A-Za-z0-9_.-]+",
    "progress": "[A-Za-z0-9_.-]+",
}


def _definition_pattern(
    template: str, *, placeholders: Mapping[str, str] = _PLACEHOLDER_PATTERNS
) -> re.Pattern[str]:
    cursor = 0
    parts: list[str] = []
    for match in re.finditer(r"\{([a-z_]+)\}", template):
        parts.append(re.escape(template[cursor : match.start()]))
        parts.append(placeholders[match.group(1)])
        cursor = match.end()
    parts.append(re.escape(template[cursor:]))
    return re.compile("^" + "".join(parts) + "$")


_DEFINITION_PATTERNS = tuple(
    (definition, _definition_pattern(definition.name)) for definition in METRIC_DEFINITIONS
)


def metric_definition(name: str) -> MetricDefinition | None:
    for definition, pattern in _DEFINITION_PATTERNS:
        if pattern.fullmatch(name):
            return definition
    return None


def validate_metric_name(name: str) -> str:
    if metric_definition(name) is None:
        raise ValueError(f"unknown metric name: {name}")
    return name


def validate_metric_payload(payload: Mapping[str, Any]) -> None:
    for name in payload:
        validate_metric_name(str(name))


def metric_path_segment(value: object) -> str:
    segment = str(value).strip()
    if not segment or _SAFE_SEGMENT_RE.fullmatch(segment) is None:
        raise ValueError(f"metric dimension must match {_SAFE_SEGMENT_RE.pattern}: {value!r}")
    return segment


def metric_value_segment(value: object) -> str:
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("metric dimension sequence must not be empty")
        return "-".join(metric_path_segment(item) for item in value)
    return metric_path_segment(value)


def stat_metric(prefix: str, stat: str) -> str:
    return validate_metric_name(f"{prefix}/{metric_path_segment(stat)}")


def eval_metric(protocol: str, suffix: str) -> str:
    protocol = metric_path_segment(protocol)
    if protocol not in EVAL_PROTOCOLS:
        raise ValueError(f"unknown evaluation protocol: {protocol}")
    return validate_metric_name(f"{EVAL_ROOT}/{protocol}/{suffix.strip('/')}")


def train_outcome_reason_count_metric(reason: object) -> str:
    return validate_metric_name(f"train/outcome/reason/{metric_path_segment(reason)}/count")


def train_outcome_reason_window_rate_metric(reason: object) -> str:
    return validate_metric_name(
        f"train/outcome/reason/{metric_path_segment(reason)}/rate/window_100"
    )


def train_early_stop_metric(condition: object, suffix: str) -> str:
    return validate_metric_name(
        f"{TRAIN_EARLY_STOP_ROOT}/{metric_path_segment(condition)}/{suffix.strip('/')}"
    )


def train_success_from_metric(start: object, suffix: str) -> str:
    return validate_metric_name(
        f"{TRAIN_OUTCOME_SUCCESS_ROOT}/from/{metric_value_segment(start)}/{suffix}"
    )


def train_success_count_metric(start: object) -> str:
    return train_success_from_metric(start, "count")


def train_success_attempts_metric(start: object) -> str:
    return train_success_from_metric(start, "attempts")


def train_success_window_rate_metric(start: object) -> str:
    return train_success_from_metric(start, "rate/window_100")


def train_reward_component_metric(component: object, stat: str) -> str:
    return validate_metric_name(
        f"{TRAIN_REWARD_ROOT}/component/{metric_path_segment(component)}/{metric_path_segment(stat)}"
    )


def train_reward_signal_metric(signal: object, stat: str) -> str:
    return validate_metric_name(
        f"{TRAIN_REWARD_ROOT}/signal/{metric_path_segment(signal)}/{metric_path_segment(stat)}"
    )


def eval_success_from_rate_metric(protocol: str, start: object) -> str:
    return eval_metric(protocol, f"outcome/success/from/{metric_value_segment(start)}/rate")


def eval_success_rate_metric(protocol: str, stat: str) -> str:
    return eval_metric(protocol, f"outcome/success/rate/{metric_path_segment(stat)}")


def eval_reason_rate_metric(protocol: str, reason: object) -> str:
    return eval_metric(protocol, f"outcome/reason/{metric_path_segment(reason)}/rate")


def eval_progress_metric(
    protocol: str,
    progress: object,
    stat: str,
) -> str:
    return eval_metric(
        protocol,
        f"progress/{metric_path_segment(progress)}/{metric_path_segment(stat)}",
    )


SB3_SHARED_ACTOR_CRITIC_SCALAR_MAP = {
    "rollout/ep_rew_mean": (TRAIN_EPISODE_RETURN_SHAPED_MEAN, 1.0),
    "rollout/ep_len_mean": (TRAIN_EPISODE_LENGTH_MEAN, 1.0),
    "train/entropy_loss": ("policy/entropy", -1.0),
    "train/explained_variance": ("value/explained_variance", 1.0),
    "train/policy_gradient_loss": ("update/policy_gradient_loss", 1.0),
    "train/policy_loss": ("update/policy_gradient_loss", 1.0),
    "train/value_loss": ("update/value_loss", 1.0),
    "train/learning_rate": ("update/learning_rate", 1.0),
    "train/std": ("policy/distribution_std", 1.0),
}

SB3_PPO_SCALAR_MAP = {
    "train/approx_kl": (TRAIN_PPO_APPROX_KL, 1.0),
    "train/clip_fraction": (TRAIN_PPO_CLIP_FRACTION, 1.0),
}

SB3_IGNORED_SCALARS = {
    "rollout/ep_rew_mean",  # mapped above; listed here only for documentation symmetry
    "time/fps",
    "time/iterations",
    "time/time_elapsed",
    "time/total_timesteps",
    "train/clip_range",
    "train/clip_range_vf",
    "train/loss",
    "train/n_updates",
}

_RLAB_OWNED_PREFIXES = (
    "train/episode/",
    "train/outcome/",
    "train/reward/",
    "train/algorithm/",
    "train/throughput/",
    "train/artifact/",
    "eval/",
    "leader/",
)


def canonical_training_scalars(
    key_values: Mapping[str, Any],
    *,
    algorithm_id: str = "ppo",
) -> dict[str, float]:
    train_algorithm_root(algorithm_id)
    payload: dict[str, float] = {}
    for key, value in key_values.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            continue
        numeric = float(value)
        raw_name = str(key)
        mapped = SB3_SHARED_ACTOR_CRITIC_SCALAR_MAP.get(raw_name)
        if mapped is not None:
            name, multiplier = mapped
            if name.startswith(("train/", "eval/", "leader/")):
                payload[name] = numeric * multiplier
            else:
                payload[train_algorithm_metric(algorithm_id, name)] = numeric * multiplier
        elif algorithm_id == "ppo" and (mapped := SB3_PPO_SCALAR_MAP.get(raw_name)) is not None:
            name, multiplier = mapped
            payload[name] = numeric * multiplier
        elif metric_definition(raw_name) is not None:
            payload[raw_name] = numeric
        elif raw_name in SB3_IGNORED_SCALARS:
            continue
        elif raw_name.startswith(_RLAB_OWNED_PREFIXES):
            raise ValueError(f"unknown rlab metric at logger boundary: {key}")
    validate_metric_payload(payload)
    return payload


def render_metric_registry_markdown() -> str:
    lines = [
        "| Metric or template | Meaning | Unit | Cadence | Surface |",
        "|---|---|---|---|---|",
    ]
    for definition in METRIC_DEFINITIONS:
        lines.append(
            f"| `{definition.name}` | {definition.description} | {definition.unit} | "
            f"{definition.cadence} | {definition.storage} |"
        )
    return "\n".join(lines)
