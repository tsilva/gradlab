from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources
from numbers import Real
from pathlib import Path
from typing import Any, Mapping


METRICS_SCHEMA_VERSION = 20
EPISODE_METRIC_WINDOW_SIZE = 100
METRICS_EPISODE_WINDOW_SIZE_CONFIG = "metrics_episode_window_size"

TRAIN_GLOBAL_STEP = "train/global_step"
EVAL_CHECKPOINT_STEP = "eval/checkpoint/step"
ORCHESTRATION_EVENT_SEQUENCE = "orchestration/event/sequence"
ORCHESTRATION_OUTBOX_PENDING_COUNT = "orchestration/outbox/pending/count"
ORCHESTRATION_OUTBOX_OLDEST_AGE_SECONDS = "orchestration/outbox/oldest/age/seconds"
ORCHESTRATION_OUTBOX_REMOTE_VISIBILITY_LAG_SECONDS = (
    "orchestration/outbox/remote/visibility/lag/seconds"
)
ORCHESTRATION_CHECKPOINT_PENDING_COUNT = "orchestration/checkpoint/pending/count"
ORCHESTRATION_EVAL_PENDING_COUNT = "orchestration/eval/pending/count"
ORCHESTRATION_DRAIN_GPU_IDLE_SECONDS = "orchestration/drain/gpu/idle/seconds"
ORCHESTRATION_SCRATCH_USED_FRACTION = "orchestration/scratch/used/fraction"
ORCHESTRATION_RUN_TERMINAL_STATE = "orchestration/run/terminal/state"
ORCHESTRATION_RUN_TERMINAL_REASON = "orchestration/run/terminal/reason"

TRAIN_EPISODE_RETURN_SHAPED_ORIGIN_TARGET_ROLLING_MEAN = (
    "train/episode/return/shaped/origin/target/rolling/mean"
)
TRAIN_EPISODE_RETURN_SHAPED_ORIGIN_TARGET_ROLLING_MAX = (
    "train/episode/return/shaped/origin/target/rolling/max"
)
TRAIN_EPISODE_LENGTH_ORIGIN_ALL_ROLLING_MEAN = (
    "train/episode/length/origin/all/rolling/mean"
)
TRAIN_EXPLORATION_CELL_UNIQUE_ORIGIN_TARGET_ROLLING_MEAN = (
    "train/exploration/cell/unique/origin/target/rolling/mean"
)
TRAIN_PROGRESS_KILLS_ORIGIN_TARGET_ROLLING_MEAN = (
    "train/progress/kills/origin/target/rolling/mean"
)
TRAIN_EPISODE_COMPLETED_COUNT = "train/episode/completed/count"

TRAIN_ARCHIVE_CURRICULUM_ROOT = "train/curriculum/archive"
TRAIN_ARCHIVE_CURRICULUM_CELL_COUNT = f"{TRAIN_ARCHIVE_CURRICULUM_ROOT}/cell/count"
TRAIN_ARCHIVE_CURRICULUM_ENTRY_COUNT = f"{TRAIN_ARCHIVE_CURRICULUM_ROOT}/entry/count"
TRAIN_ARCHIVE_ADMISSION_CANDIDATE_COUNT = (
    f"{TRAIN_ARCHIVE_CURRICULUM_ROOT}/admission/candidate/count"
)
TRAIN_ARCHIVE_ADMISSION_ACCEPTED_COUNT = (
    f"{TRAIN_ARCHIVE_CURRICULUM_ROOT}/admission/accepted/count"
)
TRAIN_ARCHIVE_EVICTED_COUNT = f"{TRAIN_ARCHIVE_CURRICULUM_ROOT}/evicted/count"
TRAIN_ARCHIVE_CAPTURE_CALL_COUNT = f"{TRAIN_ARCHIVE_CURRICULUM_ROOT}/capture/call/count"
TRAIN_ARCHIVE_RESTORE_EPISODE_COUNT = f"{TRAIN_ARCHIVE_CURRICULUM_ROOT}/restore/episode/count"
TRAIN_ARCHIVE_RESTORE_FORCED_BOUNDARY_COUNT = (
    f"{TRAIN_ARCHIVE_CURRICULUM_ROOT}/restore/forced_boundary/count"
)
TRAIN_ARCHIVE_FEEDBACK_TRAJECTORY_COUNT = (
    f"{TRAIN_ARCHIVE_CURRICULUM_ROOT}/feedback/trajectory/count"
)
TRAIN_ARCHIVE_TRANSITION_SHARE = f"{TRAIN_ARCHIVE_CURRICULUM_ROOT}/transition/share"
TRAIN_ARCHIVE_SAMPLING_PROBABILITY_MAX = (
    f"{TRAIN_ARCHIVE_CURRICULUM_ROOT}/sampling/probability/max"
)
TRAIN_ARCHIVE_SAMPLING_EFFECTIVE_CELL_COUNT = (
    f"{TRAIN_ARCHIVE_CURRICULUM_ROOT}/sampling/effective/cell/count"
)
TRAIN_ARCHIVE_CAPTURE_SECONDS = f"{TRAIN_ARCHIVE_CURRICULUM_ROOT}/capture/seconds"
TRAIN_ARCHIVE_RESTORE_SECONDS = f"{TRAIN_ARCHIVE_CURRICULUM_ROOT}/restore/seconds"

TRAIN_OUTCOME_SUCCESS_ROOT = "train/outcome/success"
TRAIN_OUTCOME_SUCCESS_STARTS_OBSERVED_CUMULATIVE_RATE_MIN = (
    f"{TRAIN_OUTCOME_SUCCESS_ROOT}/starts/observed/cumulative/rate/min"
)
TRAIN_OUTCOME_SUCCESS_STARTS_OBSERVED_CUMULATIVE_RATE_MEAN = (
    f"{TRAIN_OUTCOME_SUCCESS_ROOT}/starts/observed/cumulative/rate/mean"
)
TRAIN_OUTCOME_SUCCESS_STARTS_ALL_ROLLING_RATE_MIN = (
    f"{TRAIN_OUTCOME_SUCCESS_ROOT}/starts/all/rolling/rate/min"
)
TRAIN_OUTCOME_SUCCESS_STARTS_ALL_ROLLING_RATE_MEAN = (
    f"{TRAIN_OUTCOME_SUCCESS_ROOT}/starts/all/rolling/rate/mean"
)

TRAIN_EARLY_STOP_ROOT = "train/early_stop"
TRAIN_REWARD_ROOT = "train/reward"

TRAIN_ALGORITHM_ROOT = "train/algorithm"
TRAIN_ACTOR_CRITIC_ALGORITHMS = ("ppo", "a2c")
TRAIN_ALGORITHM_JERK_ROOT = f"{TRAIN_ALGORITHM_ROOT}/jerk"
TRAIN_ALGORITHM_JERK_RETAINED_COUNT = f"{TRAIN_ALGORITHM_JERK_ROOT}/retained/count"
TRAIN_ALGORITHM_JERK_BEST_RETURN_MEAN = f"{TRAIN_ALGORITHM_JERK_ROOT}/best/return/mean"
TRAIN_ALGORITHM_JERK_BEST_PROGRAM_STEPS = f"{TRAIN_ALGORITHM_JERK_ROOT}/best/program/steps"

TRAIN_ALGORITHM_GO_EXPLORE_ROOT = f"{TRAIN_ALGORITHM_ROOT}/go-explore"
TRAIN_GO_EXPLORE_ARCHIVE_CELL_COUNT = f"{TRAIN_ALGORITHM_GO_EXPLORE_ROOT}/archive/cell/count"
TRAIN_GO_EXPLORE_ARCHIVE_BLOB_BYTES = f"{TRAIN_ALGORITHM_GO_EXPLORE_ROOT}/archive/blob/bytes"
TRAIN_GO_EXPLORE_ARCHIVE_VISIT_COUNT = f"{TRAIN_ALGORITHM_GO_EXPLORE_ROOT}/archive/visit/count"
TRAIN_GO_EXPLORE_ARCHIVE_CELL_DISCOVERY_RATE = (
    f"{TRAIN_ALGORITHM_GO_EXPLORE_ROOT}/archive/cell/discovery/rate"
)
TRAIN_GO_EXPLORE_BEST_PROGRESS = f"{TRAIN_ALGORITHM_GO_EXPLORE_ROOT}/best/progress"
TRAIN_GO_EXPLORE_BEST_RETURN = f"{TRAIN_ALGORITHM_GO_EXPLORE_ROOT}/best/return"
TRAIN_GO_EXPLORE_BEST_PROGRAM_STEPS = f"{TRAIN_ALGORITHM_GO_EXPLORE_ROOT}/best/program/steps"


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
TRAIN_THROUGHPUT_LOOP_RATE = f"{TRAIN_THROUGHPUT_ROOT}/loop/rate"
TRAIN_THROUGHPUT_PROVIDER_STEP_RATE = f"{TRAIN_THROUGHPUT_ROOT}/provider/step/rate"
TRAIN_THROUGHPUT_ROLLOUT_OVERHEAD_SECONDS = f"{TRAIN_THROUGHPUT_ROOT}/rollout/overhead/seconds"
TRAIN_THROUGHPUT_BETWEEN_ROLLOUTS_SECONDS = f"{TRAIN_THROUGHPUT_ROOT}/between/rollouts/seconds"
TRAIN_ARTIFACT_SAVE_SECONDS = "train/artifact/save/seconds"

EVAL_ROOT = "eval"
EVAL_FULL_ROOT = f"{EVAL_ROOT}/full"
EVAL_FULL_EPISODE_RETURN_SHAPED_MEAN = f"{EVAL_FULL_ROOT}/episode/return/shaped/mean"
EVAL_FULL_EPISODE_RETURN_SHAPED_MAX = f"{EVAL_FULL_ROOT}/episode/return/shaped/max"
EVAL_FULL_PROGRESS_X_MAX = f"{EVAL_FULL_ROOT}/progress/x/max"
EVAL_FULL_OUTCOME_SUCCESS_STARTS_RATE_MIN = (
    f"{EVAL_FULL_ROOT}/outcome/success/starts/rate/min"
)
EVAL_FULL_OUTCOME_SUCCESS_STARTS_RATE_MEAN = (
    f"{EVAL_FULL_ROOT}/outcome/success/starts/rate/mean"
)
EVAL_FULL_START_TABLE = f"{EVAL_FULL_ROOT}/start/table"
EVAL_ACCEPTANCE_PASS = "eval/acceptance/pass"
EVAL_ACCEPTANCE_EPISODE_PLANNED_COUNT = "eval/acceptance/episode/planned/count"
EVAL_ACCEPTANCE_EPISODE_COMPLETED_COUNT = "eval/acceptance/episode/completed/count"
EVAL_START_TABLE_COLUMNS = (
    "start_id",
    "episode_count",
    "success_count",
    "success_rate",
    "shaped_return_mean",
    "failure_reasons",
)

LEADER_CHECKPOINT_OUTCOME_SUCCESS_STARTS_RATE_MIN = (
    "leader/checkpoint/outcome/success/starts/rate/min"
)
LEADER_CHECKPOINT_RETURN_SHAPED_MEAN = "leader/checkpoint/episode/return/shaped/mean"
LEADER_CHECKPOINT_RETURN_SHAPED_MAX = "leader/checkpoint/episode/return/shaped/max"
LEADER_CHECKPOINT_STEP = "leader/checkpoint/step"
LEADER_CHECKPOINT_ARTIFACT_REF = "leader/checkpoint/artifact/ref"
LEADER_CHECKPOINT_EVALUATION_SOURCE = "leader/checkpoint/evaluation/source"
LEADER_CHECKPOINT_PROJECTION_TIMESTAMP = "leader/checkpoint/projection/timestamp"


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    display_label: str
    description: str
    unit: str
    cadence: str
    placement: str
    summary_reducer: str


def require_current_metrics_schema(version: object) -> int:
    try:
        normalized = int(version)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unsupported metrics schema version: {version!r}") from exc
    if normalized != METRICS_SCHEMA_VERSION:
        raise ValueError(f"unsupported metrics schema version: {normalized}")
    return normalized


def leader_checkpoint_progress_metric(progress: object, statistic: str = "max") -> str:
    if statistic not in {"mean", "max"}:
        raise ValueError("leader checkpoint progress statistic must be 'mean' or 'max'")
    return validate_metric_name(
        f"leader/checkpoint/progress/{metric_path_segment(progress)}/{statistic}"
    )


def leader_metric_for_rank_metric(
    metric: str,
    *,
    schema_version: int = METRICS_SCHEMA_VERSION,
) -> str:
    require_current_metrics_schema(schema_version)
    fixed = {
        EVAL_FULL_OUTCOME_SUCCESS_STARTS_RATE_MIN: LEADER_CHECKPOINT_OUTCOME_SUCCESS_STARTS_RATE_MIN,
        EVAL_FULL_EPISODE_RETURN_SHAPED_MEAN: LEADER_CHECKPOINT_RETURN_SHAPED_MEAN,
        EVAL_FULL_EPISODE_RETURN_SHAPED_MAX: LEADER_CHECKPOINT_RETURN_SHAPED_MAX,
        LEADER_CHECKPOINT_STEP: LEADER_CHECKPOINT_STEP,
    }
    if (mapped := fixed.get(metric)) is not None:
        return mapped
    prefix = f"{EVAL_FULL_ROOT}/progress/"
    if metric.startswith(prefix):
        for statistic in ("mean", "max"):
            suffix = f"/{statistic}"
            if metric.endswith(suffix):
                progress = metric[len(prefix) : -len(suffix)]
                if "/" not in progress:
                    return leader_checkpoint_progress_metric(progress, statistic)
    raise ValueError(f"evaluation rank criterion cannot be projected: {metric}")


_METRIC_REGISTRY_START = "<!-- METRIC_REGISTRY_START -->"
_METRIC_REGISTRY_END = "<!-- METRIC_REGISTRY_END -->"
_METRIC_REGISTRY_HEADER = (
    "| Metric or template | Display label | Meaning | Unit | Cadence | Placement | Summary |"
)
_METRIC_REGISTRY_SEPARATOR = "|---|---|---|---|---|---|---|"


def _metrics_markdown() -> str:
    source_document = Path(__file__).resolve().parents[2] / "METRICS.md"
    if source_document.is_file():
        return source_document.read_text(encoding="utf-8")
    return resources.files("gradlab").joinpath("METRICS.md").read_text(encoding="utf-8")


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
        if len(columns) != 7:
            raise RuntimeError(
                f"METRICS.md metric registry row {line_number} must have seven columns"
            )
        name = columns[0]
        if len(name) < 3 or not name.startswith("`") or not name.endswith("`"):
            raise RuntimeError(
                f"METRICS.md metric registry row {line_number} must use a code metric name"
            )
        definition = MetricDefinition(name[1:-1], *columns[1:])
        if definition.placement not in {"history", "summary"}:
            raise RuntimeError(f"invalid metric placement: {definition.placement}")
        if definition.summary_reducer not in {"last", "max", "none"}:
            raise RuntimeError(f"invalid metric summary reducer: {definition.summary_reducer}")
        definitions.append(definition)
    names = [definition.name for definition in definitions]
    if not names or len(names) != len(set(names)):
        raise RuntimeError("METRICS.md metric registry must be non-empty and unique")
    return tuple(definitions)


METRIC_DEFINITIONS = _load_metric_definitions()

_SAFE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_PLACEHOLDER_PATTERNS = {
    "algorithm": "(?:ppo|a2c)",
    "reason": "[A-Za-z0-9_.-]+",
    "start": "[A-Za-z0-9_.-]+",
    "component": "[A-Za-z0-9_.-]+",
    "condition": "[A-Za-z0-9_.-]+",
    "event": "[A-Za-z0-9_.-]+",
    "progress": "[A-Za-z0-9_.-]+",
}


def _definition_pattern(template: str) -> re.Pattern[str]:
    cursor = 0
    parts: list[str] = []
    for match in re.finditer(r"\{([a-z_]+)\}", template):
        name = match.group(1)
        parts.append(re.escape(template[cursor : match.start()]))
        parts.append(f"(?P<{name}>{_PLACEHOLDER_PATTERNS[name]})")
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


def metric_display_label(name: str) -> str:
    for definition, pattern in _DEFINITION_PATTERNS:
        if (match := pattern.fullmatch(name)) is not None:
            return definition.display_label.format_map(match.groupdict())
    raise ValueError(f"unknown metric name: {name}")


def validate_metric_name(name: str) -> str:
    if metric_definition(name) is None:
        raise ValueError(f"unknown metric name: {name}")
    return name


def validate_metric_payload(
    payload: Mapping[str, Any], *, placement: str = "history"
) -> None:
    if placement not in {"history", "summary"}:
        raise ValueError(f"unknown metric placement: {placement}")
    for raw_name in payload:
        name = str(raw_name)
        definition = metric_definition(name)
        if definition is None:
            raise ValueError(f"unknown metric name: {name}")
        if definition.placement != placement:
            raise ValueError(
                f"metric {name} belongs in {definition.placement}, not {placement}"
            )


def summary_value(value: Any) -> Any:
    while isinstance(value, Mapping) or callable(getattr(value, "items", None)):
        if not isinstance(value, Mapping):
            try:
                value = dict(value.items())
            except (TypeError, ValueError):
                return value
        for reducer in ("max", "last", "min"):
            if reducer in value:
                value = value[reducer]
                break
        else:
            if len(value) != 1:
                return None
            value = next(iter(value.values()))
    return value


def summary_metric_value(summary: Mapping[str, Any], name: str) -> Any:
    direct = summary_value(summary.get(name))
    if direct is not None:
        return direct
    definition = metric_definition(name)
    if definition is None or definition.summary_reducer == "none":
        return None
    return summary_value(summary.get(f"{name}.{definition.summary_reducer}"))


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


def train_outcome_reason_rolling_rate_metric(reason: object) -> str:
    return validate_metric_name(
        f"train/outcome/failure/reason/{metric_path_segment(reason)}/rolling/rate"
    )


def train_outcome_reason_count_metric(reason: object) -> str:
    return validate_metric_name(
        f"train/outcome/failure/reason/{metric_path_segment(reason)}/count"
    )


def train_outcome_reason_rolling_count_metric(reason: object) -> str:
    return validate_metric_name(
        f"train/outcome/failure/reason/{metric_path_segment(reason)}/rolling/count"
    )


def train_progress_origin_target_rolling_mean_metric(progress: object) -> str:
    return validate_metric_name(
        f"train/progress/{metric_path_segment(progress)}/origin/target/rolling/mean"
    )


def train_early_stop_metric(condition: object, suffix: str) -> str:
    return validate_metric_name(
        f"{TRAIN_EARLY_STOP_ROOT}/{metric_path_segment(condition)}/{suffix.strip('/')}"
    )


def train_success_start_metric(start: object, suffix: str) -> str:
    return validate_metric_name(
        f"{TRAIN_OUTCOME_SUCCESS_ROOT}/start/{metric_value_segment(start)}/{suffix}"
    )


def train_success_count_metric(start: object) -> str:
    return train_success_start_metric(start, "episode/count")


def train_success_rolling_rate_metric(start: object) -> str:
    return train_success_start_metric(start, "rolling/rate")


def train_reward_component_metric(component: object, stat: str) -> str:
    suffix = "nonzero/rate" if stat in {"nonzero_rate", "nonzero/rate"} else stat
    return validate_metric_name(
        f"{TRAIN_REWARD_ROOT}/component/{metric_path_segment(component)}/{suffix}"
    )


def train_reward_event_metric(event: object, stat: str) -> str:
    suffix = "nonzero/rate" if stat in {"nonzero_rate", "nonzero/rate"} else stat
    return validate_metric_name(
        f"{TRAIN_REWARD_ROOT}/event/{metric_path_segment(event)}/{suffix}"
    )


def eval_full_outcome_success_starts_rate_metric(statistic: str) -> str:
    if statistic not in {"min", "mean"}:
        raise ValueError("full-evaluation success statistic must be 'min' or 'mean'")
    return validate_metric_name(
        f"{EVAL_FULL_ROOT}/outcome/success/starts/rate/"
        f"{metric_path_segment(statistic)}"
    )


def eval_full_progress_metric(progress: object, statistic: str) -> str:
    if statistic not in {"mean", "max"}:
        raise ValueError("full-evaluation progress statistic must be 'mean' or 'max'")
    return validate_metric_name(
        f"{EVAL_FULL_ROOT}/progress/{metric_path_segment(progress)}/"
        f"{metric_path_segment(statistic)}"
    )


SB3_SHARED_ACTOR_CRITIC_SCALAR_MAP = {
    "train/entropy_loss": ("policy/entropy", -1.0),
    "train/explained_variance": ("value/explained_variance", 1.0),
    "train/policy_gradient_loss": ("update/policy_gradient_loss", 1.0),
    "train/policy_loss": ("update/policy_gradient_loss", 1.0),
    "train/value_loss": ("update/value_loss", 1.0),
    "train/learning_rate": ("update/learning_rate", 1.0),
    "train/std": ("policy/distribution/std", 1.0),
}
SB3_PPO_SCALAR_MAP = {
    "train/approx_kl": (TRAIN_PPO_APPROX_KL, 1.0),
    "train/clip_fraction": (TRAIN_PPO_CLIP_FRACTION, 1.0),
}
SB3_IGNORED_SCALARS = {
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
    "time/fps",
    "time/iterations",
    "time/time_elapsed",
    "time/total_timesteps",
    "train/clip_range",
    "train/clip_range_vf",
    "train/loss",
    "train/n_updates",
}
_GRADLAB_OWNED_PREFIXES = (
    "train/episode/",
    "train/exploration/",
    "train/progress/",
    "train/outcome/",
    "train/early_stop/",
    "train/curriculum/",
    "train/reward/",
    "train/algorithm/",
    "train/throughput/",
    "train/artifact/",
    "eval/",
    "leader/",
    "orchestration/",
)


def canonical_training_scalars(
    key_values: Mapping[str, Any], *, algorithm_id: str = "ppo"
) -> dict[str, float]:
    train_algorithm_root(algorithm_id)
    payload: dict[str, float] = {}
    for key, value in key_values.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            continue
        numeric = float(value)
        raw_name = str(key)
        if (mapped := SB3_SHARED_ACTOR_CRITIC_SCALAR_MAP.get(raw_name)) is not None:
            suffix, multiplier = mapped
            payload[train_algorithm_metric(algorithm_id, suffix)] = numeric * multiplier
        elif algorithm_id == "ppo" and (mapped := SB3_PPO_SCALAR_MAP.get(raw_name)) is not None:
            name, multiplier = mapped
            payload[name] = numeric * multiplier
        elif (definition := metric_definition(raw_name)) is not None:
            if definition.placement == "history":
                payload[raw_name] = numeric
        elif raw_name in SB3_IGNORED_SCALARS:
            continue
        elif raw_name.startswith(_GRADLAB_OWNED_PREFIXES):
            raise ValueError(f"unknown gradlab metric at logger boundary: {key}")
    validate_metric_payload(payload)
    return payload
