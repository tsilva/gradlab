from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

METRICS_SCHEMA_VERSION = 22
EPISODE_METRIC_WINDOW_SIZE = 100
METRICS_EPISODE_WINDOW_SIZE_CONFIG = "metrics_episode_window_size"

TRAIN_GLOBAL_STEP = "train/step"
EVAL_CHECKPOINT_STEP = "eval/step"
ORCHESTRATION_EVENT_SEQUENCE = "ops/sequence"
ORCHESTRATION_OUTBOX_PENDING_COUNT = "ops/outbox/count"
ORCHESTRATION_OUTBOX_OLDEST_AGE_SECONDS = "ops/outbox/age/seconds"
ORCHESTRATION_OUTBOX_REMOTE_VISIBILITY_LAG_SECONDS = "ops/visibility/seconds"
ORCHESTRATION_CHECKPOINT_PENDING_COUNT = "ops/checkpoints/count"
ORCHESTRATION_EVAL_PENDING_COUNT = "ops/evals/count"
ORCHESTRATION_DRAIN_GPU_IDLE_SECONDS = "ops/drain/seconds"
ORCHESTRATION_SCRATCH_USED_FRACTION = "ops/scratch/fraction"
ORCHESTRATION_RUN_TERMINAL_STATE = "ops/state"
ORCHESTRATION_RUN_TERMINAL_REASON = "ops/reason"

TRAIN_EPISODE_RETURN_SHAPED_ORIGIN_TARGET_ROLLING_MEAN = "train/return/mean"
TRAIN_EPISODE_RETURN_SHAPED_ORIGIN_TARGET_ROLLING_MAX = "train/return/max"
TRAIN_EPISODE_LENGTH_ORIGIN_ALL_ROLLING_MEAN = "train/episode_steps/mean"
TRAIN_EXPLORATION_CELL_UNIQUE_ORIGIN_TARGET_ROLLING_MEAN = "train/unique_cells/mean"
TRAIN_PROGRESS_KILLS_ORIGIN_TARGET_ROLLING_MEAN = "train/progress/kills/mean"
TRAIN_EPISODE_COMPLETED_COUNT = "train/episodes/count"

TRAIN_ARCHIVE_CURRICULUM_ROOT = "train/curriculum"
TRAIN_ARCHIVE_CURRICULUM_CELL_COUNT = "train/curriculum/cells/count"
TRAIN_ARCHIVE_CURRICULUM_ENTRY_COUNT = "train/curriculum/entries/count"
TRAIN_ARCHIVE_ADMISSION_CANDIDATE_COUNT = "train/curriculum/candidates/count"
TRAIN_ARCHIVE_ADMISSION_ACCEPTED_COUNT = "train/curriculum/admitted/count"
TRAIN_ARCHIVE_EVICTED_COUNT = "train/curriculum/evicted/count"
TRAIN_ARCHIVE_CAPTURE_CALL_COUNT = "train/curriculum/capture/count"
TRAIN_ARCHIVE_RESTORE_EPISODE_COUNT = "train/curriculum/restore/count"
TRAIN_ARCHIVE_RESTORE_FORCED_BOUNDARY_COUNT = "train/curriculum/forced_boundaries/count"
TRAIN_ARCHIVE_FEEDBACK_TRAJECTORY_COUNT = "train/curriculum/feedback/count"
TRAIN_ARCHIVE_TRANSITION_SHARE = "train/curriculum/transitions/fraction"
TRAIN_ARCHIVE_SAMPLING_PROBABILITY_MAX = "train/curriculum/probability/max"
TRAIN_ARCHIVE_SAMPLING_EFFECTIVE_CELL_COUNT = "train/curriculum/effective_cells/count"
TRAIN_ARCHIVE_CAPTURE_SECONDS = "train/curriculum/capture/seconds"
TRAIN_ARCHIVE_RESTORE_SECONDS = "train/curriculum/restore/seconds"

TRAIN_OUTCOME_SUCCESS_ROOT = "train/success"
TRAIN_OUTCOME_SUCCESS_STARTS_ALL_ROLLING_RATE_MIN = "train/success/min"
TRAIN_OUTCOME_SUCCESS_STARTS_ALL_ROLLING_RATE_MEAN = "train/success/mean"

TRAIN_EARLY_STOP_ROOT = "train/patience"
TRAIN_REWARD_ROOT = "train/reward"

TRAIN_ALGORITHM_ROOT = "train"
TRAIN_ACTOR_CRITIC_ALGORITHMS = ("ppo", "a2c")
TRAIN_ALGORITHM_JERK_ROOT = f"{TRAIN_ALGORITHM_ROOT}/jerk"
TRAIN_ALGORITHM_JERK_RETAINED_COUNT = "train/jerk/programs/count"
TRAIN_ALGORITHM_JERK_BEST_RETURN_MEAN = "train/jerk/return/mean"
TRAIN_ALGORITHM_JERK_BEST_PROGRAM_STEPS = "train/program/steps"

TRAIN_ALGORITHM_GO_EXPLORE_ROOT = f"{TRAIN_ALGORITHM_ROOT}/go-explore"
TRAIN_GO_EXPLORE_ARCHIVE_CELL_COUNT = "train/go-explore/cells/count"
TRAIN_GO_EXPLORE_ARCHIVE_BLOB_BYTES = "train/go-explore/bytes"
TRAIN_GO_EXPLORE_ARCHIVE_VISIT_COUNT = "train/go-explore/visits/count"
TRAIN_GO_EXPLORE_ARCHIVE_CELL_DISCOVERY_RATE = "train/go-explore/discovery/fraction"
TRAIN_GO_EXPLORE_BEST_PROGRESS = "train/go-explore/progress"
TRAIN_GO_EXPLORE_BEST_RETURN = "train/go-explore/return"
TRAIN_GO_EXPLORE_BEST_PROGRAM_STEPS = "train/program/steps"


def train_algorithm_root(algorithm_id: str) -> str:
    if algorithm_id not in TRAIN_ACTOR_CRITIC_ALGORITHMS:
        raise ValueError(f"unsupported actor-critic algorithm id: {algorithm_id}")
    return TRAIN_ALGORITHM_ROOT


def train_algorithm_metric(algorithm_id: str, suffix: str) -> str:
    root = train_algorithm_root(algorithm_id)
    names = {
        "explained_variance": "explained_variance",
        "policy_loss": "policy_loss/mean",
        "value_loss": "value_loss/mean",
        "learning_rate": "learning_rate",
        "entropy": "entropy/mean",
        "action_std": "noise/std/mean",
        "dominant_action_rate": "action/fraction/max",
        "rollout_value": "value",
        "rollout_advantage": "advantage",
    }
    return f"{root}/{names[suffix]}"


TRAIN_ALGORITHM_PPO_ROOT = train_algorithm_root("ppo")
TRAIN_ALGORITHM_A2C_ROOT = train_algorithm_root("a2c")
TRAIN_PPO_APPROX_KL = "train/kl/mean"
TRAIN_PPO_CLIP_FRACTION = "train/clip/fraction"
TRAIN_PPO_EXPLAINED_VARIANCE = train_algorithm_metric("ppo", "explained_variance")
TRAIN_PPO_VALUE_LOSS = train_algorithm_metric("ppo", "value_loss")
TRAIN_PPO_LEARNING_RATE = train_algorithm_metric("ppo", "learning_rate")
TRAIN_PPO_POLICY_ENTROPY = train_algorithm_metric("ppo", "entropy")
TRAIN_A2C_EXPLAINED_VARIANCE = train_algorithm_metric("a2c", "explained_variance")
TRAIN_A2C_VALUE_LOSS = train_algorithm_metric("a2c", "value_loss")
TRAIN_A2C_LEARNING_RATE = train_algorithm_metric("a2c", "learning_rate")
TRAIN_A2C_POLICY_ENTROPY = train_algorithm_metric("a2c", "entropy")

TRAIN_THROUGHPUT_ROOT = "train/throughput"
TRAIN_THROUGHPUT_LOOP_RATE = "train/throughput/rate"
TRAIN_THROUGHPUT_PROVIDER_STEP_RATE = "train/provider/rate"
TRAIN_THROUGHPUT_ROLLOUT_OVERHEAD_SECONDS = "train/rollout_overhead/seconds"
TRAIN_THROUGHPUT_BETWEEN_ROLLOUTS_SECONDS = "train/between_rollouts/seconds"
TRAIN_ARTIFACT_SAVE_SECONDS = "train/save/seconds"

EVAL_ROOT = "eval"
EVAL_FULL_ROOT = EVAL_ROOT
EVAL_FULL_EPISODE_RETURN_SHAPED_MEAN = "eval/return/mean"
EVAL_FULL_EPISODE_RETURN_SHAPED_MAX = "eval/return/max"
EVAL_FULL_PROGRESS_X_MAX = f"{EVAL_FULL_ROOT}/progress/x/max"
EVAL_FULL_OUTCOME_SUCCESS_STARTS_RATE_MIN = "eval/success/min"
EVAL_FULL_OUTCOME_SUCCESS_STARTS_RATE_MEAN = "eval/success/mean"
EVAL_FULL_START_TABLE = "eval/starts/table"
EVAL_ACCEPTANCE_PASS = "eval/pass"
EVAL_ACCEPTANCE_EPISODE_COMPLETED_COUNT = "eval/episodes/count"
EVAL_START_TABLE_COLUMNS = (
    "start_id",
    "episode_count",
    "success_count",
    "success_rate",
    "shaped_return_mean",
    "failure_reasons",
)

LEADER_CHECKPOINT_OUTCOME_SUCCESS_STARTS_RATE_MIN = "leader/success/min"
LEADER_CHECKPOINT_RETURN_SHAPED_MEAN = "leader/return/mean"
LEADER_CHECKPOINT_RETURN_SHAPED_MAX = "leader/return/max"
LEADER_CHECKPOINT_STEP = "leader/step"
LEADER_CHECKPOINT_ARTIFACT_REF = "leader/artifact"
LEADER_CHECKPOINT_EVALUATION_SOURCE = "leader/source"
LEADER_CHECKPOINT_PROJECTION_TIMESTAMP = "leader/updated_at"


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    display_label: str
    description: str
    unit: str
    cadence: str
    placement: str
    summary_reducer: str
    axis: str
    evidence: str
    leader: str
    training_proxy: str


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
    return validate_metric_name(f"leader/progress/{metric_path_segment(progress)}/{statistic}")


def leader_metric_for_rank_metric(
    metric: str,
    *,
    schema_version: int = METRICS_SCHEMA_VERSION,
) -> str:
    require_current_metrics_schema(schema_version)
    mapped = metric_relationship(metric, "leader")
    if mapped is not None:
        return mapped
    raise ValueError(f"evaluation rank criterion cannot be projected: {metric}")


_METRIC_REGISTRY_START = "<!-- METRIC_REGISTRY_START -->"
_METRIC_REGISTRY_END = "<!-- METRIC_REGISTRY_END -->"
_METRIC_REGISTRY_HEADER = "| Metric or template | Display label | Meaning | Unit | Cadence | Placement | Summary | Axis | Evidence | Leader | Training proxy |"
_METRIC_REGISTRY_SEPARATOR = "|---|---|---|---|---|---|---|---|---|---|---|"


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
        if len(columns) != 11:
            raise RuntimeError(
                f"METRICS.md metric registry row {line_number} must have eleven columns"
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
        if definition.evidence not in {
            "training",
            "evaluation",
            "acceptance",
            "evaluation_table",
            "selection",
            "operational",
        }:
            raise RuntimeError(f"invalid metric evidence category: {definition.evidence}")
        definitions.append(definition)
    names = [definition.name for definition in definitions]
    if not names or len(names) != len(set(names)):
        raise RuntimeError("METRICS.md metric registry must be non-empty and unique")
    registered = set(names)
    for definition in definitions:
        for relation in ("axis", "leader", "training_proxy"):
            target = getattr(definition, relation)
            if target != "-" and target not in registered:
                raise RuntimeError(f"unregistered {relation} for {definition.name}: {target}")
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


def metric_relationship(name: str, relationship: str) -> str | None:
    if relationship not in {"leader", "training_proxy"}:
        raise ValueError(f"unknown metric relationship: {relationship}")
    for definition, pattern in _DEFINITION_PATTERNS:
        if (match := pattern.fullmatch(name)) is not None:
            target = getattr(definition, relationship)
            return (
                None
                if target == "-"
                else validate_metric_name(target.format_map(match.groupdict()))
            )
    raise ValueError(f"unknown metric name: {name}")


def training_proxy_metric(name: str, *, progress_fields: frozenset[str]) -> str | None:
    proxy = metric_relationship(name, "training_proxy")
    if proxy and proxy.startswith("train/progress/"):
        field = proxy.split("/")[2]
        if field not in progress_fields:
            return None
    return proxy


def metric_display_label(name: str) -> str:
    for definition, pattern in _DEFINITION_PATTERNS:
        if (match := pattern.fullmatch(name)) is not None:
            return definition.display_label.format_map(match.groupdict())
    raise ValueError(f"unknown metric name: {name}")


def validate_metric_name(name: str) -> str:
    if metric_definition(name) is None:
        raise ValueError(f"unknown metric name: {name}")
    return name


def validate_metric_payload(payload: Mapping[str, Any], *, placement: str = "history") -> None:
    if placement not in {"history", "summary"}:
        raise ValueError(f"unknown metric placement: {placement}")
    for raw_name in payload:
        name = str(raw_name)
        definition = metric_definition(name)
        if definition is None:
            raise ValueError(f"unknown metric name: {name}")
        if definition.placement != placement:
            raise ValueError(f"metric {name} belongs in {definition.placement}, not {placement}")


def summary_value(value: Any) -> Any:
    while isinstance(value, Mapping) or callable(getattr(value, "items", None)):
        if not isinstance(value, Mapping):
            try:
                value = dict(value.items())
            except TypeError, ValueError:
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
    return validate_metric_name(f"train/unsuccessful/{metric_path_segment(reason)}/fraction")


def train_progress_origin_target_rolling_mean_metric(progress: object) -> str:
    return validate_metric_name(f"train/progress/{metric_path_segment(progress)}/mean")


def train_progress_origin_target_rolling_min_metric(progress: object) -> str:
    return validate_metric_name(f"train/progress/{metric_path_segment(progress)}/min")


def train_progress_origin_target_rolling_max_metric(progress: object) -> str:
    return validate_metric_name(f"train/progress/{metric_path_segment(progress)}/max")


def train_early_stop_metric(condition: object, suffix: str = "fraction") -> str:
    return validate_metric_name(
        f"{TRAIN_EARLY_STOP_ROOT}/{metric_path_segment(condition)}/{suffix.strip('/')}"
    )


def train_success_start_metric(start: object, suffix: str) -> str:
    return validate_metric_name(
        f"{TRAIN_OUTCOME_SUCCESS_ROOT}/{metric_value_segment(start)}/{suffix}"
    )


def train_success_count_metric(start: object) -> str:
    return train_success_start_metric(start, "count")


def train_success_rolling_rate_metric(start: object) -> str:
    return train_success_start_metric(start, "fraction")


def train_reward_component_metric(component: object, stat: str) -> str:
    suffix = "fraction" if stat == "nonzero_rate" else stat
    return validate_metric_name(
        f"{TRAIN_REWARD_ROOT}/part/{metric_path_segment(component)}/{suffix}"
    )


def train_reward_event_metric(event: object, stat: str) -> str:
    suffix = "fraction" if stat == "nonzero_rate" else stat
    return validate_metric_name(f"{TRAIN_REWARD_ROOT}/event/{metric_path_segment(event)}/{suffix}")


def eval_full_outcome_success_starts_rate_metric(statistic: str) -> str:
    if statistic not in {"min", "mean"}:
        raise ValueError("full-evaluation success statistic must be 'min' or 'mean'")
    return validate_metric_name(f"{EVAL_FULL_ROOT}/success/{metric_path_segment(statistic)}")


def eval_full_progress_metric(progress: object, statistic: str) -> str:
    if statistic not in {"mean", "max"}:
        raise ValueError("full-evaluation progress statistic must be 'mean' or 'max'")
    return validate_metric_name(
        f"{EVAL_FULL_ROOT}/progress/{metric_path_segment(progress)}/"
        f"{metric_path_segment(statistic)}"
    )


SB3_SHARED_ACTOR_CRITIC_SCALAR_MAP = {
    "train/entropy_loss": ("entropy", -1.0),
    "train/explained_variance": ("explained_variance", 1.0),
    "train/policy_gradient_loss": ("policy_loss", 1.0),
    "train/policy_loss": ("policy_loss", 1.0),
    "train/value_loss": ("value_loss", 1.0),
    "train/learning_rate": ("learning_rate", 1.0),
    "train/std": ("action_std", 1.0),
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
    "train/",
    "eval/",
    "leader/",
    "ops/",
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
