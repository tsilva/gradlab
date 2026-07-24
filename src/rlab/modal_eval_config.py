from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from rlab.config_loader import load_mapping_document


DEFAULT_MODAL_EVAL_CONFIG = Path(__file__).resolve().parents[2] / "experiments" / "modal_eval.yaml"


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _positive_int(value: object, *, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if result < 1:
        raise ValueError(f"{label} must be at least 1")
    return result


def _nonnegative_int(value: object, *, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _positive_float(value: object, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _bool(value: object, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


@dataclass(frozen=True)
class ModalEvalConfig:
    enabled: bool
    environment_name: str
    app_name_prefix: str
    function_name: str
    cpu: float
    memory_mib: int
    min_containers: int
    buffer_containers: int
    max_containers: int
    single_use_containers: bool
    scaledown_window_seconds: int
    startup_timeout_seconds: int
    worker_timeout_seconds: int
    acceptance_timeout_seconds: int
    child_margin_seconds: int
    expiry_margin_seconds: int
    max_attempts: int

def load_modal_eval_config(path: Path = DEFAULT_MODAL_EVAL_CONFIG) -> ModalEvalConfig:
    document = load_mapping_document(path, label=str(path))
    allowed = {
        "enabled",
        "deployment",
        "resources",
        "timeouts",
        "protocol",
    }
    unknown = sorted(set(document) - allowed)
    if unknown:
        raise ValueError(f"{path} has unknown field(s): {', '.join(unknown)}")
    deployment = _mapping(document.get("deployment"), label="deployment")
    resources = _mapping(document.get("resources"), label="resources")
    timeouts = _mapping(document.get("timeouts"), label="timeouts")
    protocol = _mapping(document.get("protocol"), label="protocol")
    sections = {
        "deployment": (
            deployment,
            {"environment_name", "app_name_prefix", "function_name"},
        ),
        "resources": (
            resources,
            {
                "cpu",
                "memory_mib",
                "min_containers",
                "buffer_containers",
                "max_containers",
                "single_use_containers",
                "scaledown_window_seconds",
                "startup_timeout_seconds",
            },
        ),
        "timeouts": (
            timeouts,
            {
                "worker_seconds",
                "acceptance_seconds",
                "child_margin_seconds",
                "expiry_margin_seconds",
            },
        ),
        "protocol": (protocol, {"max_attempts"}),
    }
    for section_name, (section, section_allowed) in sections.items():
        section_unknown = sorted(set(section) - section_allowed)
        if section_unknown:
            raise ValueError(
                f"{path} {section_name} has unknown field(s): {', '.join(section_unknown)}"
            )
    modal_cap = _positive_int(resources.get("max_containers"), label="resources.max_containers")
    max_attempts = _positive_int(protocol.get("max_attempts"), label="protocol.max_attempts")
    if max_attempts > 2:
        raise ValueError("protocol.max_attempts must not exceed 2")
    prefix = str(deployment.get("app_name_prefix") or "").strip()
    function_name = str(deployment.get("function_name") or "").strip()
    environment_name = str(deployment.get("environment_name") or "").strip()
    if not environment_name or not prefix or not function_name:
        raise ValueError("deployment names must be non-empty")
    result = ModalEvalConfig(
        enabled=_bool(document.get("enabled", False), label="enabled"),
        environment_name=environment_name,
        app_name_prefix=prefix,
        function_name=function_name,
        cpu=_positive_float(resources.get("cpu"), label="resources.cpu"),
        memory_mib=_positive_int(resources.get("memory_mib"), label="resources.memory_mib"),
        min_containers=_nonnegative_int(
            resources.get("min_containers"), label="resources.min_containers"
        ),
        buffer_containers=_nonnegative_int(
            resources.get("buffer_containers"), label="resources.buffer_containers"
        ),
        max_containers=modal_cap,
        single_use_containers=_bool(
            resources.get("single_use_containers"),
            label="resources.single_use_containers",
        ),
        scaledown_window_seconds=_positive_int(
            resources.get("scaledown_window_seconds"),
            label="resources.scaledown_window_seconds",
        ),
        startup_timeout_seconds=_positive_int(
            resources.get("startup_timeout_seconds"), label="resources.startup_timeout_seconds"
        ),
        worker_timeout_seconds=_positive_int(
            timeouts.get("worker_seconds"), label="timeouts.worker_seconds"
        ),
        acceptance_timeout_seconds=_positive_int(
            timeouts.get("acceptance_seconds"),
            label="timeouts.acceptance_seconds",
        ),
        child_margin_seconds=_positive_int(
            timeouts.get("child_margin_seconds"), label="timeouts.child_margin_seconds"
        ),
        expiry_margin_seconds=_positive_int(
            timeouts.get("expiry_margin_seconds"), label="timeouts.expiry_margin_seconds"
        ),
        max_attempts=max_attempts,
    )
    if result.child_margin_seconds >= min(
        result.worker_timeout_seconds,
        result.acceptance_timeout_seconds,
    ):
        raise ValueError("timeouts.child_margin_seconds must be smaller than every eval timeout")
    return result


def modal_app_name(prefix: str, source_sha: str) -> str:
    revision = str(source_sha).strip().lower()
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("Modal eval source must be a full lowercase Git SHA")
    return f"{prefix}-{revision[:12]}"
