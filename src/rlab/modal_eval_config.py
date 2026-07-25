from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field, StringConstraints, model_validator

from rlab.boundary_schema import BoundaryModel, validate_boundary
from rlab.config_loader import load_mapping_document


DEFAULT_MODAL_EVAL_CONFIG = Path(__file__).resolve().parents[2] / "experiments" / "modal_eval.yaml"


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveFloat = Annotated[float, Field(strict=True, gt=0)]


class _Deployment(BoundaryModel):
    environment_name: NonEmptyText
    app_name_prefix: NonEmptyText
    function_name: NonEmptyText


class _Resources(BoundaryModel):
    cpu: PositiveFloat
    memory_mib: PositiveInt
    min_containers: NonNegativeInt
    buffer_containers: NonNegativeInt
    max_containers: PositiveInt
    single_use_containers: bool
    scaledown_window_seconds: PositiveInt
    startup_timeout_seconds: PositiveInt


class _Timeouts(BoundaryModel):
    worker_seconds: PositiveInt
    acceptance_seconds: PositiveInt
    child_margin_seconds: PositiveInt
    expiry_margin_seconds: PositiveInt


class _Protocol(BoundaryModel):
    max_attempts: Annotated[int, Field(strict=True, ge=1, le=2)]


class _ModalEvalDocument(BoundaryModel):
    enabled: bool = False
    deployment: _Deployment
    resources: _Resources
    timeouts: _Timeouts
    protocol: _Protocol

    @model_validator(mode="after")
    def validate_child_margin(self) -> "_ModalEvalDocument":
        if self.timeouts.child_margin_seconds >= min(
            self.timeouts.worker_seconds,
            self.timeouts.acceptance_seconds,
        ):
            raise ValueError("child_margin_seconds must be smaller than every eval timeout")
        return self


class ModalEvalConfig(BoundaryModel):
    enabled: bool
    environment_name: NonEmptyText
    app_name_prefix: NonEmptyText
    function_name: NonEmptyText
    cpu: PositiveFloat
    memory_mib: PositiveInt
    min_containers: NonNegativeInt
    buffer_containers: NonNegativeInt
    max_containers: PositiveInt
    single_use_containers: bool
    scaledown_window_seconds: PositiveInt
    startup_timeout_seconds: PositiveInt
    worker_timeout_seconds: PositiveInt
    acceptance_timeout_seconds: PositiveInt
    child_margin_seconds: PositiveInt
    expiry_margin_seconds: PositiveInt
    max_attempts: Annotated[int, Field(strict=True, ge=1, le=2)]


def load_modal_eval_config(path: Path = DEFAULT_MODAL_EVAL_CONFIG) -> ModalEvalConfig:
    document = load_mapping_document(path, label=str(path))
    validated = validate_boundary(_ModalEvalDocument, document, label=str(path))
    return ModalEvalConfig(
        enabled=validated.enabled,
        environment_name=validated.deployment.environment_name,
        app_name_prefix=validated.deployment.app_name_prefix,
        function_name=validated.deployment.function_name,
        cpu=validated.resources.cpu,
        memory_mib=validated.resources.memory_mib,
        min_containers=validated.resources.min_containers,
        buffer_containers=validated.resources.buffer_containers,
        max_containers=validated.resources.max_containers,
        single_use_containers=validated.resources.single_use_containers,
        scaledown_window_seconds=validated.resources.scaledown_window_seconds,
        startup_timeout_seconds=validated.resources.startup_timeout_seconds,
        worker_timeout_seconds=validated.timeouts.worker_seconds,
        acceptance_timeout_seconds=validated.timeouts.acceptance_seconds,
        child_margin_seconds=validated.timeouts.child_margin_seconds,
        expiry_margin_seconds=validated.timeouts.expiry_margin_seconds,
        max_attempts=validated.protocol.max_attempts,
    )


def modal_app_name(prefix: str, source_sha: str) -> str:
    revision = str(source_sha).strip().lower()
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("Modal eval source must be a full lowercase Git SHA")
    return f"{prefix}-{revision[:12]}"
