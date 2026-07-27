from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field, StringConstraints, model_validator

from gradlab.boundary_schema import BoundaryModel, validate_boundary
from gradlab.config_loader import load_mapping_document


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


class ModalEvalConfig(BoundaryModel):
    deployment: _Deployment
    resources: _Resources
    timeouts: _Timeouts
    protocol: _Protocol

    @model_validator(mode="after")
    def validate_child_margin(self) -> "ModalEvalConfig":
        if self.timeouts.child_margin_seconds >= min(
            self.timeouts.worker_seconds,
            self.timeouts.acceptance_seconds,
        ):
            raise ValueError("child_margin_seconds must be smaller than every eval timeout")
        return self


def load_modal_eval_config(path: Path = DEFAULT_MODAL_EVAL_CONFIG) -> ModalEvalConfig:
    document = load_mapping_document(path, label=str(path))
    return validate_boundary(ModalEvalConfig, document, label=str(path))


def modal_app_name(prefix: str, source_sha: str) -> str:
    revision = str(source_sha).strip().lower()
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("Modal eval source must be a full lowercase Git SHA")
    return f"{prefix}-{revision[:12]}"
