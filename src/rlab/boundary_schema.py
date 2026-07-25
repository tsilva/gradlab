from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError


class BoundaryModel(BaseModel):
    """Strict immutable model for persisted or externally supplied documents."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


ModelT = TypeVar("ModelT", bound=BoundaryModel)


def _path(label: str, location: tuple[Any, ...]) -> str:
    suffix = ".".join(str(part) for part in location)
    return f"{label}.{suffix}" if suffix else label


def boundary_error_message(error: ValidationError, *, label: str) -> str:
    grouped: dict[tuple[str, tuple[Any, ...]], list[str]] = defaultdict(list)
    ordinary: list[Mapping[str, Any]] = []
    for detail in error.errors(include_url=False):
        location = tuple(detail["loc"])
        if detail["type"] == "extra_forbidden":
            grouped[("unknown", location[:-1])].append(str(location[-1]))
        elif detail["type"] == "missing":
            grouped[("missing", location[:-1])].append(str(location[-1]))
        else:
            ordinary.append(detail)
    if grouped:
        (kind, parent), fields = next(iter(grouped.items()))
        description = "unknown field(s)" if kind == "unknown" else "missing required field(s)"
        return f"{_path(label, parent)} has {description}: {', '.join(sorted(fields))}"
    detail = ordinary[0]
    return f"{_path(label, tuple(detail['loc']))} {detail['msg']}"


def validate_boundary(
    model: type[ModelT],
    value: object,
    *,
    label: str,
    error_type: type[ValueError] = ValueError,
) -> ModelT:
    try:
        return model.model_validate(value)
    except ValidationError as exc:
        raise error_type(boundary_error_message(exc, label=label)) from exc
