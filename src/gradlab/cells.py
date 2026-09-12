from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numba import njit

from gradlab.json_utils import canonical_json_bytes


@njit(cache=True)
def bucket_number(value, mode, size, minimum, maximum):
    if not np.isfinite(value):
        raise ValueError("cell values must be finite")
    if mode == 1:
        return int(value == size)
    value = min(max(value, minimum), maximum)
    if mode == 2 and value < 0:
        raise ValueError("zero-separated buckets require nonnegative values")
    index = np.ceil(value / size) if mode == 2 else np.floor(value / size)
    if index < -(2.0**63) or index >= 2.0**63:
        raise ValueError("state archive cell index exceeds signed int64")
    return int(index)


@njit(cache=True)
def bucket_columns(values, parameters):
    result = np.empty(values.shape, dtype=np.int64)
    for lane in range(values.shape[0]):
        for column in range(values.shape[1]):
            result[lane, column] = bucket_number(
                values[lane, column],
                parameters[column, 0],
                parameters[column, 1],
                parameters[column, 2],
                parameters[column, 3],
            )
    return result


_CELL_KEYS = frozenset({"dimensions"})
_CELL_DIMENSION_KEYS = frozenset(
    {"signal", "source", "bucket_size", "clamp", "equals", "zero_separate"}
)


def _finite_number(value: Any, *, label: str) -> float:
    if (
        not isinstance(value, int | float | np.number)
        or isinstance(value, bool | np.bool_)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def normalize_archive_cell_config(
    value: Any,
    *,
    label: str,
) -> dict[str, Any]:
    """Normalize the shared YAML-defined archive cell detector."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    unexpected = sorted(set(value) - _CELL_KEYS)
    if unexpected:
        raise ValueError(f"{label} has unexpected fields: {unexpected}")
    dimensions = value.get("dimensions")
    if (
        isinstance(dimensions, str | bytes)
        or not isinstance(dimensions, Sequence)
        or not dimensions
    ):
        raise ValueError(f"{label}.dimensions must be a non-empty sequence")
    if len(dimensions) > 32:
        raise ValueError(f"{label}.dimensions must contain at most 32 entries")

    normalized_dimensions: list[dict[str, Any]] = []
    seen_selectors: set[tuple[str, str]] = set()
    for index, raw_dimension in enumerate(dimensions):
        dimension_label = f"{label}.dimensions[{index}]"
        if not isinstance(raw_dimension, Mapping):
            raise ValueError(f"{dimension_label} must be an object")
        unexpected_dimension = sorted(set(raw_dimension) - _CELL_DIMENSION_KEYS)
        if unexpected_dimension:
            raise ValueError(f"{dimension_label} has unexpected fields: {unexpected_dimension}")
        selector_fields = set(raw_dimension) & {"signal", "source"}
        if len(selector_fields) != 1:
            raise ValueError(f"{dimension_label} must define exactly one of signal or source")
        selector_kind = next(iter(selector_fields))
        selector_name = str(raw_dimension.get(selector_kind) or "").strip()
        if not selector_name:
            raise ValueError(f"{dimension_label}.{selector_kind} must be a non-empty string")
        selector = (selector_kind, selector_name)
        if selector in seen_selectors:
            raise ValueError(
                f"{label}.dimensions contains duplicate {selector_kind} {selector_name!r}"
            )
        seen_selectors.add(selector)

        has_equals = "equals" in raw_dimension
        has_bucket = "bucket_size" in raw_dimension
        has_clamp = "clamp" in raw_dimension
        if has_equals and (has_bucket or has_clamp):
            raise ValueError(
                f"{dimension_label}.equals cannot be combined with bucket_size or clamp"
            )
        normalized_dimension: dict[str, Any] = {selector_kind: selector_name}
        if has_equals:
            normalized_dimension["equals"] = _finite_number(
                raw_dimension["equals"],
                label=f"{dimension_label}.equals",
            )
        else:
            bucket_size = _finite_number(
                raw_dimension.get("bucket_size"),
                label=f"{dimension_label}.bucket_size",
            )
            if bucket_size <= 0.0:
                raise ValueError(f"{dimension_label}.bucket_size must be positive")
            normalized_dimension["bucket_size"] = bucket_size
            if has_clamp:
                clamp = raw_dimension["clamp"]
                if (
                    isinstance(clamp, str | bytes)
                    or not isinstance(clamp, Sequence)
                    or len(clamp) != 2
                ):
                    raise ValueError(f"{dimension_label}.clamp must be [minimum, maximum]")
                minimum = _finite_number(
                    clamp[0],
                    label=f"{dimension_label}.clamp[0]",
                )
                maximum = _finite_number(
                    clamp[1],
                    label=f"{dimension_label}.clamp[1]",
                )
                if minimum > maximum:
                    raise ValueError(f"{dimension_label}.clamp minimum must not exceed maximum")
                normalized_dimension["clamp"] = [minimum, maximum]
        if "zero_separate" in raw_dimension:
            if type(raw_dimension["zero_separate"]) is not bool or has_equals:
                raise ValueError(f"{dimension_label}.zero_separate requires numeric buckets")
            if raw_dimension["zero_separate"]:
                normalized_dimension["zero_separate"] = True
        normalized_dimensions.append(normalized_dimension)
    return {"dimensions": normalized_dimensions}


@dataclass(frozen=True)
class ArchiveCellDimension:
    signal: str | None = None
    source: str | None = None
    bucket_size: float | None = None
    clamp: tuple[float, float] | None = None
    equals: float | None = None
    zero_separate: bool = False

    @property
    def selector(self) -> tuple[str, str]:
        if self.signal is not None:
            return ("signal", self.signal)
        assert self.source is not None
        return ("source", self.source)

    def bucket(self, value: Any) -> int:
        kind, name = self.selector
        label = f"archive cell {kind} {name!r}"
        if isinstance(value, str | bytes):
            raise ValueError(f"{label} must be numeric")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be numeric") from exc
        if not math.isfinite(numeric):
            raise ValueError(f"{label} must be finite")
        return int(bucket_number(
            numeric, 1 if self.equals is not None else 2 if self.zero_separate else 0,
            self.equals if self.equals is not None else self.bucket_size,
            self.clamp[0] if self.clamp else -np.inf,
            self.clamp[1] if self.clamp else np.inf,
        ))


@dataclass(frozen=True)
class ArchiveCellConfig:
    dimensions: tuple[ArchiveCellDimension, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, label: str) -> "ArchiveCellConfig":
        normalized = normalize_archive_cell_config(value, label=label)
        return cls(
            dimensions=tuple(
                ArchiveCellDimension(
                    signal=(str(dimension["signal"]) if "signal" in dimension else None),
                    source=(str(dimension["source"]) if "source" in dimension else None),
                    bucket_size=(
                        float(dimension["bucket_size"]) if "bucket_size" in dimension else None
                    ),
                    clamp=(
                        (
                            float(dimension["clamp"][0]),
                            float(dimension["clamp"][1]),
                        )
                        if "clamp" in dimension
                        else None
                    ),
                    equals=(float(dimension["equals"]) if "equals" in dimension else None),
                    zero_separate=dimension.get("zero_separate", False),
                )
                for dimension in normalized["dimensions"]
            )
        )

    @property
    def signals(self) -> tuple[str, ...]:
        return tuple(
            dimension.signal for dimension in self.dimensions if dimension.signal is not None
        )

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(
            dimension.source for dimension in self.dimensions if dimension.source is not None
        )


class ArchiveCellDetector:
    """Encode semantic signals or provider sources into deterministic cell keys."""

    def __init__(self, config: ArchiveCellConfig):
        self.config = config
        self.selectors = tuple(dim.selector for dim in config.dimensions)
        self.parameters = np.asarray(
            [
                [
                    1 if dim.equals is not None else 2 if dim.zero_separate else 0,
                    dim.equals if dim.equals is not None else dim.bucket_size,
                    dim.clamp[0] if dim.clamp else -np.inf,
                    dim.clamp[1] if dim.clamp else np.inf,
                ]
                for dim in config.dimensions
            ],
            dtype=np.float64,
        )

    def indices(
        self,
        values_by_selector: Mapping[tuple[str, str], Any],
        *,
        n_envs: int,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Owned integer assignments; inactive lanes never require signal values."""
        columns = []
        for dimension in self.config.dimensions:
            values = np.asarray(values_by_selector[dimension.selector])
            if values.shape != (n_envs,):
                raise ValueError(f"cell values must have shape ({n_envs},)")
            selected = values if mask is None else values[mask]
            if selected.dtype.kind not in "biuf":
                if selected.dtype.kind != "O" or any(
                    not isinstance(item, (int, float, np.number, bool)) for item in selected
                ):
                    raise ValueError("cell values must be numeric")
            columns.append(selected)
        return bucket_columns(np.column_stack(columns).astype(np.float64), self.parameters)

    def keys(
        self,
        values_by_selector: Mapping[tuple[str, str], Any],
        *,
        n_envs: int,
    ) -> tuple[bytes, ...]:
        rows = self.indices(values_by_selector, n_envs=n_envs).tolist()
        return self.keys_from_indices(rows)

    def keys_from_indices(self, rows) -> tuple[bytes, ...]:
        if isinstance(rows, np.ndarray):
            rows = rows.tolist()
        if len(self.config.dimensions) == 1:
            dimension = self.config.dimensions[0]
            if (
                dimension.signal is not None
                and dimension.clamp is None
                and dimension.equals is None
                and not dimension.zero_separate
            ):
                return tuple(f"{dimension.signal}:{row[0]}".encode("ascii") for row in rows)
        return tuple(canonical_json_bytes(row) for row in rows)
