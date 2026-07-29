from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def label_path(label: str, key: str) -> str:
    return f"{label}.{key}" if label else key


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def require_int(
    document: Mapping[str, Any],
    key: str,
    *,
    label: str,
    minimum: int | None = None,
    require_present: bool = True,
) -> int:
    value = require_key(document, key, label=label) if require_present else document.get(key)
    if not is_int(value):
        raise ValueError(f"{label_path(label, key)} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label_path(label, key)} must be >= {minimum}")
    return value


def require_key(document: Mapping[str, Any], key: str, *, label: str) -> Any:
    if key not in document:
        raise ValueError(f"{label_path(label, key)} is required")
    return document[key]


def require_non_empty_string(
    document: Mapping[str, Any],
    key: str,
    *,
    label: str,
    require_present: bool = True,
    strip: bool = True,
) -> str:
    value = require_key(document, key, label=label) if require_present else document.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label_path(label, key)} must be a non-empty string")
    return value.strip() if strip else value


def string_list(
    value: Any, *, label: str, allow_empty: bool = False, strip: bool = True
) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{label} must be a list")
    if not value and not allow_empty:
        raise ValueError(f"{label} must not be empty")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{label}[{index}] must be a non-empty string")
        result.append(item.strip() if strip else item)
    return result


def int_list(value: Any, *, label: str, allow_empty: bool = False) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{label} must be a list")
    if not value and not allow_empty:
        raise ValueError(f"{label} must not be empty")
    result: list[int] = []
    for index, item in enumerate(value):
        if not is_int(item):
            raise ValueError(f"{label}[{index}] must be an integer")
        result.append(item)
    return result


def normalize_obs_crop(
    value: Any,
    *,
    label: str = "obs_crop",
) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or len(value) != 4:
        raise ValueError(f"{label} must be [top, right, bottom, left]")
    result: list[int] = []
    for index, item in enumerate(value):
        if not is_int(item) or item < 0:
            raise ValueError(f"{label}[{index}] must be a non-negative integer")
        result.append(item)
    return tuple(result)  # type: ignore[return-value]


def normalize_obs_resize(
    value: Any,
    *,
    label: str = "obs_resize",
) -> tuple[int, int]:
    raw = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{label} must be [height, width]")
        raw = tuple(part.strip() for part in text.strip("[]").split(","))
        if len(raw) == 1:
            raw = (raw[0], raw[0])
        try:
            raw = tuple(int(part) for part in raw)
        except ValueError as exc:
            raise ValueError(f"{label} must be [height, width]") from exc
    elif is_int(value):
        raw = (value, value)
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes) or len(raw) != 2:
        raise ValueError(f"{label} must be [height, width]")
    result: list[int] = []
    for index, item in enumerate(raw):
        if not is_int(item) or item < 0:
            raise ValueError(f"{label}[{index}] must be a non-negative integer")
        result.append(item)
    if (result[0] == 0) != (result[1] == 0):
        raise ValueError(f"{label} dimensions must both be zero or both be positive")
    return result[0], result[1]
