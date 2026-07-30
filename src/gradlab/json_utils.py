from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any


def json_value(value: Any) -> Any:
    if hasattr(value, "name") and hasattr(value, "value"):
        return str(value.name)
    if hasattr(value, "tolist") and type(value).__module__.startswith("numpy"):
        return json_value(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return [json_value(item) for item in value]
    return value


def canonical_json_text(
    value: Any,
    *,
    default: Callable[[Any], Any] | None = None,
    allow_nan: bool = False,
    ensure_ascii: bool = False,
) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=ensure_ascii,
        allow_nan=allow_nan,
        default=default,
    )


def canonical_json_bytes(
    value: Any,
    *,
    default: Callable[[Any], Any] | None = None,
    allow_nan: bool = False,
    ensure_ascii: bool = False,
) -> bytes:
    return canonical_json_text(
        value,
        default=default,
        allow_nan=allow_nan,
        ensure_ascii=ensure_ascii,
    ).encode("utf-8")


def canonical_json_line_bytes(
    value: Any,
    *,
    default: Callable[[Any], Any] | None = None,
    allow_nan: bool = False,
    ensure_ascii: bool = False,
) -> bytes:
    return canonical_json_bytes(
        value,
        default=default,
        allow_nan=allow_nan,
        ensure_ascii=ensure_ascii,
    ) + b"\n"


def canonical_json_sha256(
    value: Any,
    *,
    default: Callable[[Any], Any] | None = None,
    allow_nan: bool = False,
    ensure_ascii: bool = False,
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            value,
            default=default,
            allow_nan=allow_nan,
            ensure_ascii=ensure_ascii,
        )
    ).hexdigest()


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(nested) for key, nested in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(nested) for nested in value]
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        try:
            return json_safe(value.tolist())
        except (TypeError, ValueError):
            pass
    return str(value)
