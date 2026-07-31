from __future__ import annotations

import difflib
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, Literal

import yaml

from gradlab.json_utils import canonical_json_bytes
from gradlab.validation import is_secret_like_key


INSPECTION_SCHEMA_VERSION = 1
MAX_INSPECTION_DOCUMENT_BYTES = 1024 * 1024
Availability = Literal["exact", "static-preview", "summary-only", "unavailable"]

def _json_pointer(parts: Sequence[str]) -> str:
    if not parts:
        return ""
    return "/" + "/".join(part.replace("~", "~0").replace("/", "~1") for part in parts)


def structural_changes(
    base: object,
    resolved: object,
    *,
    path: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Return deterministic, typed changes addressed by JSON Pointer."""

    if isinstance(base, Mapping) and isinstance(resolved, Mapping):
        changes: list[dict[str, Any]] = []
        keys = sorted({str(key) for key in base} | {str(key) for key in resolved})
        for key in keys:
            nested = (*path, key)
            if key not in base:
                changes.append(
                    {
                        "path": _json_pointer(nested),
                        "kind": "added",
                        "before": None,
                        "after": deepcopy(resolved[key]),
                    }
                )
            elif key not in resolved:
                changes.append(
                    {
                        "path": _json_pointer(nested),
                        "kind": "removed",
                        "before": deepcopy(base[key]),
                        "after": None,
                    }
                )
            else:
                changes.extend(structural_changes(base[key], resolved[key], path=nested))
        return changes
    if (
        isinstance(base, Sequence)
        and not isinstance(base, str | bytes)
        and isinstance(resolved, Sequence)
        and not isinstance(resolved, str | bytes)
    ):
        changes = []
        shared = min(len(base), len(resolved))
        for index in range(shared):
            changes.extend(
                structural_changes(
                    base[index],
                    resolved[index],
                    path=(*path, str(index)),
                )
            )
        for index in range(shared, len(base)):
            changes.append(
                {
                    "path": _json_pointer((*path, str(index))),
                    "kind": "removed",
                    "before": deepcopy(base[index]),
                    "after": None,
                }
            )
        for index in range(shared, len(resolved)):
            changes.append(
                {
                    "path": _json_pointer((*path, str(index))),
                    "kind": "added",
                    "before": None,
                    "after": deepcopy(resolved[index]),
                }
            )
        return changes
    if base == resolved:
        return []
    return [
        {
            "path": _json_pointer(path),
            "kind": "changed",
            "before": deepcopy(base),
            "after": deepcopy(resolved),
        }
    ]


def _assert_safe_value(
    value: object,
    *,
    label: str,
    allow_placeholders: bool,
) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if is_secret_like_key(key):
                raise ValueError(f"{label}.{key} is secret-like and cannot be inspected")
            _assert_safe_value(
                nested,
                label=f"{label}.{key}",
                allow_placeholders=allow_placeholders,
            )
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for index, nested in enumerate(value):
            _assert_safe_value(
                nested,
                label=f"{label}[{index}]",
                allow_placeholders=allow_placeholders,
            )
        return
    if isinstance(value, float) and (value != value or abs(value) == float("inf")):
        raise ValueError(f"{label} contains a non-finite number")
    if not isinstance(value, str):
        return
    if not allow_placeholders and ("${" in value or "{{" in value or "}}" in value):
        raise ValueError(f"{label} contains unresolved interpolation")


def yaml_text(
    value: object,
    *,
    label: str,
    allow_placeholders: bool = False,
) -> str:
    _assert_safe_value(value, label=label, allow_placeholders=allow_placeholders)
    try:
        rendered = yaml.safe_dump(
            deepcopy(value),
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=True,
        )
    except yaml.YAMLError as exc:
        raise ValueError(f"{label} cannot be rendered as safe YAML") from exc
    if len(rendered.encode("utf-8")) > MAX_INSPECTION_DOCUMENT_BYTES:
        raise ValueError(f"{label} exceeds the inspection size limit")
    return rendered


def inspection_document(
    *,
    kind: Literal["goal", "recipe"],
    title: str,
    availability: Availability,
    resolved: Mapping[str, Any] | None = None,
    base: Mapping[str, Any] | None = None,
    variant_id: str | None = None,
    message: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    allow_placeholders: bool = False,
) -> dict[str, Any]:
    if availability in {"exact", "static-preview"} and resolved is None:
        raise ValueError(f"{availability} inspection requires a resolved contract")
    if availability in {"summary-only", "unavailable"} and resolved is not None:
        raise ValueError(f"{availability} inspection cannot claim a resolved contract")

    resolved_yaml = (
        yaml_text(
            resolved,
            label=f"{kind} resolved contract",
            allow_placeholders=allow_placeholders,
        )
        if resolved is not None
        else None
    )
    base_yaml = (
        yaml_text(
            base,
            label=f"{kind} base contract",
            allow_placeholders=allow_placeholders,
        )
        if base is not None
        else None
    )
    changes = (
        structural_changes(base, resolved) if base is not None and resolved is not None else []
    )
    unified = ""
    if base_yaml is not None and resolved_yaml is not None:
        unified = "".join(
            difflib.unified_diff(
                base_yaml.splitlines(keepends=True),
                resolved_yaml.splitlines(keepends=True),
                fromfile=f"{kind}-base.yaml",
                tofile=f"{kind}-resolved.yaml",
            )
        )
    document = {
        "schema_version": INSPECTION_SCHEMA_VERSION,
        "kind": kind,
        "title": str(title).strip() or kind.title(),
        "availability": availability,
        "variant_id": str(variant_id or ""),
        "is_variant": bool(changes),
        "message": str(message or ""),
        "metadata": deepcopy(dict(metadata or {})),
        "views": {
            "resolved": resolved_yaml,
            "base": base_yaml,
            "changes": {
                "unified_diff": unified,
                "entries": changes,
            },
        },
    }
    encoded = canonical_json_bytes(document)
    if len(encoded) > MAX_INSPECTION_DOCUMENT_BYTES * 3:
        raise ValueError("inspection response exceeds the supported size")
    return document
