from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from gradlab.recipe_documents import goal_contract_sha256
from gradlab.reward_programs import goal_for_contract_validation


GOAL_VARIANT_SCHEMA_VERSION = 1
GOAL_VARIANT_INDEX_SCHEMA_VERSION = 1
GOAL_VARIANT_RUN_INDEX_SCHEMA_VERSION = 1
GOAL_VARIANT_ID_PATTERN = re.compile(r"^goal-variant-[0-9a-f]{24}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MAX_DIFF_ENTRIES = 24

_PRESENTATION_ROOTS = frozenset({"defaults", "metadata", "notes", "tags", "title"})


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def goal_variant_id(
    *,
    goal_slug: object,
    goal_contract_sha256_value: object,
    effective_goal_contract_sha256: object,
) -> str:
    identity = {
        "goal_slug": str(goal_slug or "").strip(),
        "goal_contract_sha256": str(goal_contract_sha256_value or "").strip().lower(),
        "effective_goal_contract_sha256": str(effective_goal_contract_sha256 or "").strip().lower(),
    }
    if not identity["goal_slug"]:
        raise ValueError("goal variant goal_slug must not be empty")
    for key in ("goal_contract_sha256", "effective_goal_contract_sha256"):
        if SHA256_PATTERN.fullmatch(identity[key]) is None:
            raise ValueError(f"goal variant {key} must be a lowercase SHA-256")
    digest = hashlib.sha256(_canonical_json(identity)).hexdigest()
    return f"goal-variant-{digest[:24]}"


def goal_variant_scope_key(*, entity: object, project: object, goal_slug: object) -> str:
    identity = {
        "entity": str(entity or "").strip(),
        "project": str(project or "").strip(),
        "goal_slug": str(goal_slug or "").strip(),
    }
    if not all(identity.values()):
        raise ValueError("goal variant scope requires entity, project, and goal_slug")
    digest = hashlib.sha256(_canonical_json(identity)).hexdigest()
    return f"goal-variants/v1/scopes/{digest}"


def _path_text(parts: Sequence[str]) -> str:
    return ".".join(parts)


def _excluded_path(parts: Sequence[str]) -> bool:
    return bool(parts) and parts[0] in _PRESENTATION_ROOTS


def _compact_value(value: object) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    rendered = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return rendered if len(rendered) <= 160 else rendered[:157] + "…"


def _contract_diff(
    before: object,
    after: object,
    *,
    path: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    if _excluded_path(path):
        return []
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        rows: list[dict[str, Any]] = []
        keys = sorted({str(key) for key in before} | {str(key) for key in after})
        for key in keys:
            in_before = key in before
            in_after = key in after
            nested_path = (*path, key)
            if _excluded_path(nested_path):
                continue
            if not in_before:
                rows.append(
                    {
                        "path": _path_text(nested_path),
                        "before": None,
                        "after": _compact_value(after[key]),
                        "kind": "added",
                    }
                )
            elif not in_after:
                rows.append(
                    {
                        "path": _path_text(nested_path),
                        "before": _compact_value(before[key]),
                        "after": None,
                        "kind": "removed",
                    }
                )
            else:
                rows.extend(_contract_diff(before[key], after[key], path=nested_path))
        return rows
    if (
        isinstance(before, Sequence)
        and not isinstance(before, str | bytes)
        and isinstance(after, Sequence)
        and not isinstance(after, str | bytes)
    ):
        if list(before) == list(after):
            return []
    elif before == after:
        return []
    return [
        {
            "path": _path_text(path),
            "before": _compact_value(before),
            "after": _compact_value(after),
            "kind": "changed",
        }
    ]


def _display_path(value: object) -> str:
    path = str(value or "")
    replacements = (
        ("train.environment.env_config.", ""),
        ("eval.environment.env_config.", ""),
        ("train+eval.environment.env_config.", ""),
        ("train.environment.task.", ""),
        ("eval.environment.task.", ""),
        ("train+eval.environment.task.", ""),
        ("train.environment.", ""),
        ("eval.environment.", ""),
    )
    for prefix, replacement in replacements:
        if path.startswith(prefix):
            path = replacement + path[len(prefix) :]
            break
    return path.replace("_", " ")


def _collapse_phase_diff(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    pending = [dict(entry) for entry in entries]
    collapsed: list[dict[str, Any]] = []
    while pending:
        entry = pending.pop(0)
        path = str(entry["path"])
        phase, separator, suffix = path.partition(".")
        counterpart_phase = "eval" if phase == "train" else "train" if phase == "eval" else ""
        counterpart_index = next(
            (
                index
                for index, candidate in enumerate(pending)
                if counterpart_phase
                and candidate["path"] == f"{counterpart_phase}.{suffix}"
                and {key: candidate[key] for key in ("before", "after", "kind")}
                == {key: entry[key] for key in ("before", "after", "kind")}
            ),
            None,
        )
        if separator and counterpart_index is not None:
            pending.pop(counterpart_index)
            entry["path"] = f"train+eval.{suffix}"
        collapsed.append(entry)
    return collapsed


def _display_value(value: object) -> str:
    if value is None:
        return "unset"
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _diff_label(entry: Mapping[str, Any]) -> str:
    path = _display_path(entry.get("path"))
    kind = str(entry.get("kind") or "")
    if kind == "added":
        return f"{path} → {_display_value(entry.get('after'))}"
    if kind == "removed":
        return f"{path} {_display_value(entry.get('before'))} → unset"
    return f"{path} {_display_value(entry.get('before'))} → {_display_value(entry.get('after'))}"


def build_goal_variant_descriptor(
    *,
    goal_slug: object,
    source_sha: object,
    authored_goal: Mapping[str, Any],
    effective_goal: Mapping[str, Any],
) -> dict[str, Any]:
    slug = str(goal_slug or "").strip()
    source = str(source_sha or "").strip().lower()
    if not slug:
        raise ValueError("goal variant goal_slug must not be empty")
    if source and re.fullmatch(r"[0-9a-f]{40}", source) is None:
        raise ValueError("goal variant source_sha must be a full lowercase Git SHA")
    authored = deepcopy(dict(authored_goal))
    effective = deepcopy(dict(effective_goal))
    canonical_effective = goal_for_contract_validation(
        authored,
        label=f"goal variant {slug}",
    )
    authored_hash = goal_contract_sha256(authored)
    effective_hash = goal_contract_sha256(effective)
    canonical_effective_hash = goal_contract_sha256(canonical_effective)
    identifier = goal_variant_id(
        goal_slug=slug,
        goal_contract_sha256_value=authored_hash,
        effective_goal_contract_sha256=effective_hash,
    )
    raw_diff = _collapse_phase_diff(_contract_diff(canonical_effective, effective))
    truncated = len(raw_diff) > MAX_DIFF_ENTRIES
    diff = raw_diff[:MAX_DIFF_ENTRIES]
    goal_id = str(authored.get("goal_id") or slug.rsplit("/", 1)[-1]).strip()
    goal_name = str(authored.get("title") or goal_id).strip()
    relation = "canonical" if effective_hash == canonical_effective_hash else "changed"
    labels = [_diff_label(entry) for entry in diff[:3]]
    if truncated:
        labels.append(f"+{len(raw_diff) - len(diff)} more")
    label = goal_name if not labels else f"{goal_name} · {' · '.join(labels)}"
    descriptor = {
        "schema_version": GOAL_VARIANT_SCHEMA_VERSION,
        "variant_id": identifier,
        "goal_slug": slug,
        "goal_id": goal_id,
        "goal_name": goal_name,
        "source_sha": source,
        "goal_contract_sha256": authored_hash,
        "effective_goal_contract_sha256": effective_hash,
        "canonical_effective_goal_contract_sha256": canonical_effective_hash,
        "source_relation": relation,
        "label": label,
        "diff": diff,
        "diff_truncated": truncated,
    }
    validate_goal_variant_descriptor(descriptor)
    return descriptor


def validate_goal_variant_descriptor(value: Mapping[str, Any]) -> dict[str, Any]:
    document = dict(value)
    allowed = {
        "schema_version",
        "variant_id",
        "goal_slug",
        "goal_id",
        "goal_name",
        "source_sha",
        "goal_contract_sha256",
        "effective_goal_contract_sha256",
        "canonical_effective_goal_contract_sha256",
        "source_relation",
        "label",
        "diff",
        "diff_truncated",
    }
    unknown = set(document) - allowed
    if unknown:
        raise ValueError(
            "goal variant descriptor has unknown fields: "
            + ", ".join(sorted(str(key) for key in unknown))
        )
    if int(document.get("schema_version") or 0) != GOAL_VARIANT_SCHEMA_VERSION:
        raise ValueError("unsupported goal variant descriptor schema")
    for field in ("goal_slug", "goal_id", "goal_name", "label"):
        if not str(document.get(field) or "").strip():
            raise ValueError(f"goal variant {field} must not be empty")
    source = str(document.get("source_sha") or "")
    if source and re.fullmatch(r"[0-9a-f]{40}", source) is None:
        raise ValueError("goal variant source_sha must be a full lowercase Git SHA")
    for field in (
        "goal_contract_sha256",
        "effective_goal_contract_sha256",
        "canonical_effective_goal_contract_sha256",
    ):
        if SHA256_PATTERN.fullmatch(str(document.get(field) or "")) is None:
            raise ValueError(f"goal variant {field} must be a lowercase SHA-256")
    expected_id = goal_variant_id(
        goal_slug=document["goal_slug"],
        goal_contract_sha256_value=document["goal_contract_sha256"],
        effective_goal_contract_sha256=document["effective_goal_contract_sha256"],
    )
    if document.get("variant_id") != expected_id:
        raise ValueError("goal variant descriptor identity mismatch")
    if document.get("source_relation") not in {"canonical", "changed"}:
        raise ValueError("goal variant source_relation must be canonical or changed")
    diff = document.get("diff")
    if isinstance(diff, str | bytes) or not isinstance(diff, Sequence):
        raise ValueError("goal variant diff must be a sequence")
    if len(diff) > MAX_DIFF_ENTRIES:
        raise ValueError("goal variant diff exceeds its bounded size")
    for entry in diff:
        if not isinstance(entry, Mapping):
            raise ValueError("goal variant diff entry must be a mapping")
        if set(entry) != {"path", "before", "after", "kind"}:
            raise ValueError("goal variant diff entry fields are invalid")
        if not str(entry.get("path") or "").strip():
            raise ValueError("goal variant diff path must not be empty")
        if entry.get("kind") not in {"added", "changed", "removed"}:
            raise ValueError("goal variant diff kind is invalid")
    if not isinstance(document.get("diff_truncated"), bool):
        raise ValueError("goal variant diff_truncated must be a boolean")
    return deepcopy(document)


def goal_variant_projection(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    validated = validate_goal_variant_descriptor(descriptor)
    descriptor_sha = hashlib.sha256(_canonical_json(validated)).hexdigest()
    return {
        "goal_variant_id": validated["variant_id"],
        "goal_variant_label": validated["label"],
        "goal_variant_source_relation": validated["source_relation"],
        "goal_variant_descriptor_sha256": descriptor_sha,
        "goal_variant_diff_json": json.dumps(
            validated["diff"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ),
    }


def unknown_goal_variant_id(*, goal_slug: object) -> str:
    slug = str(goal_slug or "").strip()
    if not slug:
        raise ValueError("unknown goal variant requires goal_slug")
    digest = hashlib.sha256(f"unknown-goal-variant-v1:{slug}".encode()).hexdigest()
    return f"goal-variant-unknown-{digest[:16]}"


__all__ = [
    "GOAL_VARIANT_ID_PATTERN",
    "GOAL_VARIANT_INDEX_SCHEMA_VERSION",
    "GOAL_VARIANT_RUN_INDEX_SCHEMA_VERSION",
    "GOAL_VARIANT_SCHEMA_VERSION",
    "build_goal_variant_descriptor",
    "goal_variant_id",
    "goal_variant_projection",
    "goal_variant_scope_key",
    "unknown_goal_variant_id",
    "validate_goal_variant_descriptor",
]
