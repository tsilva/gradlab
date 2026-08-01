from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from gradlab.goal_variants import validate_goal_variant_descriptor
from gradlab.json_utils import canonical_json_sha256
from gradlab.recipe_documents import goal_contract_sha256
from gradlab.run_contracts import RUN_ID_PATTERN, SHA256_PATTERN


CATALOG_GENERATION_SCHEMA_VERSION = 1
CATALOG_POINTER_SCHEMA_VERSION = 1
CATALOG_V3_ROOT = "goal-variants/v3"
CATALOG_POINTER_KEY = f"{CATALOG_V3_ROOT}/current.json"
CATALOG_VARIANT_PROJECTION_FIELDS = frozenset(
    {
        "descriptor_key",
        "first_run_id",
        "exact_resolution_run_id",
        "resolved_goal",
    }
)


def catalog_generation_key(digest: str) -> str:
    normalized = str(digest or "").strip().lower()
    if SHA256_PATTERN.fullmatch(normalized) is None:
        raise ValueError("catalog generation requires a lowercase SHA-256")
    return f"{CATALOG_V3_ROOT}/generations/{normalized}.json"


def catalog_generation_digest(document: Mapping[str, Any]) -> str:
    return canonical_json_sha256(dict(document))


def validate_catalog_generation(
    document: Mapping[str, Any],
    *,
    expected_digest: str | None = None,
) -> dict[str, Any]:
    if int(document.get("schema_version") or 0) != CATALOG_GENERATION_SCHEMA_VERSION:
        raise ValueError("unsupported catalog generation schema")
    generated_at = str(document.get("generated_at") or "").strip()
    if not generated_at:
        raise ValueError("catalog generation has no generated_at")
    raw_scopes = document.get("scopes")
    if not isinstance(raw_scopes, list):
        raise ValueError("catalog generation scopes must be a list")
    scopes: list[dict[str, Any]] = []
    seen_scopes: set[str] = set()
    for raw_scope in raw_scopes:
        if not isinstance(raw_scope, Mapping):
            raise ValueError("catalog generation contains an invalid scope")
        goal_slug = str(raw_scope.get("goal_slug") or "").strip()
        if not goal_slug or goal_slug in seen_scopes:
            raise ValueError("catalog generation contains an invalid or duplicate scope")
        seen_scopes.add(goal_slug)
        raw_variants = raw_scope.get("variants")
        raw_runs = raw_scope.get("runs")
        if not isinstance(raw_variants, list) or not isinstance(raw_runs, list):
            raise ValueError("catalog generation scope lists are malformed")
        variants: list[dict[str, Any]] = []
        variant_ids: set[str] = set()
        for raw_variant in raw_variants:
            if not isinstance(raw_variant, Mapping):
                raise ValueError("catalog generation contains an invalid variant")
            descriptor = {
                key: value
                for key, value in raw_variant.items()
                if key not in CATALOG_VARIANT_PROJECTION_FIELDS
            }
            validated = validate_goal_variant_descriptor(descriptor)
            variant_id = str(validated["variant_id"])
            if variant_id in variant_ids:
                raise ValueError("catalog generation contains a duplicate variant")
            resolved_goal = raw_variant.get("resolved_goal")
            if resolved_goal is not None:
                if not isinstance(resolved_goal, Mapping):
                    raise ValueError("catalog variant resolved goal must be an object")
                if (
                    goal_contract_sha256(resolved_goal)
                    != validated["effective_goal_contract_sha256"]
                ):
                    raise ValueError(
                        "catalog variant resolved goal does not match its effective contract"
                    )
            variant_ids.add(variant_id)
            variants.append(deepcopy(dict(raw_variant)))
        runs: list[dict[str, Any]] = []
        run_ids: set[str] = set()
        for raw_run in raw_runs:
            if not isinstance(raw_run, Mapping):
                raise ValueError("catalog generation contains an invalid run")
            run_id = str(raw_run.get("run_id") or "")
            variant_id = str(raw_run.get("goal_variant_id") or "")
            if (
                RUN_ID_PATTERN.fullmatch(run_id) is None
                or run_id in run_ids
                or variant_id not in variant_ids
                or raw_run.get("goal_slug") != goal_slug
                or not isinstance(raw_run.get("metrics"), Mapping)
            ):
                raise ValueError("catalog generation contains an invalid or duplicate run")
            run_ids.add(run_id)
            runs.append(dict(raw_run))
        scopes.append(
            {
                "goal_slug": goal_slug,
                "variants": variants,
                "runs": runs,
            }
        )
    validated_document = {
        "schema_version": CATALOG_GENERATION_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scopes": scopes,
    }
    digest = catalog_generation_digest(validated_document)
    if expected_digest is not None and digest != str(expected_digest).strip().lower():
        raise ValueError("catalog generation content does not match its pointer")
    return deepcopy(validated_document)


def validate_catalog_pointer(document: Mapping[str, Any]) -> dict[str, Any]:
    if int(document.get("schema_version") or 0) != CATALOG_POINTER_SCHEMA_VERSION:
        raise ValueError("unsupported catalog pointer schema")
    digest = str(document.get("generation_sha256") or "").strip().lower()
    key = str(document.get("generation_key") or "").strip()
    generated_at = str(document.get("generated_at") or "").strip()
    if (
        SHA256_PATTERN.fullmatch(digest) is None
        or key != catalog_generation_key(digest)
        or not generated_at
    ):
        raise ValueError("catalog pointer is malformed")
    return {
        "schema_version": CATALOG_POINTER_SCHEMA_VERSION,
        "generation_sha256": digest,
        "generation_key": key,
        "generated_at": generated_at,
    }


def empty_catalog_generation(*, generated_at: str) -> dict[str, Any]:
    return {
        "schema_version": CATALOG_GENERATION_SCHEMA_VERSION,
        "generated_at": str(generated_at),
        "scopes": [],
    }


__all__ = [
    "CATALOG_GENERATION_SCHEMA_VERSION",
    "CATALOG_POINTER_KEY",
    "CATALOG_POINTER_SCHEMA_VERSION",
    "CATALOG_VARIANT_PROJECTION_FIELDS",
    "CATALOG_V3_ROOT",
    "catalog_generation_digest",
    "catalog_generation_key",
    "empty_catalog_generation",
    "validate_catalog_generation",
    "validate_catalog_pointer",
]
