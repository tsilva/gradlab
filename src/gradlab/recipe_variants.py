from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from gradlab.config_loader import dotlist_to_mapping
from gradlab.json_utils import canonical_json_sha256


BASE_RECIPE_VARIANT_ID = "base"
RECIPE_VARIANT_ID_PREFIX = "v-"
RECIPE_VARIANT_DIGEST_LENGTH = 8


def normalize_recipe_overrides(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        if text.startswith("["):
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError:
                pass
            else:
                return normalize_recipe_overrides(decoded)
        return (text,)
    if isinstance(value, Mapping):
        return tuple(
            f"{str(key)}={json.dumps(item, sort_keys=True, separators=(',', ':'))}"
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, Sequence) and not isinstance(
        value,
        (bytes, bytearray, memoryview),
    ):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def recipe_variant_id(
    *,
    recipe_slug: object,
    source_sha: object,
    recipe_overrides: object,
) -> str:
    overrides = normalize_recipe_overrides(recipe_overrides)
    if not overrides:
        return BASE_RECIPE_VARIANT_ID
    try:
        canonical_overrides: Any = dotlist_to_mapping(
            overrides,
            label="recipe variant overrides",
        )
    except ValueError:
        canonical_overrides = sorted(overrides)
    digest = canonical_json_sha256(
        {
            "recipe_slug": str(recipe_slug or "").strip(),
            "source_sha": str(source_sha or "").strip(),
            "recipe_overrides": canonical_overrides,
        }
    )
    return f"{RECIPE_VARIANT_ID_PREFIX}{digest[:RECIPE_VARIANT_DIGEST_LENGTH]}"
