from __future__ import annotations

from rlab.recipe_variants import (
    BASE_RECIPE_VARIANT_ID,
    normalize_recipe_overrides,
    recipe_variant_id,
)


def test_base_recipe_variant_is_explicit() -> None:
    assert (
        recipe_variant_id(
            recipe_slug="ppo",
            source_sha="a" * 40,
            recipe_overrides=[],
        )
        == BASE_RECIPE_VARIANT_ID
    )


def test_recipe_variant_identity_is_stable_across_override_order() -> None:
    left = recipe_variant_id(
        recipe_slug="ppo",
        source_sha="a" * 40,
        recipe_overrides=[
            "train.backend.config.learning_rate=0.0002",
            "train.backend.config.batch_size=256",
        ],
    )
    right = recipe_variant_id(
        recipe_slug="ppo",
        source_sha="a" * 40,
        recipe_overrides=[
            "train.backend.config.batch_size=256",
            "train.backend.config.learning_rate=0.0002",
        ],
    )

    assert left == right
    assert left.startswith("v-")


def test_wandb_json_recipe_overrides_remain_exact() -> None:
    assert normalize_recipe_overrides('["train.backend.config.learning_rate=0.0002"]') == (
        "train.backend.config.learning_rate=0.0002",
    )
