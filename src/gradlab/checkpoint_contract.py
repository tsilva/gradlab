from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from gradlab.policy_bundle import canonical_json_sha256, evaluation_contract_sha256


def checkpoint_manifest_contract_sha256(recipe_document: Mapping[str, Any]) -> str:
    """Return the contract digest persisted in a checkpoint manifest.

    Recipes with automatic evaluation persist their acceptance contract directly.
    Recipes that disabled automatic evaluation persist the playback contract that
    was active while training, even when their embedded goal remains eligible for
    a later operator-requested evaluation.
    """

    recipe = recipe_document.get("recipe")
    if not isinstance(recipe, Mapping):
        raise ValueError("recipe.json is missing its recipe mapping")
    if isinstance(recipe.get("eval"), Mapping):
        return evaluation_contract_sha256(recipe_document)
    playback = recipe.get("playback")
    if not isinstance(playback, Mapping):
        raise ValueError("recipe.json has neither an evaluation nor playback contract")
    return canonical_json_sha256(
        {
            "training_only": True,
            "playback": dict(playback),
        }
    )


__all__ = ["checkpoint_manifest_contract_sha256"]
