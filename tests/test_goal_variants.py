from __future__ import annotations

from copy import deepcopy

import pytest

from gradlab.goal_variants import (
    build_goal_variant_descriptor,
    goal_variant_id,
    goal_variant_projection,
    validate_goal_variant_descriptor,
)


def goal_document() -> dict[str, object]:
    return {
        "goal_id": "Level1-1",
        "title": "Mario Level1-1",
        "train": {
            "environment": {
                "task": {
                    "action": {
                        "sticky_probability": 0,
                    }
                }
            }
        },
        "eval": {
            "environment": {
                "task": {
                    "action": {
                        "sticky_probability": 0,
                    }
                }
            }
        },
    }


def test_goal_variant_identity_uses_authored_and_effective_contracts() -> None:
    authored = goal_document()
    effective = deepcopy(authored)
    effective["train"]["environment"]["task"]["action"][  # type: ignore[index]
        "sticky_probability"
    ] = 0.25
    effective["eval"]["environment"]["task"]["action"][  # type: ignore[index]
        "sticky_probability"
    ] = 0.25
    descriptor = build_goal_variant_descriptor(
        goal_slug="SuperMarioBros-Nes-v0/Level1-1",
        source_sha="a" * 40,
        authored_goal=authored,
        effective_goal=effective,
    )

    assert descriptor["source_relation"] == "changed"
    assert descriptor["variant_id"] == goal_variant_id(
        goal_slug=descriptor["goal_slug"],
        goal_contract_sha256_value=descriptor["goal_contract_sha256"],
        effective_goal_contract_sha256=descriptor["effective_goal_contract_sha256"],
    )
    assert "sticky probability 0 → 0.25" in descriptor["label"]
    assert descriptor["diff"] == [
        {
            "path": "train+eval.environment.task.action.sticky_probability",
            "before": 0,
            "after": 0.25,
            "kind": "changed",
        }
    ]
    projection = goal_variant_projection(descriptor)
    assert projection["goal_variant_id"] == descriptor["variant_id"]
    assert "sticky_probability" in projection["goal_variant_diff_json"]


def test_goal_variant_identity_preserves_canonical_json_hash_bytes() -> None:
    assert (
        goal_variant_id(
            goal_slug="SuperMarioBros-Nes-v0/Level1-1",
            goal_contract_sha256_value="1" * 64,
            effective_goal_contract_sha256="2" * 64,
        )
        == "goal-variant-a4af1a829454a35e02f38c72"
    )


def test_presentation_only_changes_do_not_fabricate_a_goal_diff() -> None:
    authored = goal_document()
    effective = deepcopy(authored)
    effective["title"] = "Renamed for display"
    descriptor = build_goal_variant_descriptor(
        goal_slug="SuperMarioBros-Nes-v0/Level1-1",
        source_sha="b" * 40,
        authored_goal=authored,
        effective_goal=effective,
    )

    assert descriptor["source_relation"] == "changed"
    assert descriptor["diff"] == []
    assert descriptor["label"] == "Mario Level1-1"


def test_descriptor_rejects_identity_tampering() -> None:
    authored = goal_document()
    descriptor = build_goal_variant_descriptor(
        goal_slug="SuperMarioBros-Nes-v0/Level1-1",
        source_sha="c" * 40,
        authored_goal=authored,
        effective_goal=authored,
    )
    descriptor["variant_id"] = "goal-variant-" + "0" * 24

    with pytest.raises(ValueError, match="identity mismatch"):
        validate_goal_variant_descriptor(descriptor)
