from __future__ import annotations

import pytest

from gradlab.contract_inspection import inspection_document, structural_changes


def test_structural_changes_are_typed_and_use_json_pointers() -> None:
    changes = structural_changes(
        {"train": {"a/b": 1, "removed": True}, "items": ["a"]},
        {"train": {"a/b": 2, "added": False}, "items": ["a", "b"]},
    )

    assert changes == [
        {
            "path": "/items/1",
            "kind": "added",
            "before": None,
            "after": "b",
        },
        {
            "path": "/train/a~1b",
            "kind": "changed",
            "before": 1,
            "after": 2,
        },
        {
            "path": "/train/added",
            "kind": "added",
            "before": None,
            "after": False,
        },
        {
            "path": "/train/removed",
            "kind": "removed",
            "before": True,
            "after": None,
        },
    ]


def test_inspection_document_renders_yaml_and_diff_from_the_same_pair() -> None:
    document = inspection_document(
        kind="recipe",
        title="PPO",
        availability="exact",
        base={"train": {"gamma": 0.99}},
        resolved={"train": {"gamma": 0.97}},
        variant_id="v-12345678",
    )

    assert document["is_variant"] is True
    assert "gamma: 0.99" in document["views"]["base"]
    assert "gamma: 0.97" in document["views"]["resolved"]
    assert "-  gamma: 0.99" in document["views"]["changes"]["unified_diff"]
    assert "+  gamma: 0.97" in document["views"]["changes"]["unified_diff"]


def test_static_preview_may_keep_launch_placeholders_but_exact_contract_may_not() -> None:
    preview = inspection_document(
        kind="recipe",
        title="PPO",
        availability="static-preview",
        resolved={"description": "seed {{ seed }}"},
        allow_placeholders=True,
    )
    assert "{{ seed }}" in preview["views"]["resolved"]

    with pytest.raises(ValueError, match="unresolved interpolation"):
        inspection_document(
            kind="recipe",
            title="PPO",
            availability="exact",
            resolved={"description": "seed {{ seed }}"},
        )


def test_secret_like_fields_are_never_rendered() -> None:
    with pytest.raises(ValueError, match="secret-like"):
        inspection_document(
            kind="goal",
            title="unsafe",
            availability="exact",
            resolved={"api_token": "do-not-render"},
        )
