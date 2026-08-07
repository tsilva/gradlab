from __future__ import annotations

from copy import deepcopy

import pytest

from gradlab.publication_evidence import validate_evaluation_evidence_document


def document() -> dict:
    return {
        "document_type": "gradlab.evaluation_evidence",
        "format_version": 1,
        "status": "accepted",
        "identity": {"checkpoint_step": 10},
        "protocol": {"episodes": 2, "action_sampling": "stochastic"},
        "episode_results": [{"episode": 0}, {"episode": 1}],
        "aggregates": {"eval/full/progress/kills/mean": 12.5},
        "acceptance": {
            "rules": [],
            "outcomes": [{"metric": "eval/full/progress/kills/mean", "passed": True}],
            "passed": True,
        },
        "ranking": {"rules": [], "outcomes": []},
        "contracts": {"materialized_goal": {}, "evaluation": {}, "environment": {}},
        "authoritative_hashes": {"verified_result_sha256": "a" * 64},
    }


def test_evaluation_evidence_requires_complete_episode_results() -> None:
    assert validate_evaluation_evidence_document(document()) == document()
    incomplete = deepcopy(document())
    incomplete["episode_results"].pop()
    with pytest.raises(ValueError, match="episode count is incomplete"):
        validate_evaluation_evidence_document(incomplete)


def test_evaluation_evidence_rejects_unknown_contract_fields() -> None:
    changed = deepcopy(document())
    changed["legacy_alias"] = True
    with pytest.raises(ValueError, match="unsupported field set"):
        validate_evaluation_evidence_document(changed)
