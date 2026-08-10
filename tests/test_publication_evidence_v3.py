from __future__ import annotations

from copy import deepcopy

import pytest

from gradlab.publication_evidence import validate_evaluation_evidence_document


def document() -> dict:
    return {
        "document_type": "gradlab.evaluation_evidence",
        "format_version": 2,
        "tier": "research",
        "status": "accepted",
        "provenance": {"origin": "gradlab-verified-evaluation"},
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


def test_historical_evidence_must_preserve_failed_acceptance() -> None:
    historical = deepcopy(document())
    historical["tier"] = "historical-import"
    historical["status"] = "evaluated-not-accepted"
    historical["acceptance"]["passed"] = False
    historical["acceptance"]["outcomes"][0]["passed"] = False
    historical["provenance"] = {"origin": "legacy-exact-contract-rerun"}
    assert validate_evaluation_evidence_document(historical) == historical

    historical["acceptance"]["passed"] = True
    with pytest.raises(ValueError, match="failed acceptance"):
        validate_evaluation_evidence_document(historical)
