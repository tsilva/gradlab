from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from gradlab.checkpoint_contract import checkpoint_manifest_contract_sha256
from gradlab.eval_metrics import eval_by_start_rows
from gradlab.json_utils import canonical_json_sha256
from gradlab.modal_eval_protocol import normalize_attempt_result
from gradlab.policy_bundle import evaluation_contract, evaluation_contract_sha256
from gradlab.run_contracts import CheckpointManifest, EvalIntent, EvalResult


EVALUATION_EQUIVALENCE_FIELDS = (
    "episodes",
    "n_envs",
    "seed",
    "seed_protocol",
    "manifest",
    "watchdog_steps",
    "environment",
    "asset",
    "action_sampling",
    "acceptance",
)


def validate_publication_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    recipe = evidence.get("recipe")
    if not isinstance(recipe, Mapping):
        raise ValueError("publication evidence is missing recipe.json")
    intent_value = evidence.get("intent")
    raw = evidence.get("raw_result")
    verified_value = evidence.get("verified_result")
    checkpoint_value = evidence.get("checkpoint_manifest")
    if not all(
        isinstance(value, Mapping)
        for value in (intent_value, raw, verified_value, checkpoint_value)
    ):
        raise ValueError("publication evidence intent/raw/verified/checkpoint join is incomplete")
    assert isinstance(intent_value, Mapping)
    assert isinstance(raw, Mapping)
    assert isinstance(verified_value, Mapping)
    assert isinstance(checkpoint_value, Mapping)
    intent = EvalIntent.from_dict(intent_value)
    intent.validate()
    verified = EvalResult.from_dict(verified_value)
    verified.validate()
    checkpoint = CheckpointManifest.from_dict(checkpoint_value)
    checkpoint.validate()
    if not (
        intent.run_id == verified.run_id == checkpoint.run_id
        and intent.checkpoint_id == verified.checkpoint_id == checkpoint.checkpoint_id
        and intent.idempotency_key == verified.idempotency_key
        and intent.checkpoint_sha256 == checkpoint.sha256
    ):
        raise ValueError("publication evaluation documents disagree on identity")
    if verified.status != "accepted":
        raise ValueError("only an accepted verified checkpoint evaluation may be published")

    expected = evaluation_contract(recipe)
    actual = dict(intent.execution_contract)
    for field in EVALUATION_EQUIVALENCE_FIELDS:
        if actual.get(field) != expected.get(field):
            raise ValueError(f"evaluation intent {field} differs from snapshot recipe")
    contract_sha256 = evaluation_contract_sha256(recipe)
    if intent.evaluation_contract_sha256 != contract_sha256:
        raise ValueError("evaluation contract digest differs from snapshot recipe")
    checkpoint_contract_sha256 = checkpoint_manifest_contract_sha256(recipe)
    if checkpoint.evaluation_contract_sha256 != checkpoint_contract_sha256:
        raise ValueError("checkpoint manifest contract digest differs from snapshot recipe")
    if actual.get("checkpoint_sha256") != checkpoint.sha256:
        raise ValueError("evaluation execution contract checkpoint hash is inconsistent")
    if actual.get("recipe_sha256") != checkpoint.recipe_document_sha256:
        raise ValueError("evaluation execution contract recipe hash is inconsistent")

    declared_episodes = int(expected.get("episodes") or 0)
    normalized = normalize_attempt_result(
        raw,
        contract=actual,
        attempt_id=str(raw.get("attempt_id") or ""),
    )
    if declared_episodes <= 0 or len(normalized["episode_results"]) != declared_episodes:
        raise ValueError("evaluation did not complete its declared episode count")
    if len(verified.episode_results) != declared_episodes:
        raise ValueError("verified evaluation episode count differs from the declared count")
    expected_verified = {
        "status": normalized["status"],
        "episode_results": normalized["episode_results"],
        "aggregates": normalized["aggregates"],
        "evidence_sha256": normalized["evidence_sha256"],
        "error": normalized["error"],
    }
    observed_verified = {
        "status": verified.status,
        "episode_results": list(verified.episode_results),
        "aggregates": dict(verified.aggregates),
        "evidence_sha256": list(verified.evidence_sha256),
        "error": verified.error,
    }
    if expected_verified != observed_verified:
        raise ValueError("verified evaluation does not equal normalized raw evidence")

    metrics = raw.get("metrics")
    publication = dict(metrics) if isinstance(metrics, Mapping) else {}
    publication.update(deepcopy(dict(verified.aggregates)))
    publication.update(
        {
            "action_sampling": str(expected.get("action_sampling") or ""),
            "protocol": "full",
            "checkpoint_step": checkpoint.step,
            "checkpoint_artifact": checkpoint.public_url,
            "episodes": declared_episodes,
            "by_start": eval_by_start_rows(
                [dict(item) for item in verified.episode_results]
            ),
            "evaluation_evidence": {
                "checkpoint_sha256": checkpoint.sha256,
                "recipe_sha256": checkpoint.recipe_document_sha256,
                "recipe_format_version": int(recipe.get("format_version") or 0),
                "evaluation_contract_sha256": contract_sha256,
                "exact_contract": True,
                "intent_sha256": canonical_json_sha256(dict(intent_value)),
                "raw_result_sha256": canonical_json_sha256(dict(raw)),
                "verified_result_sha256": canonical_json_sha256(dict(verified_value)),
            },
        }
    )
    return {
        "evaluation": publication,
        "recipe": deepcopy(dict(recipe)),
        "intent": deepcopy(dict(intent_value)),
        "raw_result": deepcopy(dict(raw)),
        "verified_result": deepcopy(dict(verified_value)),
        "checkpoint_manifest": deepcopy(dict(checkpoint_value)),
    }


__all__ = ["EVALUATION_EQUIVALENCE_FIELDS", "validate_publication_evidence"]
