from __future__ import annotations

import re
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from gradlab.checkpoint_contract import checkpoint_manifest_contract_sha256
from gradlab.eval_metrics import eval_by_start_records
from gradlab.json_utils import canonical_json_sha256
from gradlab.metric_names import LEADER_CHECKPOINT_STEP, metric_definition, metric_display_label
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
EVALUATION_EVIDENCE_DOCUMENT_TYPE = "gradlab.evaluation_evidence"
EVALUATION_EVIDENCE_FORMAT_VERSION = 1


def _metric_record(metric: str, value: object) -> dict[str, Any]:
    definition = metric_definition(metric)
    if definition is None:
        raise ValueError(f"publication metric is not documented in METRICS.md: {metric}")
    return {
        "metric": metric,
        "label": metric_display_label(metric),
        "unit": definition.unit,
        "value": deepcopy(value),
    }


def _acceptance_outcomes(
    rules: object,
    aggregates: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(rules, list):
        raise ValueError("evaluation acceptance contract must be a list")
    outcomes: list[dict[str, Any]] = []
    for index, value in enumerate(rules):
        if not isinstance(value, Mapping):
            raise ValueError(f"acceptance rule {index} must be an object")
        rule = dict(value)
        metric = str(rule.get("metric") or "")
        observed = aggregates.get(metric)
        if observed is None:
            raise ValueError(f"accepted evidence is missing acceptance metric {metric!r}")
        operator = str(rule.get("operator") or "")
        threshold = rule.get("threshold")
        comparisons = {
            ">=": lambda: float(observed) >= float(threshold),
            ">": lambda: float(observed) > float(threshold),
            "<=": lambda: float(observed) <= float(threshold),
            "<": lambda: float(observed) < float(threshold),
            "==": lambda: observed == threshold,
        }
        if operator not in comparisons:
            raise ValueError(f"unsupported acceptance operator {operator!r}")
        outcomes.append(
            {
                **_metric_record(metric, observed),
                "operator": operator,
                "threshold": deepcopy(threshold),
                "passed": bool(comparisons[operator]()),
            }
        )
    return outcomes


def _ranking_outcomes(
    rules: object,
    aggregates: Mapping[str, Any],
    *,
    checkpoint_step: int,
) -> list[dict[str, Any]]:
    if not isinstance(rules, list):
        raise ValueError("goal ranking contract must be a list")
    outcomes: list[dict[str, Any]] = []
    for index, raw in enumerate(rules):
        match = re.fullmatch(r"(max|min)\(([^)]+)\)", str(raw))
        if match is None:
            raise ValueError(f"unsupported ranking rule {raw!r} at index {index}")
        direction, metric = match.groups()
        observed = checkpoint_step if metric == LEADER_CHECKPOINT_STEP else aggregates.get(metric)
        if observed is None:
            raise ValueError(f"accepted evidence is missing ranking metric {metric!r}")
        outcomes.append(
            {
                **_metric_record(metric, observed),
                "direction": direction,
                "rank_value": float(observed) if direction == "max" else -float(observed),
            }
        )
    return outcomes


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

    publication = deepcopy(dict(verified.aggregates))
    publication.update(
        {
            "action_sampling": str(expected.get("action_sampling") or ""),
            "protocol": "full",
            "checkpoint_step": checkpoint.step,
            "checkpoint_artifact": checkpoint.public_url,
            "episodes": declared_episodes,
            "by_start": eval_by_start_records(
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


def build_evaluation_evidence_document(evidence: Mapping[str, Any]) -> dict[str, Any]:
    validated = validate_publication_evidence(evidence)
    recipe = dict(validated["recipe"])
    recipe_value = recipe.get("recipe")
    if not isinstance(recipe_value, Mapping):
        raise ValueError("publication recipe is missing its materialized recipe")
    goal = recipe_value.get("goal")
    if not isinstance(goal, Mapping):
        raise ValueError("publication recipe is missing its materialized goal")
    verified = dict(validated["verified_result"])
    intent = dict(validated["intent"])
    checkpoint = CheckpointManifest.from_dict(validated["checkpoint_manifest"])
    aggregates = dict(verified.get("aggregates") or {})
    execution_contract = dict(intent.get("execution_contract") or {})
    acceptance_rules = execution_contract.get("acceptance")
    objective = goal.get("objective")
    ranking_rules = objective.get("rank") if isinstance(objective, Mapping) else None
    acceptance = _acceptance_outcomes(acceptance_rules, aggregates)
    ranking = _ranking_outcomes(
        ranking_rules,
        aggregates,
        checkpoint_step=checkpoint.step,
    )
    if not acceptance or not all(row["passed"] for row in acceptance):
        raise ValueError("accepted publication evidence does not pass every acceptance rule")
    return {
        "document_type": EVALUATION_EVIDENCE_DOCUMENT_TYPE,
        "format_version": EVALUATION_EVIDENCE_FORMAT_VERSION,
        "status": "accepted",
        "identity": {
            "run_id": checkpoint.run_id,
            "checkpoint_id": checkpoint.checkpoint_id,
            "checkpoint_step": checkpoint.step,
            "checkpoint_sha256": checkpoint.sha256,
            "recipe_sha256": checkpoint.recipe_document_sha256,
        },
        "protocol": {
            "action_sampling": execution_contract.get("action_sampling"),
            "episodes": execution_contract.get("episodes"),
            "seed": execution_contract.get("seed"),
            "seed_protocol": deepcopy(execution_contract.get("seed_protocol")),
            "manifest": deepcopy(execution_contract.get("manifest")),
        },
        "episode_results": deepcopy(list(verified.get("episode_results") or [])),
        "aggregates": aggregates,
        "acceptance": {
            "rules": deepcopy(acceptance_rules),
            "outcomes": acceptance,
            "passed": True,
        },
        "ranking": {"rules": deepcopy(ranking_rules), "outcomes": ranking},
        "contracts": {
            "materialized_goal": deepcopy(dict(goal)),
            "evaluation": deepcopy(execution_contract),
            "environment": deepcopy(execution_contract.get("environment")),
        },
        "authoritative_hashes": {
            "intent_sha256": canonical_json_sha256(validated["intent"]),
            "raw_result_sha256": canonical_json_sha256(validated["raw_result"]),
            "verified_result_sha256": canonical_json_sha256(validated["verified_result"]),
            "checkpoint_manifest_sha256": canonical_json_sha256(
                validated["checkpoint_manifest"]
            ),
            "recipe_sha256": checkpoint.recipe_document_sha256,
            "checkpoint_sha256": checkpoint.sha256,
            "evaluation_contract_sha256": evaluation_contract_sha256(recipe),
        },
    }


def validate_evaluation_evidence_document(document: Mapping[str, Any]) -> dict[str, Any]:
    expected_fields = {
        "document_type",
        "format_version",
        "status",
        "identity",
        "protocol",
        "episode_results",
        "aggregates",
        "acceptance",
        "ranking",
        "contracts",
        "authoritative_hashes",
    }
    if set(document) != expected_fields:
        raise ValueError("evaluation_evidence.json has an unsupported field set")
    if document.get("document_type") != EVALUATION_EVIDENCE_DOCUMENT_TYPE:
        raise ValueError("evaluation_evidence.json has an invalid document_type")
    if document.get("format_version") != EVALUATION_EVIDENCE_FORMAT_VERSION:
        raise ValueError("evaluation_evidence.json has an unsupported format_version")
    if document.get("status") != "accepted":
        raise ValueError("research release evidence must be accepted")
    acceptance = document.get("acceptance")
    if not isinstance(acceptance, Mapping) or acceptance.get("passed") is not True:
        raise ValueError("research release evidence did not pass acceptance")
    episodes = document.get("episode_results")
    protocol = document.get("protocol")
    if not isinstance(episodes, list) or not isinstance(protocol, Mapping):
        raise ValueError("evaluation_evidence.json is missing episode results or protocol")
    if len(episodes) != int(protocol.get("episodes") or 0):
        raise ValueError("evaluation_evidence.json episode count is incomplete")
    return deepcopy(dict(document))


__all__ = [
    "EVALUATION_EQUIVALENCE_FIELDS",
    "EVALUATION_EVIDENCE_DOCUMENT_TYPE",
    "EVALUATION_EVIDENCE_FORMAT_VERSION",
    "build_evaluation_evidence_document",
    "validate_evaluation_evidence_document",
    "validate_publication_evidence",
]
