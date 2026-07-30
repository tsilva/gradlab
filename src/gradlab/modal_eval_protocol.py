from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from gradlab.checkpoint_acceptance import (
    SEED_PROTOCOL,
    acceptance_aggregates,
    aggregates_match,
    evaluate_acceptance,
    validate_episode_rows,
)
from gradlab.json_utils import canonical_json_sha256, canonical_json_text


PROTOCOL_SCHEMA_VERSION = 4


def _sha256(value: object, *, label: str) -> str:
    text = str(value or "").lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a SHA-256 hex digest")
    return text


def canonical_json(value: object) -> str:
    return canonical_json_text(value, default=str, allow_nan=True)


def stable_hash(value: object) -> str:
    return canonical_json_sha256(value, default=str, allow_nan=True)


def build_execution_contract(
    *,
    checkpoint_sha256: str,
    runtime_image_ref: str,
    eval_environment: Mapping[str, Any],
    episodes: int,
    n_envs: int,
    max_steps: int,
    seed: int,
    seed_protocol: str,
    asset_manifest: Mapping[str, Any] | None,
    recipe_sha256: str,
    recipe_format_version: int,
    evaluation_contract_sha256: str,
    action_sampling: str = "stochastic",
) -> dict[str, Any]:
    """Build the hash-bound execution envelope used by one Modal attempt."""

    if seed_protocol != SEED_PROTOCOL:
        raise ValueError(f"unsupported eval seed protocol: {seed_protocol}")
    contract: dict[str, Any] = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "checkpoint_sha256": _sha256(
            checkpoint_sha256,
            label="checkpoint hash",
        ),
        "runtime_image_ref": str(runtime_image_ref),
        "environment": dict(eval_environment),
        "episodes": int(episodes),
        "n_envs": int(n_envs),
        "max_steps": int(max_steps),
        "deterministic": False,
        "action_sampling": str(action_sampling),
        "seed": int(seed),
        "seed_protocol": seed_protocol,
        "recipe_sha256": _sha256(recipe_sha256, label="recipe hash"),
        "recipe_format_version": int(recipe_format_version),
        "evaluation_contract_sha256": _sha256(
            evaluation_contract_sha256,
            label="evaluation contract hash",
        ),
        "asset": (
            {str(key): value for key, value in asset_manifest.items() if str(key) != "object_uri"}
            if asset_manifest is not None
            else None
        ),
    }
    if contract["episodes"] < 1 or contract["n_envs"] < 1 or contract["max_steps"] < 1:
        raise ValueError("eval episodes, n_envs, and max_steps must be positive")
    if not str(runtime_image_ref).startswith("docker:") or "@sha256:" not in str(runtime_image_ref):
        raise ValueError("eval runtime image must be an immutable docker reference")
    if asset_manifest is not None and not str(asset_manifest.get("sha256") or ""):
        raise ValueError("eval asset manifest must include sha256")
    return contract


def execution_key(contract: Mapping[str, Any]) -> str:
    return stable_hash(dict(contract))


def _validate_finite(value: object, *, label: str) -> None:
    if isinstance(value, bool | str) or value is None:
        return
    if isinstance(value, int | float):
        if not math.isfinite(float(value)):
            raise ValueError(f"{label} is not finite")
        return
    if isinstance(value, Mapping):
        for name, nested in value.items():
            _validate_finite(nested, label=f"{label}.{name}")
        return
    if isinstance(value, list):
        for index, nested in enumerate(value):
            _validate_finite(nested, label=f"{label}[{index}]")


def validate_attempt_result(
    result: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    attempt_id: str,
) -> dict[str, Any]:
    """Validate a complete or fail-fast acceptance result from Modal."""

    if int(result.get("schema_version") or 0) != PROTOCOL_SCHEMA_VERSION:
        raise ValueError("eval result schema version mismatch")
    if str(result.get("attempt_id") or "") != attempt_id:
        raise ValueError("eval result attempt id mismatch")
    if str(result.get("execution_key") or "") != execution_key(contract):
        raise ValueError("eval result execution key mismatch")
    if str(result.get("checkpoint_sha256") or "") != str(contract["checkpoint_sha256"]):
        raise ValueError("eval result checkpoint hash mismatch")
    for name, message in (
        ("recipe_sha256", "recipe hash"),
        ("evaluation_contract_sha256", "evaluation contract hash"),
    ):
        if str(result.get(name) or "") != str(contract[name]):
            raise ValueError(f"eval result {message} mismatch")
    if int(result.get("recipe_format_version") or 0) != int(contract["recipe_format_version"]):
        raise ValueError("eval result recipe format version mismatch")
    if int(result.get("contract_schema_version") or 0) != int(contract["schema_version"]):
        raise ValueError("eval result contract schema version mismatch")
    if str(result.get("runtime_image_ref") or "") != str(contract["runtime_image_ref"]):
        raise ValueError("eval result runtime identity mismatch")
    asset = contract.get("asset")
    expected_rom_sha = str(asset.get("sha256") or "") if isinstance(asset, Mapping) else ""
    if str(result.get("rom_sha256") or "") != expected_rom_sha:
        raise ValueError("eval result ROM hash mismatch")
    if str(result.get("seed_protocol") or "") != str(contract["seed_protocol"]):
        raise ValueError("eval result seed protocol mismatch")
    if int(result.get("n_envs") or 0) != int(contract["n_envs"]):
        raise ValueError("eval result n_envs mismatch")
    if int(result.get("episodes") or 0) != int(contract["episodes"]):
        raise ValueError("eval result episode contract mismatch")
    if str(result.get("status") or "") != "succeeded":
        raise ValueError("eval attempt did not succeed")
    if "acceptance" not in contract:
        raise ValueError("Modal evaluation requires an acceptance contract")

    episodes = result.get("episode_results")
    if not isinstance(episodes, list):
        raise ValueError("acceptance result episode rows must be a list")
    verdict = str(result.get("verdict") or "")
    validated_rows = validate_episode_rows(
        episodes,
        contract=contract,
        verdict=verdict,
    )
    computed = acceptance_aggregates(validated_rows, contract=contract)
    claimed = result.get("claimed_aggregates")
    if not isinstance(claimed, Mapping) or not aggregates_match(claimed, computed):
        raise ValueError("acceptance result claimed aggregates do not match episode evidence")
    metrics = result.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("acceptance result metrics must be a mapping")
    _validate_finite(metrics, label="eval metrics")
    _validate_finite(validated_rows, label="eval episodes")
    fail_fast = str(contract.get("evidence_policy", {}).get("fail_fast") or "")
    if fail_fast == "first_failed_episode" and verdict == "rejected":
        if any(str(name).startswith("eval/full/") for name in metrics):
            raise ValueError("partial rejection must not emit completed eval/full metrics")
        if int(computed["failure_count"]) < 1:
            raise ValueError("acceptance rejection has no failed episode")
    else:
        accepted, _observed = evaluate_acceptance(computed, contract=contract)
        if accepted != (verdict == "accepted"):
            raise ValueError("acceptance verdict does not match its acceptance rules")
        for rule in contract["acceptance"]:
            name = str(rule["metric"])
            if name not in computed:
                raise ValueError("acceptance result is missing its decisive aggregate")
            if name not in metrics or not math.isclose(
                float(metrics[name]),
                float(computed[name]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("acceptance result decisive metric mismatch")
    validated_result = dict(result)
    validated_result["episode_results"] = validated_rows
    validated_result["claimed_aggregates"] = computed
    return validated_result


def normalize_attempt_result(
    result: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    attempt_id: str,
) -> dict[str, Any]:
    """Normalize one identity-verified Modal result for durable supervisor storage."""

    raw_status = str(result.get("status") or "")
    if raw_status == "succeeded":
        validated = validate_attempt_result(
            result,
            contract=contract,
            attempt_id=attempt_id,
        )
        verdict = str(validated.get("verdict") or "")
        status = "accepted" if verdict == "accepted" else "rejected"
        episodes = list(validated.get("episode_results") or [])
        aggregates = dict(validated.get("claimed_aggregates") or {})
        error = None
    else:
        if str(result.get("attempt_id") or "") != attempt_id:
            raise ValueError("failed eval result attempt id mismatch")
        if str(result.get("execution_key") or "") != execution_key(contract):
            raise ValueError("failed eval result execution key mismatch")
        status = "expired" if raw_status == "expired" else "failed"
        episodes = list(result.get("episode_results") or [])
        aggregates = dict(result.get("claimed_aggregates") or {})
        error = str(result.get("error") or f"Modal eval status={raw_status or 'unknown'}")

    evidence_values = [
        episodes,
        result.get("evaluation_evidence") or {},
        result.get("preview") or {},
    ]
    return {
        "status": status,
        "episode_results": episodes,
        "aggregates": aggregates,
        "duration_seconds": float(result.get("duration_seconds") or 0.0),
        "evidence_sha256": [
            canonical_json_sha256({"evidence": value})
            for value in evidence_values
            if value not in (None, {}, [])
        ],
        "error": error,
    }
