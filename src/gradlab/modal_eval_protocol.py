from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from gradlab.checkpoint_acceptance import (
    SEED_PROTOCOL,
    acceptance_aggregates,
    evaluate_acceptance,
    validate_episode_rows,
)
from gradlab.json_utils import canonical_json_sha256


PROTOCOL_SCHEMA_VERSION = 6
_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "contract_schema_version",
        "attempt_id",
        "execution_key",
        "checkpoint_sha256",
        "runtime_image_ref",
        "rom_sha256",
        "seed_protocol",
        "n_envs",
        "episodes",
        "recipe_sha256",
        "recipe_format_version",
        "evaluation_contract_sha256",
        "status",
        "duration_seconds",
        "episode_results",
        "evaluation_evidence",
        "verdict",
        "preview",
        "error",
    }
)


def _sha256(value: object, *, label: str) -> str:
    text = str(value or "").lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a SHA-256 hex digest")
    return text


def build_execution_contract(
    *,
    checkpoint_sha256: str,
    runtime_image_ref: str,
    eval_environment: Mapping[str, Any],
    episodes: int,
    n_envs: int,
    watchdog_steps: int,
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
        "watchdog_steps": int(watchdog_steps),
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
    if contract["episodes"] < 1 or contract["n_envs"] < 1 or contract["watchdog_steps"] < 1:
        raise ValueError("eval episodes, n_envs, and watchdog_steps must be positive")
    if not str(runtime_image_ref).startswith("docker:") or "@sha256:" not in str(runtime_image_ref):
        raise ValueError("eval runtime image must be an immutable docker reference")
    if asset_manifest is not None and not str(asset_manifest.get("sha256") or ""):
        raise ValueError("eval asset manifest must include sha256")
    return contract


def execution_key(contract: Mapping[str, Any]) -> str:
    return canonical_json_sha256(dict(contract), default=str, allow_nan=True)


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


def _validate_result_shape(result: Mapping[str, Any]) -> None:
    for retired_field in ("metrics", "claimed_aggregates"):
        if retired_field in result:
            raise ValueError(f"eval protocol v6 forbids result field: {retired_field}")
    unknown = sorted(str(name) for name in result if name not in _RESULT_FIELDS)
    if unknown:
        raise ValueError(f"eval protocol v6 result has unknown field: {unknown[0]}")


def _validate_result_identity(
    result: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    attempt_id: str,
) -> None:
    def require_matching_int(name: str, expected: int, *, message: str) -> None:
        value = result.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise ValueError(f"eval result {message} mismatch")

    schema_version = result.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != PROTOCOL_SCHEMA_VERSION
    ):
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
    require_matching_int(
        "recipe_format_version",
        int(contract["recipe_format_version"]),
        message="recipe format version",
    )
    require_matching_int(
        "contract_schema_version",
        int(contract["schema_version"]),
        message="contract schema version",
    )
    if str(result.get("runtime_image_ref") or "") != str(contract["runtime_image_ref"]):
        raise ValueError("eval result runtime identity mismatch")
    asset = contract.get("asset")
    expected_rom_sha = str(asset.get("sha256") or "") if isinstance(asset, Mapping) else ""
    if str(result.get("rom_sha256") or "") != expected_rom_sha:
        raise ValueError("eval result ROM hash mismatch")
    if str(result.get("seed_protocol") or "") != str(contract["seed_protocol"]):
        raise ValueError("eval result seed protocol mismatch")
    require_matching_int("n_envs", int(contract["n_envs"]), message="n_envs")
    require_matching_int(
        "episodes",
        int(contract["episodes"]),
        message="episode contract",
    )


def _duration_seconds(result: Mapping[str, Any], *, required: bool) -> float:
    value = result.get("duration_seconds")
    if value is None and not required:
        return 0.0
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError("eval result duration_seconds must be a finite non-negative number")
    return float(value)


def validate_attempt_result(
    result: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    attempt_id: str,
) -> dict[str, Any]:
    """Validate and scientifically recompute one successful Modal result."""

    _validate_result_shape(result)
    _validate_result_identity(result, contract=contract, attempt_id=attempt_id)
    if str(result.get("status") or "") != "succeeded":
        raise ValueError("eval attempt did not succeed")
    if "acceptance" not in contract:
        raise ValueError("Modal evaluation requires an acceptance contract")
    episodes = result.get("episode_results")
    if not isinstance(episodes, list):
        raise ValueError("acceptance result episode rows must be a list")
    validated_rows = validate_episode_rows(
        episodes,
        contract=contract,
    )
    computed = acceptance_aggregates(validated_rows, contract=contract)
    _validate_finite(validated_rows, label="eval episodes")
    complete = len(validated_rows) == int(contract["episodes"])
    if complete:
        for rule in contract["acceptance"]:
            if str(rule["metric"]) not in computed:
                raise ValueError("acceptance result is missing its decisive aggregate")
        accepted, _observed = evaluate_acceptance(computed, contract=contract)
        computed_verdict = "accepted" if accepted else "rejected"
    else:
        computed_verdict = "rejected"
    verdict = str(result.get("verdict") or "")
    if verdict not in {"accepted", "rejected"}:
        raise ValueError("acceptance verdict must be accepted or rejected")
    if verdict != computed_verdict:
        raise ValueError("acceptance verdict does not match supervisor recomputation")
    _duration_seconds(result, required=True)
    validated_result = dict(result)
    validated_result["episode_results"] = validated_rows
    validated_result["aggregates"] = computed
    validated_result["verdict"] = computed_verdict
    return validated_result


def normalize_attempt_result(
    result: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    attempt_id: str,
) -> dict[str, Any]:
    """Normalize one identity-verified Modal result for durable supervisor storage."""

    raw_status = str(result.get("status") or "")
    if raw_status not in {"succeeded", "failed", "expired"}:
        raise ValueError(f"unsupported eval result status: {raw_status!r}")
    _validate_result_shape(result)
    if raw_status == "succeeded":
        validated = validate_attempt_result(
            result,
            contract=contract,
            attempt_id=attempt_id,
        )
        verdict = str(validated.get("verdict") or "")
        status = "accepted" if verdict == "accepted" else "rejected"
        episodes = list(validated.get("episode_results") or [])
        aggregates = dict(validated.get("aggregates") or {})
        error = None
    else:
        _validate_result_identity(result, contract=contract, attempt_id=attempt_id)
        status = "expired" if raw_status == "expired" else "failed"
        episodes = []
        aggregates = {}
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
        "duration_seconds": _duration_seconds(result, required=raw_status == "succeeded"),
        "evidence_sha256": [
            canonical_json_sha256({"evidence": value})
            for value in evidence_values
            if value not in (None, {}, [])
        ],
        "error": error,
    }
