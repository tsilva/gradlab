from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from gradlab.json_utils import canonical_json_sha256
from gradlab.model_inputs import runtime_model_input_contract
from gradlab.policy_model_config import POLICY_ROLES, normalize_policy_model


POLICY_EXECUTION_SCHEMA_VERSION = 1


def _class_path(value: Any) -> str:
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def compile_policy_execution_contract(model: Any, env: Any) -> dict[str, Any] | None:
    policy = getattr(model, "policy", None)
    raw_policy_model = getattr(policy, "policy_model", None)
    if not isinstance(raw_policy_model, Mapping):
        return None
    policy_model = normalize_policy_model(raw_policy_model)
    model_inputs = runtime_model_input_contract(env)
    role_inputs = {
        role: [
            "observation",
            *[
                f"context/{name}"
                for name, routes in policy_model["routes"].items()
                if role in routes
            ],
        ]
        for role in POLICY_ROLES
    }
    payload = {
        "schema_version": POLICY_EXECUTION_SCHEMA_VERSION,
        "model_inputs": model_inputs,
        "policy_model": policy_model,
        "policy_class": _class_path(policy),
        "role_inputs": role_inputs,
    }
    return {
        **payload,
        "sha256": canonical_json_sha256(
            payload,
            default=str,
            ensure_ascii=True,
        ),
    }


def normalize_policy_execution_contract(
    value: Any,
    *,
    label: str = "policy_execution_contract",
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    expected = {
        "schema_version",
        "model_inputs",
        "policy_model",
        "policy_class",
        "role_inputs",
        "sha256",
    }
    if set(value) != expected:
        raise ValueError(
            f"{label} must define exactly {sorted(expected)}, got {sorted(value)}"
        )
    if value["schema_version"] != POLICY_EXECUTION_SCHEMA_VERSION:
        raise ValueError(
            f"{label}.schema_version must be {POLICY_EXECUTION_SCHEMA_VERSION}"
        )
    payload = {key: deepcopy(value[key]) for key in expected if key != "sha256"}
    actual = canonical_json_sha256(payload, default=str, ensure_ascii=True)
    if value["sha256"] != actual:
        raise ValueError(f"{label}.sha256 does not match its canonical payload")
    return {**payload, "sha256": actual}


def verify_policy_execution_contract(
    model: Any,
    env: Any,
    saved_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    policy = getattr(model, "policy", None)
    routed = isinstance(getattr(policy, "policy_model", None), Mapping)
    saved = saved_contract
    if saved is None:
        candidate = getattr(model, "gradlab_policy_execution_contract", None)
        saved = candidate if isinstance(candidate, Mapping) else None
    if saved is None:
        if routed:
            raise ValueError("configured policy artifact is missing its execution contract")
        return None
    normalized_saved = normalize_policy_execution_contract(saved)
    runtime = compile_policy_execution_contract(model, env)
    if runtime is None or runtime != normalized_saved:
        raise ValueError(
            "policy execution contract does not match the provider-resolved runtime"
        )
    model.gradlab_policy_execution_contract = deepcopy(normalized_saved)
    return normalized_saved
