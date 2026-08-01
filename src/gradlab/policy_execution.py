from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from gradlab.json_utils import canonical_json_sha256
from gradlab.model_inputs import runtime_model_input_contract
from gradlab.policy_model_config import (
    LEGACY_POLICY_MODEL_SCHEMA_VERSION,
    POLICY_MODEL_SCHEMA_VERSION,
    POLICY_ROLES,
    normalize_artifact_policy_model,
)


POLICY_EXECUTION_SCHEMA_VERSION = 1


def _class_path(value: Any) -> str:
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def compile_policy_execution_contract(model: Any, env: Any) -> dict[str, Any] | None:
    policy = getattr(model, "policy", None)
    raw_policy_model = getattr(policy, "policy_model", None)
    if not isinstance(raw_policy_model, Mapping):
        return None
    policy_model = normalize_artifact_policy_model(raw_policy_model)
    model_inputs = runtime_model_input_contract(env)
    if policy_model["schema_version"] == LEGACY_POLICY_MODEL_SCHEMA_VERSION:
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
    else:
        context = model_inputs.get("context") if isinstance(model_inputs, Mapping) else None
        context_names = sorted(context) if isinstance(context, Mapping) else []
        shared_inputs = [
            "observation",
            *(f"context/{name}" for name in context_names),
        ]
        extractor = getattr(policy, "features_extractor", None)
        extractor_context = tuple(getattr(extractor, "context_names", ()))
        if extractor_context != tuple(context_names):
            raise ValueError(
                "shared policy context disagrees with the runtime model-input contract: "
                f"policy has {list(extractor_context)}, runtime has {context_names}"
            )
        role_inputs = {role: list(shared_inputs) for role in POLICY_ROLES}
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
        raise ValueError(f"{label} must define exactly {sorted(expected)}, got {sorted(value)}")
    if value["schema_version"] != POLICY_EXECUTION_SCHEMA_VERSION:
        raise ValueError(f"{label}.schema_version must be {POLICY_EXECUTION_SCHEMA_VERSION}")
    payload = {key: deepcopy(value[key]) for key in expected if key != "sha256"}
    actual = canonical_json_sha256(payload, default=str, ensure_ascii=True)
    if value["sha256"] != actual:
        raise ValueError(f"{label}.sha256 does not match its canonical payload")
    policy_model = normalize_artifact_policy_model(
        payload["policy_model"],
        label=f"{label}.policy_model",
    )
    if policy_model != payload["policy_model"]:
        raise ValueError(f"{label}.policy_model must be normalized")
    role_inputs = payload["role_inputs"]
    if not isinstance(role_inputs, Mapping) or set(role_inputs) != set(POLICY_ROLES):
        raise ValueError(f"{label}.role_inputs must define exactly {list(POLICY_ROLES)}")
    for role in POLICY_ROLES:
        inputs = role_inputs[role]
        if (
            not isinstance(inputs, list)
            or not inputs
            or any(not isinstance(item, str) or not item for item in inputs)
        ):
            raise ValueError(f"{label}.role_inputs.{role} must be a non-empty string list")
    if (
        policy_model["schema_version"] == POLICY_MODEL_SCHEMA_VERSION
        and role_inputs["action"] != role_inputs["state_value"]
    ):
        raise ValueError(f"{label}.role_inputs must be identical for policy-model v2")
    if policy_model["schema_version"] == POLICY_MODEL_SCHEMA_VERSION:
        model_inputs = payload["model_inputs"]
        context = model_inputs.get("context") if isinstance(model_inputs, Mapping) else None
        context_names = sorted(context) if isinstance(context, Mapping) else []
        expected_inputs = [
            "observation",
            *(f"context/{name}" for name in context_names),
        ]
        if role_inputs["action"] != expected_inputs:
            raise ValueError(
                f"{label}.role_inputs must match model_inputs: expected "
                f"{expected_inputs}, got {role_inputs['action']}"
            )
    return {**payload, "sha256": actual}


def verify_policy_execution_contract(
    model: Any,
    env: Any,
    saved_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    policy = getattr(model, "policy", None)
    configured = isinstance(getattr(policy, "policy_model", None), Mapping)
    saved = saved_contract
    if saved is None:
        candidate = getattr(model, "gradlab_policy_execution_contract", None)
        saved = candidate if isinstance(candidate, Mapping) else None
    if saved is None:
        if configured:
            raise ValueError("configured policy artifact is missing its execution contract")
        return None
    normalized_saved = normalize_policy_execution_contract(saved)
    runtime = compile_policy_execution_contract(model, env)
    if runtime is None or runtime != normalized_saved:
        raise ValueError("policy execution contract does not match the provider-resolved runtime")
    model.gradlab_policy_execution_contract = deepcopy(normalized_saved)
    return normalized_saved
