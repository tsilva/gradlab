from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from gradlab.metric_names import metric_path_segment
from gradlab.model_inputs import model_input_fields


POLICY_MODEL_SCHEMA_VERSION = 2
LEGACY_POLICY_MODEL_SCHEMA_VERSION = 1
POLICY_ROLES = ("action", "state_value")
POLICY_TOPOLOGIES = frozenset({"shared_encoder", "separate_encoders"})
OBSERVATION_ENCODERS = frozenset({"nature_cnn", "flatten"})
CONTEXT_ENCODERS = frozenset({"identity", "one_hot"})
MLP_ACTIVATIONS = frozenset({"tanh", "relu"})


def _normalize_encoder(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    unexpected = sorted(set(value) - {"kind", "features_dim"})
    if unexpected:
        raise ValueError(f"{label} has unexpected fields: {unexpected}")
    kind = str(value.get("kind") or "")
    if kind not in OBSERVATION_ENCODERS:
        raise ValueError(f"{label}.kind must be one of {sorted(OBSERVATION_ENCODERS)}")
    if kind == "nature_cnn":
        features_dim = value.get("features_dim", 512)
        if not isinstance(features_dim, int) or isinstance(features_dim, bool) or features_dim <= 0:
            raise ValueError(f"{label}.features_dim must be a positive integer")
        return {"kind": kind, "features_dim": int(features_dim)}
    if "features_dim" in value:
        raise ValueError(f"{label}.features_dim is unsupported for flatten")
    return {"kind": kind}


def _normalize_mlp(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    unexpected = sorted(set(value) - {"hidden_sizes", "activation"})
    if unexpected:
        raise ValueError(f"{label} has unexpected fields: {unexpected}")
    raw_sizes = value.get("hidden_sizes", ())
    if not isinstance(raw_sizes, Sequence) or isinstance(raw_sizes, str | bytes):
        raise ValueError(f"{label}.hidden_sizes must be a list")
    sizes = []
    for index, item in enumerate(raw_sizes):
        if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
            raise ValueError(f"{label}.hidden_sizes[{index}] must be a positive integer")
        sizes.append(int(item))
    activation = str(value.get("activation", "tanh"))
    if activation not in MLP_ACTIVATIONS:
        raise ValueError(f"{label}.activation must be one of {sorted(MLP_ACTIVATIONS)}")
    return {"hidden_sizes": sizes, "activation": activation}


def normalize_policy_model(value: Any, *, label: str = "policy_model") -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    if value.get("schema_version") != POLICY_MODEL_SCHEMA_VERSION:
        raise ValueError(f"{label}.schema_version must be {POLICY_MODEL_SCHEMA_VERSION}")
    allowed = {
        "schema_version",
        "encoder",
        "fusion",
        "normalize_images",
        "orthogonal_init",
    }
    unexpected = sorted(set(value) - allowed)
    if unexpected:
        raise ValueError(f"{label} has unexpected fields: {unexpected}")
    normalize_images = value.get("normalize_images", True)
    orthogonal_init = value.get("orthogonal_init", True)
    if not isinstance(normalize_images, bool):
        raise ValueError(f"{label}.normalize_images must be a boolean")
    if not isinstance(orthogonal_init, bool):
        raise ValueError(f"{label}.orthogonal_init must be a boolean")
    return {
        "schema_version": POLICY_MODEL_SCHEMA_VERSION,
        "encoder": _normalize_encoder(
            value.get("encoder"),
            label=f"{label}.encoder",
        ),
        "fusion": _normalize_mlp(
            value.get("fusion"),
            label=f"{label}.fusion",
        ),
        "normalize_images": normalize_images,
        "orthogonal_init": orthogonal_init,
    }


def normalize_legacy_policy_model(
    value: Any,
    *,
    label: str = "policy_model",
) -> dict[str, Any]:
    """Validate the v1 policy embedded in active, read-only model artifacts."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    allowed = {
        "schema_version",
        "topology",
        "fusion",
        "context_encoders",
        "routes",
        "heads",
        "normalize_images",
        "orthogonal_init",
    }
    unexpected = sorted(set(value) - allowed)
    if unexpected:
        raise ValueError(f"{label} has unexpected fields: {unexpected}")
    if value.get("schema_version") != LEGACY_POLICY_MODEL_SCHEMA_VERSION:
        raise ValueError(f"{label}.schema_version must be {LEGACY_POLICY_MODEL_SCHEMA_VERSION}")
    topology = value.get("topology")
    if not isinstance(topology, Mapping):
        raise ValueError(f"{label}.topology must be an object")
    topology_kind = str(topology.get("kind") or "")
    if topology_kind not in POLICY_TOPOLOGIES:
        raise ValueError(f"{label}.topology.kind must be one of {sorted(POLICY_TOPOLOGIES)}")
    if topology_kind == "shared_encoder":
        unexpected_topology = sorted(set(topology) - {"kind", "encoder"})
        if unexpected_topology:
            raise ValueError(f"{label}.topology has unexpected fields: {unexpected_topology}")
        normalized_topology = {
            "kind": topology_kind,
            "encoder": _normalize_encoder(
                topology.get("encoder"),
                label=f"{label}.topology.encoder",
            ),
        }
    else:
        unexpected_topology = sorted(set(topology) - {"kind", "encoders"})
        if unexpected_topology:
            raise ValueError(f"{label}.topology has unexpected fields: {unexpected_topology}")
        encoders = topology.get("encoders")
        if not isinstance(encoders, Mapping) or set(encoders) != set(POLICY_ROLES):
            raise ValueError(f"{label}.topology.encoders must define exactly {list(POLICY_ROLES)}")
        normalized_topology = {
            "kind": topology_kind,
            "encoders": {
                role: _normalize_encoder(
                    encoders[role],
                    label=f"{label}.topology.encoders.{role}",
                )
                for role in POLICY_ROLES
            },
        }
    fusion = str(value.get("fusion") or "")
    if fusion != "post_encoder_concat":
        raise ValueError(f"{label}.fusion must be 'post_encoder_concat'")

    raw_context_encoders = value.get("context_encoders", {})
    if not isinstance(raw_context_encoders, Mapping):
        raise ValueError(f"{label}.context_encoders must be an object")
    context_encoders: dict[str, Any] = {}
    for raw_name, raw_encoder in raw_context_encoders.items():
        name = str(raw_name)
        metric_path_segment(name)
        if not isinstance(raw_encoder, Mapping):
            raise ValueError(f"{label}.context_encoders.{name} must be an object")
        unexpected_encoder = sorted(set(raw_encoder) - {"kind"})
        if unexpected_encoder:
            raise ValueError(
                f"{label}.context_encoders.{name} has unexpected fields: {unexpected_encoder}"
            )
        kind = str(raw_encoder.get("kind") or "")
        if kind not in CONTEXT_ENCODERS:
            raise ValueError(
                f"{label}.context_encoders.{name}.kind must be one of {sorted(CONTEXT_ENCODERS)}"
            )
        context_encoders[name] = {"kind": kind}

    raw_routes = value.get("routes", {})
    if not isinstance(raw_routes, Mapping):
        raise ValueError(f"{label}.routes must be an object")
    routes: dict[str, list[str]] = {}
    for raw_name, raw_roles in raw_routes.items():
        name = str(raw_name)
        metric_path_segment(name)
        if not isinstance(raw_roles, Sequence) or isinstance(raw_roles, str | bytes):
            raise ValueError(f"{label}.routes.{name} must be a non-empty list")
        roles = [str(role) for role in raw_roles]
        if not roles:
            raise ValueError(f"{label}.routes.{name} must be a non-empty list")
        if len(set(roles)) != len(roles) or any(role not in POLICY_ROLES for role in roles):
            raise ValueError(
                f"{label}.routes.{name} must contain unique roles from {list(POLICY_ROLES)}"
            )
        routes[name] = [role for role in POLICY_ROLES if role in roles]

    heads = value.get("heads")
    if not isinstance(heads, Mapping) or set(heads) != set(POLICY_ROLES):
        raise ValueError(f"{label}.heads must define exactly {list(POLICY_ROLES)}")
    normalize_images = value.get("normalize_images", True)
    orthogonal_init = value.get("orthogonal_init", True)
    if not isinstance(normalize_images, bool):
        raise ValueError(f"{label}.normalize_images must be a boolean")
    if not isinstance(orthogonal_init, bool):
        raise ValueError(f"{label}.orthogonal_init must be a boolean")
    return {
        "schema_version": LEGACY_POLICY_MODEL_SCHEMA_VERSION,
        "topology": normalized_topology,
        "fusion": fusion,
        "context_encoders": {name: context_encoders[name] for name in sorted(context_encoders)},
        "routes": {name: routes[name] for name in sorted(routes)},
        "heads": {
            role: _normalize_mlp(
                heads[role],
                label=f"{label}.heads.{role}",
            )
            for role in POLICY_ROLES
        },
        "normalize_images": normalize_images,
        "orthogonal_init": orthogonal_init,
    }


def normalize_artifact_policy_model(
    value: Any,
    *,
    label: str = "policy_model",
) -> dict[str, Any]:
    """Validate a current v2 model or an explicitly retained active v1 artifact."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    schema_version = value.get("schema_version")
    if schema_version == POLICY_MODEL_SCHEMA_VERSION:
        return normalize_policy_model(value, label=label)
    if schema_version == LEGACY_POLICY_MODEL_SCHEMA_VERSION:
        return normalize_legacy_policy_model(value, label=label)
    raise ValueError(
        f"{label}.schema_version must be {POLICY_MODEL_SCHEMA_VERSION} for new models "
        f"or {LEGACY_POLICY_MODEL_SCHEMA_VERSION} for active read-only artifacts"
    )


def validate_policy_model_context(
    policy_model: Mapping[str, Any] | None,
    task: Mapping[str, Any],
    *,
    label: str = "policy_model",
) -> None:
    fields = model_input_fields(task)
    if fields and policy_model is None:
        raise ValueError(f"{label} is required when task.model_inputs declares context")
