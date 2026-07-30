from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast


PolicyAlgorithmId: TypeAlias = Literal[
    "ppo",
    "a2c",
    "action-program",
    "cell-graph",
]
RuntimePolicyAlgorithmId: TypeAlias = Literal[
    "ppo",
    "a2c",
    "action-program",
    "cell-graph",
]
Sb3AlgorithmId: TypeAlias = Literal["ppo", "a2c"]
PolicyRuntimeFamily: TypeAlias = Literal["sb3", "action-program", "cell-graph"]


@dataclass(frozen=True)
class PolicyAlgorithmSpec:
    model_classes: tuple[str, ...]
    runtime_family: PolicyRuntimeFamily | None
    default_action_selection_mode: str


@dataclass(frozen=True)
class TrainingBackendSpec:
    module_name: str
    algorithm_id: RuntimePolicyAlgorithmId


POLICY_ALGORITHM_SPECS: dict[PolicyAlgorithmId, PolicyAlgorithmSpec] = {
    "ppo": PolicyAlgorithmSpec(
        (
            "gradlab.task_advantage.PerTaskAdvantagePPO",
            "stable_baselines3.ppo.ppo.PPO",
        ),
        "sb3",
        "stochastic",
    ),
    "a2c": PolicyAlgorithmSpec(
        ("stable_baselines3.a2c.a2c.A2C",),
        "sb3",
        "stochastic",
    ),
    "action-program": PolicyAlgorithmSpec(
        ("gradlab.action_program.ActionProgramPolicy",),
        "action-program",
        "program",
    ),
    "cell-graph": PolicyAlgorithmSpec(
        ("gradlab.cell_graph.CellGraphPolicy",),
        "cell-graph",
        "route",
    ),
}


TRAINING_BACKEND_SPECS: dict[str, TrainingBackendSpec] = {
    "gradlab.go-explore": TrainingBackendSpec("gradlab.training.go_explore", "cell-graph"),
    "gradlab.jerk": TrainingBackendSpec("gradlab.training.jerk", "action-program"),
    "sb3.a2c": TrainingBackendSpec("gradlab.training.sb3", "a2c"),
    "sb3.ppo": TrainingBackendSpec("gradlab.training.sb3", "ppo"),
}
MODEL_CLASS_ALGORITHMS: dict[str, PolicyAlgorithmId] = {
    model_class: algorithm_id
    for algorithm_id, spec in POLICY_ALGORITHM_SPECS.items()
    for model_class in spec.model_classes
}
ALGORITHM_MODEL_CLASSES: dict[str, frozenset[str]] = {
    algorithm_id: frozenset(spec.model_classes)
    for algorithm_id, spec in POLICY_ALGORITHM_SPECS.items()
}

RUNTIME_POLICY_ALGORITHMS = frozenset[PolicyAlgorithmId](
    algorithm_id
    for algorithm_id, spec in POLICY_ALGORITHM_SPECS.items()
    if spec.runtime_family is not None
)
SB3_ALGORITHMS = frozenset[PolicyAlgorithmId](
    algorithm_id
    for algorithm_id, spec in POLICY_ALGORITHM_SPECS.items()
    if spec.runtime_family == "sb3"
)


def backend_provenance_algorithm(backend_id: str) -> PolicyAlgorithmId:
    normalized = str(backend_id).strip()
    spec = TRAINING_BACKEND_SPECS.get(normalized)
    if spec is None:
        raise ValueError(f"unsupported checkpoint training backend: {backend_id}")
    return spec.algorithm_id


def default_action_selection_mode(algorithm_id: PolicyAlgorithmId) -> str:
    spec = POLICY_ALGORITHM_SPECS.get(algorithm_id)
    if spec is None:
        raise ValueError(f"unsupported checkpoint algorithm: {algorithm_id}")
    return spec.default_action_selection_mode


def resolve_policy_algorithm(
    metadata: Mapping[str, Any] | None,
    *,
    allowed: frozenset[PolicyAlgorithmId] = RUNTIME_POLICY_ALGORITHMS,
) -> PolicyAlgorithmId:
    metadata = metadata or {}

    algorithm_id = str(metadata.get("algorithm_id") or "").strip()
    if not algorithm_id:
        raise ValueError("checkpoint metadata must identify its policy algorithm")
    if algorithm_id not in POLICY_ALGORITHM_SPECS:
        raise ValueError(f"unsupported checkpoint algorithm: {algorithm_id}")
    algorithm = cast(PolicyAlgorithmId, algorithm_id)

    model_class = str(metadata.get("model_class") or "").strip()
    if model_class:
        model_algorithm = MODEL_CLASS_ALGORITHMS.get(model_class)
        if model_algorithm is None:
            raise ValueError(f"unsupported checkpoint model class: {model_class}")
        if model_algorithm != algorithm:
            raise ValueError(
                "checkpoint backend, algorithm, and model class metadata disagree"
            )

    backend_id = str(metadata.get("training_backend_id") or "").strip()
    if backend_id:
        backend = TRAINING_BACKEND_SPECS.get(backend_id)
        if backend is None:
            raise ValueError(f"unsupported checkpoint training backend: {backend_id}")
        if backend.algorithm_id != algorithm:
            raise ValueError(
                "checkpoint backend, algorithm, and model class metadata disagree"
            )

    if algorithm not in allowed:
        raise ValueError(f"unsupported checkpoint algorithm: {algorithm}")
    return algorithm
