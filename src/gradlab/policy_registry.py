from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast


PolicyAlgorithmId: TypeAlias = Literal[
    "ppo",
    "a2c",
    "action-program",
    "cell-graph",
    "dqn",
    "recurrent-ppo",
]
RuntimePolicyAlgorithmId: TypeAlias = Literal[
    "ppo",
    "a2c",
    "dqn",
    "action-program",
    "cell-graph",
]
Sb3AlgorithmId: TypeAlias = Literal["ppo", "a2c", "dqn"]
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
    "dqn": PolicyAlgorithmSpec(
        ("stable_baselines3.dqn.dqn.DQN",),
        "sb3",
        "epsilon_greedy",
    ),
    "recurrent-ppo": PolicyAlgorithmSpec(
        ("sb3_contrib.ppo_recurrent.ppo_recurrent.RecurrentPPO",),
        None,
        "stochastic",
    ),
}


TRAINING_BACKEND_SPECS: dict[str, TrainingBackendSpec] = {
    "gradlab.go-explore": TrainingBackendSpec("gradlab.training.go_explore", "cell-graph"),
    "gradlab.jerk": TrainingBackendSpec("gradlab.training.jerk", "action-program"),
    "sb3.a2c": TrainingBackendSpec("gradlab.training.sb3", "a2c"),
    "sb3.ppo": TrainingBackendSpec("gradlab.training.sb3", "ppo"),
}
BACKEND_PROVENANCE_ALGORITHMS: dict[str, frozenset[PolicyAlgorithmId]] = {
    **{
        backend_id: frozenset({spec.algorithm_id})
        for backend_id, spec in TRAINING_BACKEND_SPECS.items()
    },
    # GradLab can validate, load, evaluate, and play archived SB3 DQN
    # checkpoints without claiming that DQN is a launchable training backend.
    "sb3.dqn": frozenset({"dqn"}),
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
    algorithms = BACKEND_PROVENANCE_ALGORITHMS.get(normalized)
    if algorithms is None:
        raise ValueError(f"unsupported checkpoint training backend: {backend_id}")
    spec = TRAINING_BACKEND_SPECS.get(normalized)
    if spec is not None:
        return spec.algorithm_id
    return next(iter(algorithms))


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
    resolved: set[PolicyAlgorithmId] = set()

    algorithm_id = str(metadata.get("algorithm_id") or "").strip()
    if algorithm_id:
        if algorithm_id not in POLICY_ALGORITHM_SPECS:
            raise ValueError(f"unsupported checkpoint algorithm: {algorithm_id}")
        resolved.add(cast(PolicyAlgorithmId, algorithm_id))

    model_class = str(metadata.get("model_class") or "").strip()
    if model_class:
        algorithm = MODEL_CLASS_ALGORITHMS.get(model_class)
        if algorithm is None:
            raise ValueError(f"unsupported checkpoint model class: {model_class}")
        resolved.add(algorithm)

    backend_id = str(metadata.get("training_backend_id") or "").strip()
    if backend_id:
        backend_algorithms = BACKEND_PROVENANCE_ALGORITHMS.get(backend_id)
        if backend_algorithms is None:
            raise ValueError(f"unsupported checkpoint training backend: {backend_id}")
        if resolved:
            incompatible = resolved - backend_algorithms
            if incompatible:
                raise ValueError(
                    "checkpoint backend, algorithm, and model class metadata disagree"
                )
        else:
            resolved.add(backend_provenance_algorithm(backend_id))

    if not resolved:
        raise ValueError("checkpoint metadata must identify its policy algorithm")
    if len(resolved) > 1:
        raise ValueError("checkpoint backend, algorithm, and model class metadata disagree")

    algorithm = next(iter(resolved))
    if algorithm not in allowed:
        raise ValueError(f"unsupported checkpoint algorithm: {algorithm}")
    return algorithm
