"""Typed execution choices compiled at a recipe or learner-file boundary.

The document remains the canonical wire representation. Runtime code consumes
resolved fields and requests a copy only when adding Attempt-specific metadata.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from gradlab.env_config import env_config_from_mapping
from gradlab.environment_fields import EnvConfig
from gradlab.provider_config import provider_num_envs
from gradlab.training_backend import TrainingBackend, load_training_backend


@dataclass(frozen=True)
class ResolvedTrainConfig(Mapping[str, Any]):
    _document: dict[str, Any]
    environment: EnvConfig
    backend: TrainingBackend
    backend_config: Mapping[str, Any]
    n_envs: int

    @classmethod
    def from_validated(cls, document: Mapping[str, Any]) -> ResolvedTrainConfig:
        """Project a boundary-validated document without changing its wire bytes."""
        config = deepcopy(dict(document))
        envelope = config["training_backend"]
        return cls(
            config,
            env_config_from_mapping(config),
            load_training_backend(envelope["id"]),
            deepcopy(envelope.get("config") or {}),
            provider_num_envs(config, explicit_n_envs=config.get("n_envs")),
        )

    def to_document(self) -> dict[str, Any]:
        return deepcopy(self._document)

    def __getitem__(self, key: str) -> Any:
        return deepcopy(self._document[key])

    def __iter__(self) -> Iterator[str]:
        return iter(self._document)

    def __len__(self) -> int:
        return len(self._document)
