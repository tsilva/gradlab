from __future__ import annotations

import importlib
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from gradlab.policy_registry import TRAINING_BACKEND_SPECS
from gradlab.json_utils import canonical_json_sha256
from gradlab.training_lifecycle import TrainingResult


class GracefulStopFlag:
    def __init__(self) -> None:
        self._requested = threading.Event()
        self._lock = threading.RLock()
        self._reason = ""

    @property
    def requested(self) -> bool:
        return self._requested.is_set()

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason

    def request(self, reason: str) -> None:
        with self._lock:
            self._reason = str(reason)
            self._requested.set()


@dataclass
class BackendContext:
    train_config: dict[str, Any]
    environment: Any
    run_dir: Path
    checkpoint_dir: Path
    metric_store: Any
    wandb_enabled: bool
    stop_flag: Any
    rom_binding: Any | None
    session: Any

    @property
    def backend_config(self) -> dict[str, Any]:
        return training_backend_config(self.train_config)

    def mark_ready(self) -> Path:
        return self.session.mark_ready()


class TrainingBackend(Protocol):
    backend_id: str

    def normalize_config(
        self,
        config: Mapping[str, Any],
        *,
        label: str,
    ) -> dict[str, Any]: ...

    def validate(
        self,
        common_config: Mapping[str, Any],
        backend_config: Mapping[str, Any],
    ) -> None: ...

    def run(self, context: BackendContext) -> TrainingResult: ...

    def acceptance_mode(self, backend_config: Mapping[str, Any]) -> str: ...

    def contract_payload(self) -> dict[str, Any]: ...

    def state_archive_priority_metrics(self) -> tuple[str, ...]: ...

    def runtime_metadata(self, backend_config: Mapping[str, Any]) -> Mapping[str, str]: ...


CHECKPOINT_EVAL_ACCEPTANCE = "checkpoint_eval"
FIRST_TRAINING_SUCCESS_ACCEPTANCE = "first_training_success"


def registered_training_backend_ids() -> tuple[str, ...]:
    return tuple(sorted(TRAINING_BACKEND_SPECS))


def load_training_backend(backend_id: str) -> TrainingBackend:
    spec = TRAINING_BACKEND_SPECS.get(backend_id)
    if spec is None:
        known = ", ".join(registered_training_backend_ids())
        raise ValueError(f"unknown training backend {backend_id!r}; known: {known}")
    module = importlib.import_module(spec.module_name)
    backends = getattr(module, "BACKENDS", None)
    if not isinstance(backends, Mapping) or backend_id not in backends:
        raise RuntimeError(f"{spec.module_name} does not register backend {backend_id!r}")
    backend = cast(TrainingBackend, backends[backend_id])
    if backend.backend_id != backend_id:
        raise RuntimeError(
            f"{spec.module_name} registered backend {backend.backend_id!r} as {backend_id!r}"
        )
    return backend


def training_backend_id(config: Mapping[str, Any]) -> str:
    value = config.get("training_backend")
    if not isinstance(value, Mapping):
        raise ValueError("train_config.training_backend must be an object")
    backend_id = str(value.get("id") or "").strip()
    if not backend_id:
        raise ValueError("train_config.training_backend.id must be a non-empty string")
    return backend_id


def training_backend_config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = config.get("training_backend")
    if not isinstance(value, Mapping):
        raise ValueError("train_config.training_backend must be an object")
    backend_config = value.get("config")
    if not isinstance(backend_config, Mapping):
        raise ValueError("train_config.training_backend.config must be an object")
    return dict(backend_config)


def training_backend_acceptance_mode(config: Mapping[str, Any]) -> str:
    """Return the backend-declared acceptance authority for a train config."""

    backend_id = training_backend_id(config)
    backend_config = training_backend_config(config)
    mode = str(load_training_backend(backend_id).acceptance_mode(backend_config)).strip()
    return mode or CHECKPOINT_EVAL_ACCEPTANCE


def accepts_first_training_success(config: Mapping[str, Any]) -> bool:
    return training_backend_acceptance_mode(config) == FIRST_TRAINING_SUCCESS_ACCEPTANCE


def normalize_training_backend(
    value: Any,
    *,
    common_config: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    unexpected = sorted(set(value) - {"id", "config"})
    if unexpected:
        raise ValueError(f"{label} has unexpected fields: {unexpected}")
    backend_id = str(value.get("id") or "").strip()
    if not backend_id:
        raise ValueError(f"{label}.id must be a non-empty string")
    backend_config = value.get("config", {})
    if not isinstance(backend_config, Mapping):
        raise ValueError(f"{label}.config must be an object")
    backend = load_training_backend(backend_id)
    normalized = backend.normalize_config(
        dict(backend_config),
        label=f"{label}.config",
    )
    backend.validate(common_config, normalized)
    from gradlab.state_archive import validate_state_archive_runtime_contract

    validate_state_archive_runtime_contract(
        common_config,
        backend_id=backend_id,
        supported_priority_metrics=backend.state_archive_priority_metrics(),
    )
    return {"id": backend_id, "config": normalized}


def training_backend_contract_payload() -> dict[str, Any]:
    return {
        backend_id: load_training_backend(backend_id).contract_payload()
        for backend_id in registered_training_backend_ids()
    }


def training_backend_config_hash(config: Mapping[str, Any]) -> str:
    backend = config.get("training_backend")
    if not isinstance(backend, Mapping):
        return ""
    backend_config = backend.get("config")
    if not isinstance(backend_config, Mapping):
        return ""
    hash_config = dict(backend_config)
    # Resume source approval is an operational input injected after the
    # canonical recipe is built. The recipe document already binds that source
    # when it is user-configured; a supervisor recovery must keep attesting to
    # the same scientific backend config while pinning equivalent checkpoint
    # bytes for the learner.
    for key in ("resume", "resume_approval_hash", "resume_manifest"):
        if key in hash_config:
            hash_config[key] = None
    return canonical_json_sha256(
        hash_config,
        default=str,
        ensure_ascii=True,
    )


def training_backend_runtime_metadata(
    backend_id: str,
    backend_config: Mapping[str, Any],
) -> dict[str, str]:
    return dict(load_training_backend(backend_id).runtime_metadata(backend_config))
