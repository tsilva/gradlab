from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from gradlab.training.sb3 import PPO_DEFAULT_CONFIG, PPO_PROGRESS_FIELDS, _normalize_ppo
from gradlab.training_backend import BackendContext, CHECKPOINT_EVAL_ACCEPTANCE


PPO_PRECISIONS = ("fp32", "amp-fp16", "amp-bf16")
PPO_EXECUTION_PROFILES = (
    "sb3-parity",
    "compiled-parity",
    "compiled-fused-parity",
    "max-throughput",
)
GRADLAB_PPO_DEFAULT_CONFIG: dict[str, Any] = {
    **PPO_DEFAULT_CONFIG,
    "precision": "fp32",
    "execution_profile": "max-throughput",
}


def _normalize_gradlab_ppo(config: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    unexpected = sorted(set(config) - set(GRADLAB_PPO_DEFAULT_CONFIG))
    if unexpected:
        raise ValueError(f"{label} has unexpected fields: {unexpected}")
    precision = config.get("precision", GRADLAB_PPO_DEFAULT_CONFIG["precision"])
    if precision not in PPO_PRECISIONS:
        choices = ", ".join(PPO_PRECISIONS)
        raise ValueError(f"{label}.precision must be one of {choices}")
    execution_profile = config.get(
        "execution_profile",
        GRADLAB_PPO_DEFAULT_CONFIG["execution_profile"],
    )
    if execution_profile not in PPO_EXECUTION_PROFILES:
        choices = ", ".join(PPO_EXECUTION_PROFILES)
        raise ValueError(f"{label}.execution_profile must be one of {choices}")
    normalized = _normalize_ppo(
        {
            key: value
            for key, value in config.items()
            if key not in {"precision", "execution_profile"}
        },
        label=label,
    )
    normalized["precision"] = precision
    normalized["execution_profile"] = execution_profile
    return normalized


class GradLabPPOBackend:
    backend_id = "gradlab.ppo"

    def normalize_config(
        self,
        config: Mapping[str, Any],
        *,
        label: str,
    ) -> dict[str, Any]:
        return _normalize_gradlab_ppo(config, label=label)

    def validate(
        self,
        common_config: Mapping[str, Any],
        backend_config: Mapping[str, Any],
    ) -> None:
        del backend_config
        if common_config.get("policy_model") is None:
            raise ValueError("gradlab.ppo requires train_config.policy_model")

    def run(self, context: BackendContext):
        from gradlab.training.ppo_engine import run_gradlab_ppo

        return run_gradlab_ppo(context, progress_fields=PPO_PROGRESS_FIELDS)

    def acceptance_mode(self, backend_config: Mapping[str, Any]) -> str:
        del backend_config
        return CHECKPOINT_EVAL_ACCEPTANCE

    def state_archive_priority_metrics(self) -> tuple[str, ...]:
        return ("value_error",)

    def contract_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": "available",
            "defaults": GRADLAB_PPO_DEFAULT_CONFIG,
            "state_archive_priority_metrics": ["value_error"],
        }

    def runtime_metadata(
        self,
        backend_config: Mapping[str, Any],
    ) -> Mapping[str, str]:
        del backend_config
        return {
            "training_backend_id": self.backend_id,
            "algorithm_id": "ppo",
            "model_class": "gradlab.ppo.GradLabPPO",
        }


BACKENDS = {"gradlab.ppo": GradLabPPOBackend()}
