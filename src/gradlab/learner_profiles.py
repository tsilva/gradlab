from __future__ import annotations

import platform
from dataclasses import dataclass

from gradlab.training_lifecycle import TrainingExecutionMode


@dataclass(frozen=True)
class LocalLearnerProfile:
    """A named native learner configuration for the local training command."""

    profile_id: str
    device: str
    execution_mode: TrainingExecutionMode
    required_system: str
    required_machine: str
    requires_mps: bool = False

    @property
    def recipe_overrides(self) -> tuple[str, ...]:
        return (f"train.backend.config.device={self.device}",)

    def validate_host(self) -> None:
        if (
            platform.system() != self.required_system
            or platform.machine() != self.required_machine
        ):
            raise RuntimeError(
                f"learner profile {self.profile_id!r} requires "
                f"{self.required_system}/{self.required_machine}, got "
                f"{platform.system()}/{platform.machine()}"
            )
        if self.requires_mps:
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError(
                    f"learner profile {self.profile_id!r} requires PyTorch with MPS support"
                ) from exc
            if not torch.backends.mps.is_available():
                raise RuntimeError(
                    f"learner profile {self.profile_id!r} requires an available Apple MPS device"
                )


LOCAL_LEARNER_PROFILES: dict[str, LocalLearnerProfile] = {
    "m1-mps": LocalLearnerProfile(
        profile_id="m1-mps",
        device="mps",
        execution_mode=TrainingExecutionMode.LOCAL_NATIVE,
        required_system="Darwin",
        required_machine="arm64",
        requires_mps=True,
    ),
}


def local_learner_profile_names() -> tuple[str, ...]:
    return tuple(sorted(LOCAL_LEARNER_PROFILES))


def resolve_local_learner_profile(value: str | None) -> LocalLearnerProfile | None:
    profile_id = str(value or "").strip()
    if not profile_id:
        return None
    try:
        return LOCAL_LEARNER_PROFILES[profile_id]
    except KeyError as exc:
        choices = ", ".join(local_learner_profile_names())
        raise ValueError(
            f"unknown local learner profile {profile_id!r}; choose from {choices}"
        ) from exc
