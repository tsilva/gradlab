from __future__ import annotations

from gradlab.learner_profiles import resolve_local_learner_profile
from gradlab.training_lifecycle import TrainingExecutionMode


def test_m1_mps_profile_selects_native_mps_lifecycle() -> None:
    profile = resolve_local_learner_profile("m1-mps")

    assert profile is not None
    assert profile.device == "mps"
    assert profile.execution_mode == TrainingExecutionMode.LOCAL_NATIVE
    assert profile.recipe_overrides == ("train.backend.config.device=mps",)
