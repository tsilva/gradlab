"""Local exact restoration contracts at a live Policy decision boundary.

The registry is intentionally narrow. A portable provider snapshot alone does not
certify the Policy or wrappers. Payloads never leave the local episode store.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import random
from typing import Any

import numpy as np
import torch

from gradlab.policy_registry import POLICY_ALGORITHM_SPECS

# Authoritative supported provider/Policy combinations; contract tests must cover
# each adapter. Unknown providers, Policy implementations and exploration states
# have no approximate fallback.
LIVE_RESTORATION = {
    "env-supermariobrosnes-turbo-emu": frozenset({"ppo", "a2c", "action-program"}),
    "env-breakoutatari2600-turbo-native": frozenset({"ppo", "a2c", "action-program"}),
}


@contextmanager
def preserve_sampling():
    """Preparation and restoration cannot advance any active global Policy stream."""
    python = random.getstate()
    numpy = np.random.get_state()
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    mps = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    try:
        yield
    finally:
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        if mps is not None:
            torch.mps.set_rng_state(mps)


class LiveRestoration:
    SESSION_FIELDS = (
        "policy_obs",
        "current_frame",
        "active_task_state",
        "active_info_value",
        "active_seed",
        "episode",
        "step_index",
        "total_reward",
        "max_x_pos",
    )

    def __init__(self, session):
        self.session = session
        self.algorithm = None
        reason = self._unsupported_reason()
        self.capability = {
            "supported": reason is None,
            "reason": reason,
            "activation": "explicit",
            "contract": "decision-boundary-v1",
        }
        self.capability["automatic_candidate"] = bool(
            reason is None
            and session.env.runtime.descriptor.provider_id == "env-breakoutatari2600-turbo-native"
        )

    def _unsupported_reason(self):
        from gradlab.training.sb3_vec_env import GradLabVecEnv

        session = self.session
        if type(getattr(session, "env", None)) is not GradLabVecEnv:
            return "This environment wrapper has no exact live restoration contract."
        runtime = session.env.runtime
        policy = getattr(session, "policy_runtime", None)
        if policy is None:
            return "The Policy has no registered restoration contract."
        self.algorithm = policy.capabilities.algorithm_id
        if self.algorithm not in LIVE_RESTORATION.get(runtime.descriptor.provider_id, ()):
            return "This provider and Policy combination has no exact restoration contract."
        model_class = f"{type(session.model).__module__}.{type(session.model).__name__}"
        if model_class not in POLICY_ALGORITHM_SPECS[self.algorithm].model_classes:
            return "This Policy implementation has no exact restoration contract."
        if getattr(session.model, "use_sde", False):
            return "State-dependent exploration is not certified for live restoration."
        if runtime.num_envs != 1 or runtime.state_archive is not None:
            return "Live restoration requires a single lane without archive curricula."
        if not (
            runtime.descriptor.supports_live_snapshots
            and runtime.descriptor.live_snapshots_deterministic
            and runtime.descriptor.snapshot_codec_id
        ):
            return "The provider does not declare exact snapshot continuation."
        return None

    def capture(self) -> dict[str, Any]:
        if not self.capability["supported"]:
            raise ValueError(self.capability["reason"])
        session = self.session
        with preserve_sampling():
            memory = (
                session.model.capture_execution_state()
                if self.algorithm == "action-program"
                else {}
            )
            return deepcopy(
                {
                    "environment": session.env.runtime.capture_playback_state(),
                    "policy_memory": memory,
                    "session": {name: getattr(session, name) for name in self.SESSION_FIELDS},
                    "frames": tuple(session.frames or ()),
                }
            )

    def _restore(self, state):
        from collections import deque

        session = self.session
        session.env.runtime.restore_playback_state(state["environment"])
        if self.algorithm == "action-program":
            session.model.restore_execution_state(state["policy_memory"])
        for name, value in state["session"].items():
            setattr(session, name, deepcopy(value))
        session.frames = deque(deepcopy(state["frames"])) or None
        session.env.reset_infos = session.env.runtime.reset_infos
        session.env.mark_policy_resumed()
        session.env.take_step_diagnostics()
        session.last_transition = None

    def restore(self, state) -> None:
        """Restore transactionally; a failed candidate restores the current state.

        A caller replaces the timeline only after this succeeds. Capture is owned
        before mutating buffers, and the rollback itself is verified byte-for-byte.
        """
        from gradlab.play_trajectory import pack_record

        with preserve_sampling():
            previous = self.capture()
            previous_transition = self.session.last_transition
            try:
                self._restore(state)
                if pack_record(self.capture()) != pack_record(state):
                    raise ValueError("exact restoration verification failed")
            except Exception:
                self._restore(previous)
                self.session.last_transition = previous_transition
                if pack_record(self.capture()) != pack_record(previous):
                    raise RuntimeError("restoration rollback verification failed")
                raise
