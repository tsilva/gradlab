from __future__ import annotations

from typing import Any

from stable_baselines3 import PPO


class GradLabPPO(PPO):
    """SB3-compatible PPO artifact owned by GradLab's tensor-native learner.

    The inherited constructor, serializer, loader, and prediction surface keep
    current GradLab PPO artifacts playable. Training is deliberately unavailable
    through SB3 so the custom backend cannot silently fall back to SB3's rollout
    collector or update loop.
    """

    def collect_rollouts(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError("GradLabPPO rollouts must be collected by gradlab.ppo")

    def train(self) -> None:
        raise RuntimeError("GradLabPPO updates must be executed by gradlab.ppo")

    def learn(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError("GradLabPPO training must be executed by gradlab.ppo")

    @classmethod
    def load(cls, *args: Any, **kwargs: Any) -> GradLabPPO:
        model = super().load(*args, **kwargs)
        # The inherited loader constructs SB3's host rollout buffer while
        # restoring the policy. It is not part of the artifact and is never a
        # GradLab training dependency, so release it immediately.
        model.rollout_buffer = None
        device_type = next(model.policy.parameters()).device.type
        cuda = device_type == "cuda"
        model.policy.optimizer.defaults["fused"] = cuda
        model.policy.optimizer.defaults["capturable"] = False
        model.policy.optimizer.defaults["foreach"] = False if cuda else None
        for group in model.policy.optimizer.param_groups:
            group["fused"] = cuda
            group["capturable"] = False
            group["foreach"] = False if cuda else None
        return model
