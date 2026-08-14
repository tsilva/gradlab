from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np
import torch as th
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN

from gradlab.action_codecs import LegalTupleMultiDiscrete
from gradlab.action_distributions import LegalTupleCategoricalDistribution
from gradlab.model_inputs import (
    ProviderFrameStackBox,
    ProviderFrameStackMultiDiscrete,
)
from gradlab.policy_model_config import normalize_policy_model


def _activation(name: str) -> type[nn.Module]:
    return {"tanh": nn.Tanh, "relu": nn.ReLU}[name]


def _base_features_dim(
    observation_space: gym.Space,
    encoder: Mapping[str, Any],
) -> int:
    if encoder["kind"] == "nature_cnn":
        if not isinstance(observation_space, gym.spaces.Box):
            raise ValueError("nature_cnn requires a Box base observation")
        return int(encoder["features_dim"])
    return _context_features_dim(observation_space, label="base observation")


def _context_features_dim(observation_space: gym.Space, *, label: str) -> int:
    if isinstance(observation_space, gym.spaces.Box):
        return int(np.prod(observation_space.shape, dtype=np.int64))
    if isinstance(observation_space, gym.spaces.Discrete):
        return int(observation_space.n)
    if isinstance(observation_space, gym.spaces.MultiDiscrete):
        return int(np.sum(observation_space.nvec))
    raise ValueError(f"{label} must be a Box, Discrete, or MultiDiscrete space")


def _history_step_features_dim(observation_space: gym.Space, *, label: str) -> int:
    if isinstance(observation_space, ProviderFrameStackBox):
        return int(np.prod(observation_space.shape[1:], dtype=np.int64))
    if isinstance(observation_space, ProviderFrameStackMultiDiscrete):
        if observation_space.nvec.shape != (observation_space.frame_stack,):
            raise ValueError(f"{label} categorical history has an invalid shape")
        if np.any(observation_space.nvec != observation_space.nvec[0]):
            raise ValueError(f"{label} categorical vocabulary changes across history positions")
        return int(observation_space.nvec[0])
    raise ValueError(f"{label} is not a provider frame-stack space")


class SharedActorCriticFeatureExtractor(BaseFeaturesExtractor):
    """Encode the observation and every named context into one shared latent."""

    def __init__(
        self,
        observation_space: gym.Space,
        *,
        policy_model: Mapping[str, Any],
    ) -> None:
        normalized = normalize_policy_model(policy_model)
        self.policy_model = normalized
        if isinstance(observation_space, gym.spaces.Dict):
            spaces = observation_space.spaces
            base_space = spaces.get("observation")
            if not isinstance(
                base_space,
                gym.spaces.Box | gym.spaces.Discrete | gym.spaces.MultiDiscrete,
            ):
                raise ValueError(
                    "shared actor-critic policy requires Box, Discrete, or MultiDiscrete "
                    "'observation'"
                )
            unexpected = sorted(
                key
                for key in spaces
                if key != "observation"
                and (
                    not key.startswith("context/")
                    or not key.removeprefix("context/")
                    or "/" in key.removeprefix("context/")
                )
            )
            if unexpected:
                raise ValueError(
                    f"shared actor-critic observation has unexpected keys: {unexpected}"
                )
            context_names = tuple(
                sorted(key.removeprefix("context/") for key in spaces if key != "observation")
            )
        elif isinstance(
            observation_space,
            gym.spaces.Box | gym.spaces.Discrete | gym.spaces.MultiDiscrete,
        ):
            spaces = {}
            base_space = observation_space
            context_names = ()
        else:
            raise ValueError(
                "shared actor-critic policy requires a Box, Discrete, or MultiDiscrete "
                "observation, directly or under Dict key 'observation'"
            )

        encoder_dim = _base_features_dim(base_space, normalized["encoder"])
        context_dimensions = {
            name: _context_features_dim(
                spaces[f"context/{name}"],
                label=f"context {name!r}",
            )
            for name in context_names
        }
        history_names = tuple(
            name
            for name in context_names
            if isinstance(
                spaces[f"context/{name}"],
                ProviderFrameStackBox | ProviderFrameStackMultiDiscrete,
            )
        )
        current_context_names = tuple(
            name for name in context_names if name not in set(history_names)
        )
        history_depth = 0
        history_step_dimensions: dict[str, int] = {}
        history_features_dim = 0
        if history_names:
            depths = {int(spaces[f"context/{name}"].frame_stack) for name in history_names}
            if len(depths) != 1:
                raise ValueError("provider context histories must share one frame-stack depth")
            history_depth = depths.pop()
            history_step_dimensions = {
                name: _history_step_features_dim(
                    spaces[f"context/{name}"],
                    label=f"context {name!r}",
                )
                for name in history_names
            }
            history_features_dim = history_depth * sum(history_step_dimensions.values())
            for width in normalized.get("info_history_encoder", {}).get("hidden_sizes", ()):
                history_features_dim = int(width)
        fusion_input_dim = (
            encoder_dim
            + sum(context_dimensions[name] for name in current_context_names)
            + history_features_dim
        )
        fusion_dim = fusion_input_dim
        for width in normalized["fusion"]["hidden_sizes"]:
            fusion_dim = int(width)

        super().__init__(observation_space, features_dim=fusion_dim)
        self.base_observation_space = base_space
        self.context_names = context_names
        self.current_context_names = current_context_names
        self.provider_history_names = history_names
        self.context_dimensions = context_dimensions
        self.history_depth = history_depth
        self.history_step_dimensions = history_step_dimensions
        encoder = normalized["encoder"]
        if encoder["kind"] == "nature_cnn":
            self.observation_encoder: nn.Module = NatureCNN(
                base_space,
                features_dim=encoder_dim,
            )
        else:
            self.observation_encoder = nn.Flatten()

        if history_names:
            history_layers: list[nn.Module] = []
            previous = history_depth * sum(history_step_dimensions.values())
            history_config = normalized.get(
                "info_history_encoder",
                {"hidden_sizes": [], "activation": "tanh"},
            )
            history_activation = _activation(str(history_config["activation"]))
            for width in history_config["hidden_sizes"]:
                history_layers.extend((nn.Linear(previous, int(width)), history_activation()))
                previous = int(width)
            self.info_history_encoder: nn.Module | None = nn.Sequential(*history_layers)
        else:
            self.info_history_encoder = None

        layers: list[nn.Module] = []
        previous = fusion_input_dim
        activation = _activation(str(normalized["fusion"]["activation"]))
        for width in normalized["fusion"]["hidden_sizes"]:
            layers.extend((nn.Linear(previous, int(width)), activation()))
            previous = int(width)
        self.fusion = nn.Sequential(*layers)

    def forward(
        self,
        observations: th.Tensor | Mapping[str, th.Tensor],
    ) -> th.Tensor:
        if self.context_names:
            if not isinstance(observations, Mapping):
                raise ValueError("shared actor-critic context requires Dict observations")
            expected = {"observation"} | {f"context/{name}" for name in self.context_names}
            if set(observations) != expected:
                raise ValueError(
                    "shared actor-critic input keys disagree with the observation contract: "
                    f"expected {sorted(expected)}, got {sorted(observations)}"
                )
            base = observations["observation"]
        else:
            if isinstance(observations, Mapping):
                if set(observations) != {"observation"}:
                    raise ValueError("shared actor-critic input must contain only 'observation'")
                base = observations["observation"]
            else:
                base = observations

        encoded = self.observation_encoder(base)
        features = [encoded]
        batch_size = int(encoded.shape[0])
        if isinstance(observations, Mapping):
            features.extend(
                observations[f"context/{name}"].float().reshape(batch_size, -1)
                for name in self.current_context_names
            )
            if self.provider_history_names:
                per_step = th.cat(
                    [
                        observations[f"context/{name}"]
                        .float()
                        .reshape(
                            batch_size,
                            self.history_depth,
                            self.history_step_dimensions[name],
                        )
                        for name in self.provider_history_names
                    ],
                    dim=2,
                )
                temporal_major = per_step.reshape(batch_size, -1)
                assert self.info_history_encoder is not None
                features.append(self.info_history_encoder(temporal_major))
        fused = features[0] if len(features) == 1 else th.cat(features, dim=1)
        return self.fusion(fused)


class SharedActorCriticPolicy(ActorCriticPolicy):
    """Actor–critic policy with configurable shared or role-specific feature extractors."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule,
        *,
        policy_model: Mapping[str, Any],
        **kwargs: Any,
    ) -> None:
        self.policy_model = normalize_policy_model(policy_model)
        kwargs.pop("features_extractor_class", None)
        kwargs.pop("features_extractor_kwargs", None)
        kwargs.pop("share_features_extractor", None)
        kwargs.pop("net_arch", None)
        kwargs.pop("activation_fn", None)
        kwargs.pop("normalize_images", None)
        kwargs.pop("ortho_init", None)
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[],
            activation_fn=nn.Tanh,
            features_extractor_class=SharedActorCriticFeatureExtractor,
            features_extractor_kwargs={"policy_model": self.policy_model},
            share_features_extractor=bool(
                self.policy_model.get("share_features_extractor", True)
            ),
            normalize_images=bool(self.policy_model["normalize_images"]),
            ortho_init=bool(self.policy_model["orthogonal_init"]),
            **kwargs,
        )
        self._legal_tuple_distribution = None
        if isinstance(action_space, LegalTupleMultiDiscrete):
            self._legal_tuple_distribution = LegalTupleCategoricalDistribution(action_space)
            self.action_dist = self._legal_tuple_distribution

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor):
        distribution = getattr(self, "_legal_tuple_distribution", None)
        if distribution is None:
            return super()._get_action_dist_from_latent(latent_pi)
        action_logits = self.action_net(latent_pi)
        return distribution.proba_distribution(action_logits)

    def bind_action_contract(self, action_contract: Mapping[str, Any]) -> None:
        from gradlab.action_contract import validate_runtime_action_contract

        validate_runtime_action_contract(action_contract)
        policy = action_contract.get("policy")
        space = policy.get("space") if isinstance(policy, Mapping) else None
        runtime_legal = space.get("legal_tuples") if isinstance(space, Mapping) else None
        if isinstance(self.action_space, LegalTupleMultiDiscrete):
            expected = [list(row) for row in self.action_space.legal_tuples]
            if runtime_legal != expected:
                raise ValueError("policy legal tuples disagree with the runtime action contract")
            distribution = space.get("distribution")
            if distribution != {
                "type": self.action_space.distribution_type,
                "scoring": self.action_space.scoring_rule,
            }:
                raise ValueError("policy legal distribution disagrees with the runtime contract")
        elif runtime_legal is not None:
            raise ValueError("runtime requires legal tuples but the policy action space does not")

    def decision_distribution_and_value(
        self,
        obs: th.Tensor | Mapping[str, th.Tensor],
    ) -> tuple[Any, th.Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            assert isinstance(features, th.Tensor)
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            assert isinstance(features, tuple)
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        return (
            self._get_action_dist_from_latent(latent_pi),
            self.value_net(latent_vf),
        )

    def action_distribution(self, obs: th.Tensor | Mapping[str, th.Tensor]):
        return self.get_distribution(obs)

    def state_value(self, obs: th.Tensor | Mapping[str, th.Tensor]) -> th.Tensor:
        return self.predict_values(obs)

    def actor_log_probability(
        self,
        obs: th.Tensor | Mapping[str, th.Tensor],
        actions: th.Tensor,
    ) -> th.Tensor:
        return self.get_distribution(obs).log_prob(actions)

    def actor_image_feature_extractor(self) -> nn.Module:
        extractor = self.features_extractor
        if isinstance(extractor, SharedActorCriticFeatureExtractor):
            return extractor.observation_encoder
        return extractor

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data["policy_model"] = self.policy_model
        return data
