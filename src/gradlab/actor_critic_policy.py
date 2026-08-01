from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np
import torch as th
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN

from gradlab.policy_model_config import normalize_policy_model


def _activation(name: str) -> type[nn.Module]:
    return {"tanh": nn.Tanh, "relu": nn.ReLU}[name]


def _base_features_dim(
    observation_space: gym.spaces.Box,
    encoder: Mapping[str, Any],
) -> int:
    if encoder["kind"] == "nature_cnn":
        return int(encoder["features_dim"])
    return int(np.prod(observation_space.shape, dtype=np.int64))


def _context_features_dim(observation_space: gym.Space, *, label: str) -> int:
    if isinstance(observation_space, gym.spaces.Box):
        return int(np.prod(observation_space.shape, dtype=np.int64))
    if isinstance(observation_space, gym.spaces.Discrete):
        return int(observation_space.n)
    raise ValueError(f"{label} must be a Box or Discrete space")


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
            if not isinstance(base_space, gym.spaces.Box):
                raise ValueError("shared actor-critic policy requires Box 'observation'")
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
        elif isinstance(observation_space, gym.spaces.Box):
            spaces = {}
            base_space = observation_space
            context_names = ()
        else:
            raise ValueError(
                "shared actor-critic policy requires a Box observation or a Dict with "
                "Box 'observation'"
            )

        encoder_dim = _base_features_dim(base_space, normalized["encoder"])
        context_dimensions = {
            name: _context_features_dim(
                spaces[f"context/{name}"],
                label=f"context {name!r}",
            )
            for name in context_names
        }
        fusion_input_dim = encoder_dim + sum(context_dimensions.values())
        fusion_dim = fusion_input_dim
        for width in normalized["fusion"]["hidden_sizes"]:
            fusion_dim = int(width)

        super().__init__(observation_space, features_dim=fusion_dim)
        self.base_observation_space = base_space
        self.context_names = context_names
        self.context_dimensions = context_dimensions
        encoder = normalized["encoder"]
        if encoder["kind"] == "nature_cnn":
            self.observation_encoder: nn.Module = NatureCNN(
                base_space,
                features_dim=encoder_dim,
            )
        else:
            self.observation_encoder = nn.Flatten()

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
                for name in self.context_names
            )
        fused = features[0] if len(features) == 1 else th.cat(features, dim=1)
        return self.fusion(fused)


class SharedActorCriticPolicy(ActorCriticPolicy):
    """Actor–critic policy sharing observation, context, encoder, and fusion MLP."""

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
            share_features_extractor=True,
            normalize_images=bool(self.policy_model["normalize_images"]),
            ortho_init=bool(self.policy_model["orthogonal_init"]),
            **kwargs,
        )

    def decision_distribution_and_value(
        self,
        obs: th.Tensor | Mapping[str, th.Tensor],
    ) -> tuple[Any, th.Tensor]:
        features = self.extract_features(obs)
        assert isinstance(features, th.Tensor)
        latent_pi, latent_vf = self.mlp_extractor(features)
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
