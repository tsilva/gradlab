from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn.functional as F
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN

from gradlab.policy_model_config import POLICY_ROLES, normalize_policy_model


def _activation(name: str) -> type[nn.Module]:
    return {"tanh": nn.Tanh, "relu": nn.ReLU}[name]


class RoutedObservationEncoder(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        *,
        base_space: gym.spaces.Box,
        encoder: Mapping[str, Any],
    ) -> None:
        kind = str(encoder["kind"])
        if kind == "nature_cnn":
            features_dim = int(encoder["features_dim"])
            super().__init__(observation_space, features_dim=features_dim)
            self.encoder = NatureCNN(base_space, features_dim=features_dim)
        elif kind == "flatten":
            features_dim = int(np.prod(base_space.shape, dtype=np.int64))
            super().__init__(observation_space, features_dim=features_dim)
            self.encoder = nn.Flatten()
        else:  # pragma: no cover - normalized configuration prevents this.
            raise ValueError(f"unsupported observation encoder {kind!r}")

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.encoder(observations)


def build_configured_head(
    input_dim: int,
    config: Mapping[str, Any],
    *,
    device: th.device,
) -> tuple[nn.Sequential, int]:
    """Build one role-specific hidden stack and return its latent width."""

    layers: list[nn.Module] = []
    previous = int(input_dim)
    activation = _activation(str(config["activation"]))
    for width in config["hidden_sizes"]:
        layers.extend((nn.Linear(previous, int(width)), activation()))
        previous = int(width)
    return nn.Sequential(*layers).to(device), previous


class RoutedMlpExtractor(nn.Module):
    def __init__(
        self,
        action_input_dim: int,
        value_input_dim: int,
        heads: Mapping[str, Any],
        *,
        device: th.device,
    ) -> None:
        super().__init__()
        self.policy_net, self.latent_dim_pi = build_configured_head(
            action_input_dim,
            heads["action"],
            device=device,
        )
        self.value_net, self.latent_dim_vf = build_configured_head(
            value_input_dim,
            heads["state_value"],
            device=device,
        )

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class RoutedActorCriticPolicy(ActorCriticPolicy):
    """Actor–critic policy with named context routed independently by role."""

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
        if isinstance(observation_space, gym.spaces.Dict):
            spaces = observation_space.spaces
            base_space = spaces.get("observation")
            if not isinstance(base_space, gym.spaces.Box):
                raise ValueError("configured actor-critic policy requires Box 'observation'")
        elif isinstance(observation_space, gym.spaces.Box):
            spaces = {}
            base_space = observation_space
            if self.policy_model["context_encoders"] or self.policy_model["routes"]:
                raise ValueError(
                    "configured actor-critic policy requires Dict observations when context "
                    "is declared"
                )
        else:
            raise ValueError(
                "configured actor-critic policy requires a Box observation or a Dict with "
                "Box 'observation'"
            )
        self.base_observation_space = base_space
        self._role_contexts = {
            role: tuple(
                name
                for name, roles in self.policy_model["routes"].items()
                if role in roles
            )
            for role in POLICY_ROLES
        }
        self._context_dimensions: dict[str, int] = {}
        for name, encoder in self.policy_model["context_encoders"].items():
            key = f"context/{name}"
            if key not in spaces:
                raise ValueError(f"routed policy observation space is missing {key!r}")
            context_space = spaces[key]
            if encoder["kind"] == "identity":
                if not isinstance(context_space, gym.spaces.Box):
                    raise ValueError(f"identity context {name!r} requires a Box space")
                self._context_dimensions[name] = int(
                    np.prod(context_space.shape, dtype=np.int64)
                )
            else:
                if not isinstance(context_space, gym.spaces.Discrete):
                    raise ValueError(f"one_hot context {name!r} requires a Discrete space")
                self._context_dimensions[name] = int(context_space.n)
        if isinstance(observation_space, gym.spaces.Dict):
            expected_keys = {"observation"} | {
                f"context/{name}" for name in self.policy_model["context_encoders"]
            }
            if set(spaces) != expected_keys:
                raise ValueError(
                    "configured policy observation keys disagree with policy_model: "
                    f"expected {sorted(expected_keys)}, got {sorted(spaces)}"
                )
        topology = self.policy_model["topology"]
        self._encoder_specs = (
            {"action": topology["encoder"], "state_value": topology["encoder"]}
            if topology["kind"] == "shared_encoder"
            else dict(topology["encoders"])
        )
        self._extractor_build_count = 0
        share = topology["kind"] == "shared_encoder"
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
            features_extractor_class=RoutedObservationEncoder,
            features_extractor_kwargs={},
            share_features_extractor=share,
            normalize_images=bool(self.policy_model["normalize_images"]),
            ortho_init=bool(self.policy_model["orthogonal_init"]),
            **kwargs,
        )

    def make_features_extractor(self) -> BaseFeaturesExtractor:
        role = "action" if self._extractor_build_count == 0 else "state_value"
        self._extractor_build_count += 1
        return RoutedObservationEncoder(
            self.observation_space,
            base_space=self.base_observation_space,
            encoder=self._encoder_specs[role],
        )

    def _build_mlp_extractor(self) -> None:
        action_dim = int(self.pi_features_extractor.features_dim) + sum(
            self._context_dimensions[name] for name in self._role_contexts["action"]
        )
        value_dim = int(self.vf_features_extractor.features_dim) + sum(
            self._context_dimensions[name] for name in self._role_contexts["state_value"]
        )
        self.mlp_extractor = RoutedMlpExtractor(
            action_dim,
            value_dim,
            self.policy_model["heads"],
            device=self.device,
        )

    def _base_tensor(self, obs: Mapping[str, th.Tensor] | th.Tensor) -> th.Tensor:
        if isinstance(obs, Mapping):
            if "observation" not in obs:
                raise ValueError("configured policy input is missing 'observation'")
            base = obs["observation"]
        else:
            base = obs
        return preprocess_obs(
            base,
            self.base_observation_space,
            normalize_images=self.normalize_images,
        )

    def _context_tensor(
        self,
        obs: Mapping[str, th.Tensor] | th.Tensor,
        name: str,
        *,
        batch_size: int,
    ) -> th.Tensor:
        if not isinstance(obs, Mapping):
            raise ValueError("configured policy context requires Dict observations")
        key = f"context/{name}"
        if key not in obs:
            raise ValueError(f"routed policy input is missing {key!r}")
        value = obs[key]
        encoder = self.policy_model["context_encoders"][name]["kind"]
        if encoder == "identity":
            return value.float().reshape(batch_size, -1)
        indices = value.long().reshape(batch_size, -1)
        if indices.shape[1] != 1:
            raise ValueError(f"categorical context {name!r} must contain one index")
        return F.one_hot(
            indices[:, 0],
            num_classes=self._context_dimensions[name],
        ).float()

    def _append_context(
        self,
        features: th.Tensor,
        obs: Mapping[str, th.Tensor] | th.Tensor,
        role: str,
    ) -> th.Tensor:
        values = [features]
        batch_size = int(features.shape[0])
        values.extend(
            self._context_tensor(obs, name, batch_size=batch_size)
            for name in self._role_contexts[role]
        )
        return values[0] if len(values) == 1 else th.cat(values, dim=1)

    def _role_features(
        self,
        obs: Mapping[str, th.Tensor] | th.Tensor,
        role: str,
    ) -> th.Tensor:
        base = self._base_tensor(obs)
        extractor = (
            self.pi_features_extractor
            if role == "action"
            else self.vf_features_extractor
        )
        return self._append_context(extractor(base), obs, role)

    def _joint_features(
        self,
        obs: Mapping[str, th.Tensor] | th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        base = self._base_tensor(obs)
        if self.share_features_extractor:
            shared = self.features_extractor(base)
            return (
                self._append_context(shared, obs, "action"),
                self._append_context(shared, obs, "state_value"),
            )
        return (
            self._append_context(self.pi_features_extractor(base), obs, "action"),
            self._append_context(self.vf_features_extractor(base), obs, "state_value"),
        )

    def forward(
        self,
        obs: Mapping[str, th.Tensor] | th.Tensor,
        deterministic: bool = False,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        action_features, value_features = self._joint_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(action_features)
        latent_vf = self.mlp_extractor.forward_critic(value_features)
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def get_distribution(self, obs: Mapping[str, th.Tensor] | th.Tensor):
        features = self._role_features(obs, "action")
        latent = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent)

    def predict_values(self, obs: Mapping[str, th.Tensor] | th.Tensor) -> th.Tensor:
        features = self._role_features(obs, "state_value")
        latent = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent)

    def evaluate_actions(
        self,
        obs: Mapping[str, th.Tensor] | th.Tensor,
        actions: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor | None]:
        action_features, value_features = self._joint_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(action_features)
        latent_vf = self.mlp_extractor.forward_critic(value_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        return (
            self.value_net(latent_vf),
            distribution.log_prob(actions),
            distribution.entropy(),
        )

    def action_distribution(self, obs: Mapping[str, th.Tensor] | th.Tensor):
        return self.get_distribution(obs)

    def state_value(self, obs: Mapping[str, th.Tensor] | th.Tensor) -> th.Tensor:
        return self.predict_values(obs)

    def decision_distribution_and_value(
        self,
        obs: Mapping[str, th.Tensor] | th.Tensor,
    ) -> tuple[Any, th.Tensor]:
        action_features, value_features = self._joint_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(action_features)
        latent_vf = self.mlp_extractor.forward_critic(value_features)
        return (
            self._get_action_dist_from_latent(latent_pi),
            self.value_net(latent_vf),
        )

    def actor_log_probability(
        self,
        obs: Mapping[str, th.Tensor] | th.Tensor,
        actions: th.Tensor,
    ) -> th.Tensor:
        return self.get_distribution(obs).log_prob(actions)

    def actor_image_feature_extractor(self) -> nn.Module:
        extractor = self.pi_features_extractor
        nested = getattr(extractor, "encoder", None)
        return nested if isinstance(nested, nn.Module) else extractor

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data["policy_model"] = self.policy_model
        return data
