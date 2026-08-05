from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from gradlab.play_attribution import (
    ActionLogProbForward,
    PolicyActionAttributor,
    actor_image_feature_extractor,
    attribution_capability,
    find_last_conv2d,
)
from gradlab.play_debug import PolicyDecision
from gradlab.play_session import (
    _PlaybackSession,
    _PlaybackTransition,
    render_attribution_stack,
    render_obs_stack,
)


class TinyImageExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image.float() / 255.0)


class TinyDictExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractors = nn.ModuleDict({"image": TinyImageExtractor(), "task": nn.Identity()})


class TinyPolicy(nn.Module):
    def __init__(self, *, dict_obs: bool = False, preserve_obs_dtype: bool = False):
        super().__init__()
        self.action_space = gym.spaces.Discrete(2)
        self.pi_features_extractor = TinyDictExtractor() if dict_obs else TinyImageExtractor()
        self.action_net = nn.Linear(2, 2)
        self.value_net = nn.Linear(2, 1)
        self.dict_obs = dict_obs
        self.preserve_obs_dtype = preserve_obs_dtype

    def obs_to_tensor(self, observation):
        dtype = None if self.preserve_obs_dtype else torch.float32
        if isinstance(observation, dict):
            return {
                key: torch.as_tensor(value, dtype=dtype) for key, value in observation.items()
            }, True
        return torch.as_tensor(observation, dtype=dtype), True

    def image_features(self, obs) -> torch.Tensor:
        if isinstance(obs, dict):
            return self.pi_features_extractor.extractors["image"](obs["image"])
        return self.pi_features_extractor(obs)

    def evaluate_actions(self, obs, actions):
        features = self.image_features(obs)
        logits = self.action_net(features)
        log_probs = torch.log_softmax(logits, dim=1)
        selected = actions.reshape(-1, 1).long()
        log_prob = log_probs.gather(1, selected).squeeze(1)
        values = self.value_net(features)
        entropy = -(log_probs.exp() * log_probs).sum(dim=1)
        return values, log_prob, entropy


def test_last_conv_selection_uses_final_conv_layer() -> None:
    extractor = TinyImageExtractor()

    assert find_last_conv2d(extractor) is extractor.net[2]


def test_actor_image_feature_extractor_handles_dict_image_branch() -> None:
    policy = TinyPolicy(dict_obs=True)

    assert actor_image_feature_extractor(policy) is policy.pi_features_extractor.extractors["image"]


def test_action_log_prob_forward_returns_gradient_scalar_and_preserves_task() -> None:
    policy = TinyPolicy(dict_obs=True)
    obs = {
        "image": np.ones((1, 4, 84, 84), dtype=np.float32),
        "task": np.array([[0.0, 1.0]], dtype=np.float32),
    }
    forward = ActionLogProbForward(policy, obs, np.array([1]))
    image = forward.image_tensor.detach().requires_grad_(True)

    output = forward(image)
    output.sum().backward()

    assert output.shape == (1,)
    assert output.requires_grad
    assert image.grad is not None
    assert torch.equal(forward.fixed_obs["task"], torch.as_tensor(obs["task"]))


def test_action_log_prob_forward_converts_uint8_image_to_float_for_gradients() -> None:
    policy = TinyPolicy(preserve_obs_dtype=True)
    obs = np.ones((1, 4, 84, 84), dtype=np.uint8)

    forward = ActionLogProbForward(policy, obs, np.array([1]))
    image = forward.image_tensor.detach().requires_grad_(True)
    output = forward(image)
    output.sum().backward()

    assert forward.image_tensor.dtype == torch.float32
    assert image.grad is not None


def test_gradcam_returns_normalized_spatial_heatmap() -> None:
    policy = TinyPolicy()
    model = SimpleNamespace(policy=policy)
    attributor = PolicyActionAttributor(model)
    obs = np.random.default_rng(3).integers(0, 255, size=(1, 4, 84, 84), dtype=np.uint8)

    heatmap = attributor.attribute("gradcam", obs, np.array([1]))

    assert heatmap.shape == (84, 84)
    assert heatmap.dtype == np.float32
    assert 0.0 <= float(heatmap.min()) <= float(heatmap.max()) <= 1.0


def test_support_detection_does_not_activate_attribution() -> None:
    model = SimpleNamespace(policy=TinyPolicy())

    capability = attribution_capability(model, "ppo")

    assert capability == {
        "target": "selected_action_log_probability",
        "supported_modes": ["gradcam", "occlusion"],
        "unavailable_reason": None,
    }


def test_attribution_preserves_python_numpy_and_torch_rng() -> None:
    import random

    policy = TinyPolicy()
    model = SimpleNamespace(policy=policy)
    attributor = PolicyActionAttributor(model)
    obs = np.ones((1, 4, 84, 84), dtype=np.uint8)
    random.seed(7)
    np.random.seed(8)
    torch.manual_seed(9)
    expected = (random.random(), np.random.random(), torch.rand(1))
    random.seed(7)
    np.random.seed(8)
    torch.manual_seed(9)

    attributor.attribute("gradcam", obs, np.array([1]))
    actual = (random.random(), np.random.random(), torch.rand(1))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_occlusion_returns_normalized_spatial_heatmap() -> None:
    policy = TinyPolicy()
    model = SimpleNamespace(policy=policy)
    attributor = PolicyActionAttributor(model, occlusion_window=28, occlusion_stride=28)
    obs = np.random.default_rng(4).integers(0, 255, size=(1, 4, 84, 84), dtype=np.uint8)

    heatmap = attributor.attribute("occlusion", obs, np.array([0]))

    assert heatmap.shape == (84, 84)
    assert heatmap.dtype == np.float32
    assert 0.0 <= float(heatmap.min()) <= float(heatmap.max()) <= 1.0


def test_attribution_stack_is_separate_rgba_and_matches_clean_observation_layout() -> None:
    frames = [np.full((84, 84, 1), value, dtype=np.uint8) for value in (10, 50, 90, 130)]
    plain = render_obs_stack(frames, scale=2)
    heatmap = np.zeros((84, 84), dtype=np.float32)
    heatmap[20:40, 30:50] = 1.0

    overlay = render_attribution_stack(tuple(frames), heatmap, scale=2)

    assert plain.shape == (168, 672, 3)
    assert overlay.shape == (168, 672, 4)
    assert plain.dtype == overlay.dtype == np.uint8
    assert overlay[..., 3].max() == 255
    assert overlay[..., 3].min() == 0


def _attribution_transition(sequence: int = 3) -> _PlaybackTransition:
    return _PlaybackTransition(
        sequence=sequence,
        episode=1,
        step=sequence,
        seed=1,
        start_id=None,
        model_obs=np.ones((1, 4, 8, 8), dtype=np.uint8),
        decision=PolicyDecision(
            raw_action=np.asarray([1]),
            executed_action=np.asarray([1]),
            action_selection_mode="stochastic",
        ),
        action_source="policy",
        executed_action=1,
        diagnostics=None,
        info={},
        before_frame=None,
        after_frame=None,
        before_frames=(np.ones((8, 8, 1), dtype=np.uint8),),
        after_frames=(),
        attribution=None,
        pre_task=None,
        next_task=None,
        reward=0.0,
        total_reward=0.0,
        max_x_pos=0,
        terminated=False,
        truncated=False,
        completed=False,
        boundary=False,
    )


def test_live_attribution_is_lazy_reused_and_recomputes_only_latest_transition() -> None:
    calls = []

    class FakeAttributor:
        def attribute(self, mode, model_obs, action):
            calls.append((mode, model_obs, action))
            return np.full((8, 8), len(calls), dtype=np.float32)

    created = []
    session = _PlaybackSession.__new__(_PlaybackSession)
    session.model = object()
    session._attributor_factory = lambda model: created.append(model) or FakeAttributor()
    session.attributor = None
    session.attribution_capability = {
        "supported_modes": ["gradcam", "occlusion"],
        "unavailable_reason": None,
    }
    session.attribution_mode = "none"
    session.attribution_interval = 1
    session.attribution_status = "off"
    session.attribution_error = None
    session.attribution_generation = 0
    session.attribution_last_computed_sequence = None
    session.last_transition = _attribution_transition()

    session.configure_attribution("gradcam")
    first = session.last_transition
    session.configure_attribution("occlusion")
    second = session.last_transition
    session.configure_attribution("none")

    assert created == [session.model]
    assert [call[0] for call in calls] == ["gradcam", "occlusion"]
    assert first.attribution_generation == 1
    assert second.attribution_generation == 2
    assert second.attribution_mode == "occlusion"
    assert session.attribution_interval == 8
    assert session.last_transition.attribution_status == "off"
    assert session.last_transition.attribution is None


def test_live_attribution_failure_enters_error_without_fabricating_a_map() -> None:
    class BrokenAttributor:
        def attribute(self, mode, model_obs, action):
            raise RuntimeError("detector exploded")

    session = _PlaybackSession.__new__(_PlaybackSession)
    session.model = object()
    session._attributor_factory = lambda _model: BrokenAttributor()
    session.attributor = None
    session.attribution_capability = {
        "supported_modes": ["gradcam"],
        "unavailable_reason": None,
    }
    session.attribution_mode = "none"
    session.attribution_interval = 1
    session.attribution_status = "off"
    session.attribution_error = None
    session.attribution_generation = 0
    session.attribution_last_computed_sequence = None
    session.last_transition = _attribution_transition()

    with pytest.raises(RuntimeError, match="detector exploded"):
        session.configure_attribution("gradcam")

    assert session.attribution_status == "error"
    assert session.last_transition.attribution_status == "error"
    assert session.last_transition.attribution is None
