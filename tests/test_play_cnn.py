from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from gradlab.play_cnn import PolicyCNNInspector, cnn_inspection_capability
from gradlab.play_session import _PlaybackSession
from tests.test_play_attribution import TinyPolicy, _attribution_transition


def test_cnn_capability_describes_layers_and_exact_receptive_fields() -> None:
    model = SimpleNamespace(policy=TinyPolicy())

    capability = cnn_inspection_capability(model)

    assert capability["default_layer_id"] == "net.0"
    assert capability["unavailable_reason"] is None
    assert [layer["id"] for layer in capability["layers"]] == ["net.0", "net.2"]
    assert capability["layers"][0]["receptive_field"] == [3.0, 3.0]
    assert capability["layers"][1]["receptive_field"] == [5.0, 5.0]
    assert capability["layers"][1]["feature_stride"] == [1.0, 1.0]
    assert capability["layers"][1]["response_stage"] == "post-ReLU output"


def test_cnn_inspection_ranks_raw_responses_and_builds_spatial_atlas() -> None:
    policy = TinyPolicy()
    first = policy.pi_features_extractor.net[0]
    second = policy.pi_features_extractor.net[2]
    with torch.no_grad():
        first.weight.fill_(0.05)
        first.bias.zero_()
        second.weight[0].fill_(0.01)
        second.weight[1].fill_(0.02)
        second.bias.zero_()
    inspector = PolicyCNNInspector(SimpleNamespace(policy=policy))
    observation = np.full((1, 4, 8, 8), 255, dtype=np.uint8)

    result = inspector.inspect(
        observation,
        layer_id="net.2",
        top_k=2,
        generation=7,
    )

    assert result.generation == 7
    assert result.input_shape == (8, 8)
    assert result.activation_shape == (8, 8)
    assert [item["filter_index"] for item in result.filters] == [1, 0]
    assert result.filters[0]["peak_response"] > result.filters[1]["peak_response"] > 0
    assert result.filters[0]["peak_input_region"] == {
        "x0": 0.0,
        "y0": 0.0,
        "x1": 5.0,
        "y1": 5.0,
        "center_x": 2.5,
        "center_y": 2.5,
    }
    assert result.atlas.shape == (16, 24, 4)
    assert result.atlas.dtype == np.uint8
    assert result.atlas[..., 3].max() == 255
    payload = result.payload()
    assert payload["atlas"]["winner_tile"] == 0
    assert payload["filters"][0]["kernel_tile"] == 1
    assert payload["filters"][0]["activation_tile"] == 2
    assert "atlas" not in payload["filters"][0]


def test_cnn_inspection_preserves_rng_and_policy_mode() -> None:
    policy = TinyPolicy()
    policy.train(True)
    inspector = PolicyCNNInspector(SimpleNamespace(policy=policy))
    observation = np.ones((1, 4, 8, 8), dtype=np.uint8)
    random.seed(31)
    np.random.seed(32)
    torch.manual_seed(33)
    expected = (random.random(), np.random.random(), torch.rand(1))
    random.seed(31)
    np.random.seed(32)
    torch.manual_seed(33)

    inspector.inspect(observation, layer_id="net.0", top_k=2, generation=1)
    actual = (random.random(), np.random.random(), torch.rand(1))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
    assert policy.training is True


def test_cnn_capability_fails_closed_without_an_actor_image_encoder() -> None:
    capability = cnn_inspection_capability(SimpleNamespace(policy=nn.Identity()))

    assert capability["layers"] == []
    assert capability["default_layer_id"] is None
    assert "actor feature extractor" in capability["unavailable_reason"]


def test_live_cnn_inspection_is_lazy_reconfigurable_and_clearable() -> None:
    calls = []
    inspection = SimpleNamespace(generation=1)

    class FakeInspector:
        def inspect(self, model_obs, *, layer_id, top_k, generation):
            calls.append((model_obs, layer_id, top_k, generation))
            inspection.generation = generation
            return inspection

    session = _PlaybackSession.__new__(_PlaybackSession)
    session.model = object()
    session._cnn_inspector_factory = lambda _model: FakeInspector()
    session.cnn_inspector = None
    session.cnn_capability = {
        "layers": [{"id": "cnn.0"}, {"id": "cnn.2"}],
        "default_layer_id": "cnn.0",
        "unavailable_reason": None,
    }
    session.cnn_enabled = False
    session.cnn_layer_id = "cnn.0"
    session.cnn_interval = 1
    session.cnn_top_k = 12
    session.cnn_status = "off"
    session.cnn_error = None
    session.cnn_generation = 0
    session.cnn_last_computed_sequence = None
    session.last_transition = _attribution_transition()

    session.configure_cnn_inspection(
        enabled=True,
        layer_id="cnn.2",
        interval=3,
        top_k=8,
    )
    captured = session.last_transition
    session.configure_cnn_inspection(enabled=False)

    assert len(calls) == 1
    assert calls[0][1:] == ("cnn.2", 8, 1)
    assert captured.cnn_status == "available"
    assert captured.cnn_layer_id == "cnn.2"
    assert captured.cnn_inspection is inspection
    assert session.cnn_interval == 3
    assert session.cnn_top_k == 8
    assert session.last_transition.cnn_status == "off"
    assert session.last_transition.cnn_inspection is None
