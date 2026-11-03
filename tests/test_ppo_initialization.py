from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from gradlab.actor_critic_policy import SharedActorCriticPolicy
from gradlab.ppo import GradLabPPO
from gradlab.training.ppo import _normalize_gradlab_ppo, GradLabPPOBackend
from gradlab.training.ppo_initialization import (
    configure_critic_only,
    initialize_policy,
    normalize_initial_policy,
)
from gradlab.training.ppo_engine import (
    TensorRolloutBuffer,
    _CompiledPolicyCalls,
    _Precision,
    _ppo_update,
    _bootstrap_device_time_limits,
)


def spec():
    return dict(
        url="https://huggingface.co/owner/model/resolve/" + "a" * 40 + "/policy.pt",
        sha256="b" * 64,
        size_bytes=123,
        reset_value_head=True,
    )


def model(*, shared=False):
    return GradLabPPO(
        SharedActorCriticPolicy,
        gym.make("CartPole-v1"),
        n_steps=4,
        batch_size=4,
        n_epochs=2,
        learning_rate=1e-3,
        gamma=0.999,
        gae_lambda=1.0,
        device="cpu",
        policy_kwargs={
            "policy_model": dict(
                schema_version=2,
                encoder={"kind": "flatten"},
                fusion={"hidden_sizes": [8], "activation": "tanh"},
                normalize_images=False,
                orthogonal_init=True,
                share_features_extractor=shared,
            )
        },
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://huggingface.co/owner/model/resolve/main/policy.pt",
        "http://huggingface.co/owner/model/resolve/" + "a" * 40 + "/policy.pt",
        "https://other.example/owner/model/resolve/" + "a" * 40 + "/policy.pt",
        "https://huggingface.co@other.example/owner/model/resolve/" + "a" * 40 + "/policy.pt",
        "https://huggingface.co/owner/model/resolve/" + "a" * 40 + "/policy.pt?token=x",
        "https://huggingface.co/owner/model/resolve/" + "a" * 40 + "/../policy.pt",
    ],
)
def test_initial_policy_rejects_unpinned_or_credential_bearing_urls(url):
    with pytest.raises(ValueError):
        normalize_initial_policy({**spec(), "url": url}, label="init")


@pytest.mark.parametrize(
    "field,value",
    [
        ("size_bytes", True),
        ("size_bytes", 0),
        ("size_bytes", 2**31),
        ("sha256", "x" * 64),
        ("reset_value_head", "true"),
    ],
)
def test_initial_policy_requires_strict_hash_size_and_boolean(field, value):
    with pytest.raises(ValueError):
        normalize_initial_policy({**spec(), field: value}, label="init")


def test_initialization_loads_only_matching_finite_tensors_and_resets_critic(monkeypatch, tmp_path):
    import gradlab.model_sources as sources

    parent = model(shared=True)
    child = model()
    weights = parent.policy.state_dict()
    path = tmp_path / "policy.pt"
    torch.save(weights, path)
    calls = []

    def download(url, target, **kwargs):
        calls.append((url, target, kwargs))
        return path

    monkeypatch.setattr(sources, "_download_public_file", download)
    observations = torch.randn(16, 4)
    with torch.no_grad():
        before = parent.policy.get_distribution(observations).distribution.probs
    initialize_policy(child, spec(), root=tmp_path / "cache")
    with torch.no_grad():
        after = child.policy.get_distribution(observations).distribution.probs
    assert torch.equal(before, after)
    assert calls[0][2] == {"expected_sha256": "b" * 64, "expected_size": 123}
    assert all(torch.count_nonzero(p) == 0 for p in child.policy.value_net.parameters())
    assert not child.policy.optimizer.state
    bad = {**weights, "value_net.bias": torch.tensor([float("nan")])}
    torch.save(bad, path)
    with pytest.raises(ValueError, match="nonfinite"):
        initialize_policy(child, spec(), root=tmp_path / "cache")
    bad = {**weights, "value_net.bias": torch.zeros(2)}
    torch.save(bad, path)
    with pytest.raises(ValueError, match="incompatible"):
        initialize_policy(child, spec(), root=tmp_path / "cache")


def test_real_custom_ppo_critic_update_preserves_actor_then_allows_actor_training():
    torch.manual_seed(9)
    m = model()
    configure_critic_only(m, enabled=True)
    policy = m.policy
    observations = np.random.default_rng(2).normal(size=(2, 4)).astype("float32")
    actor_ids = {
        id(p)
        for module in (
            policy.pi_features_extractor,
            policy.mlp_extractor.policy_net,
            policy.action_net,
        )
        for p in module.parameters()
    }
    before = {name: p.detach().clone() for name, p in policy.named_parameters()}
    calls = _CompiledPolicyCalls(policy, torch.device("cpu"), compile_policy=False)
    buffer = TensorRolloutBuffer.allocate(
        observations, action_space=m.action_space, n_steps=4, n_envs=2, device=torch.device("cpu")
    )
    for step in range(4):
        obs = buffer.begin_step(observations, torch.tensor([step == 0, step == 0]))
        with torch.no_grad():
            actions, values, log_probs = calls.forward(obs)
        buffer.end_step(actions, np.ones(2, dtype="float32"), values, log_probs)
    buffer.finish(
        last_values=torch.zeros(2),
        dones=torch.ones(2, dtype=torch.bool),
        gamma=0.999,
        gae_lambda=1.0,
    )
    _ppo_update(
        m,
        buffer,
        calls=calls,
        precision=_Precision("fp32", torch.device("cpu")),
        progress_remaining=1.0,
        normalization_mode="global",
        advantage_context=None,
        ent_coef=0.0,
    )
    changed = []
    for name, p in policy.named_parameters():
        if id(p) in actor_ids:
            assert torch.equal(before[name], p), name
        elif not torch.equal(before[name], p):
            changed.append(name)
    assert changed, "Real critic optimizer step must change critic parameters"
    assert not policy.pi_features_extractor.training
    configure_critic_only(m, enabled=False)
    assert all(p.requires_grad for p in policy.parameters())


def test_shared_encoder_critic_mode_is_rejected():
    with pytest.raises(ValueError, match="separate"):
        configure_critic_only(model(shared=True), enabled=True)
    normalized = _normalize_gradlab_ppo({"critic_only": True}, label="config")
    with pytest.raises(ValueError, match="share_features_extractor"):
        GradLabPPOBackend().validate(
            {"policy_model": {"share_features_extractor": True}}, normalized
        )


def test_zero_bootstrap_contract_and_playback_need_no_terminal_critic():
    from gradlab.policy_bundle import _derive_critic_value_contract
    from gradlab.play_session import _truncation_bootstrap_value

    config = {
        "training_backend": {
            "id": "gradlab.ppo",
            "config": {"gamma": 0.999, "bootstrap_truncations": False},
        }
    }
    assert (
        _derive_critic_value_contract(config, policy_environment_hash="a" * 64)[
            "truncation_bootstrap"
        ]
        == "zero"
    )
    assert _truncation_bootstrap_value(
        policy_runtime=None,
        model=None,
        final_policy_obs=None,
        config=config,
        active_task_state=None,
        active_info_value=None,
    ) == (0.0, None)
    buffer = SimpleNamespace(final_observations=None, truncated=None)
    _bootstrap_device_time_limits(buffer, calls=None, precision=None, gamma=0.999, enabled=False)


def test_defaults_preserve_bootstrapping_and_joint_training():
    c = _normalize_gradlab_ppo({}, label="config")
    # Preserve the normalization/hash of existing artifact configurations.
    assert not {"initial_policy", "critic_only", "bootstrap_truncations"} & c.keys()
    assert (
        c.get("bootstrap_truncations", True) is True
        and c.get("critic_only", False) is False
        and c.get("initial_policy") is None
    )
