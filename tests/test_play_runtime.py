from __future__ import annotations

from gradlab.environment_fields import EnvConfig
from gradlab.play_runtime import PlaybackLoader, _with_playback_device_override


def test_with_playback_device_override_enforces_cpu_for_gradoom() -> None:
    config = EnvConfig(
        env_provider="env-gradoom-turbo-torch",
        game="VizdoomDeathmatch-v1",
        env_args={},
    )

    patched = _with_playback_device_override(config)

    assert patched == {"device": "cpu"}
    assert config.env_args == {}


def test_with_playback_device_override_only_applies_to_gradoom() -> None:
    config = EnvConfig(
        env_provider="env-vizdoom-turbo",
        game="VizdoomDeathmatch-v1",
        env_args={},
    )

    patched = _with_playback_device_override(config)

    assert patched == {}
    assert config.env_args == {}


def test_playback_contract_reports_training_and_active_frame_skip(monkeypatch) -> None:
    training_environment = {
        "env_provider": "gymnasium",
        "game": "CartPole-v1",
        "frame_skip": 4,
        "task": {},
    }
    evaluation_environment = {
        **training_environment,
        "frame_skip": 2,
    }
    source = type(
        "Source",
        (),
        {
            "artifact_ref": "fixture",
            "artifact_name": "fixture",
            "model_path": __file__,
            "bundle": type(
                "Bundle",
                (),
                {"recipe": {"recipe": {"eval": {}}}},
            )(),
        },
    )()
    monkeypatch.setattr("gradlab.play_runtime.resolve_model_source", lambda *_a, **_k: source)
    monkeypatch.setattr(
        "gradlab.play_runtime.playback_contract",
        lambda _recipe, *, mode="training": {
            "environment": (
                training_environment if mode == "training" else evaluation_environment
            ),
            "training_policy_environment_hash": "training-hash",
            "matches_training": mode == "training",
            "asset": None,
        },
    )
    monkeypatch.setattr(
        "gradlab.play_runtime.playback_contract_audit",
        lambda _recipe: {
            "requested_policy_override_paths": [],
            "mismatch_paths": ["environment.frame_skip"],
        },
    )
    monkeypatch.setattr("gradlab.play_runtime.critic_value_contract", lambda _recipe: None)
    monkeypatch.setattr("gradlab.play_runtime.resolve_env_config", lambda config: config)
    monkeypatch.setattr("gradlab.env_identity.policy_environment_hash", lambda _value: "eval-hash")
    monkeypatch.setattr("gradlab.play_runtime.assert_provider_runtime_available", lambda *_a, **_k: None)
    monkeypatch.setattr("gradlab.play_runtime.stage_model_input", lambda *_a, **_k: object())
    monkeypatch.setattr("gradlab.play_runtime.resolve_shared_playback_rom_binding", lambda **_k: None)
    loader = PlaybackLoader(
        type(
            "Args",
            (),
            {
                "public_model_root": ".",
                "hf_model_root": ".",
                "hf_revision": None,
                "public_models_base_url": "https://models.example",
                "fps": 0,
                "env_provider": None,
                "continuous_play": False,
                "seed": 10_000,
                "rom_path": None,
            },
        )(),
        argv=[],
        explicit_seed=True,
    )

    candidate = loader.prepare(
        type(
            "Spec",
            (),
            {
                "kind": "local",
                "value": "fixture",
                "run_id": "",
                "checkpoint_id": "",
                "contract_mode": "evaluation",
                "reward_clip_override": None,
                "seed": None,
            },
        )(),
        lambda *_args: None,
    )

    assert candidate.contract_details["frame_skip"] == {
        "training": 4,
        "playback": 2,
    }
