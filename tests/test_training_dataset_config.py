import pytest

from gradlab.train_config import validate_and_normalize_train_config


def config(**capture):
    return {
        "game": "Breakout-Atari2600-v0",
        "env_provider": "env-breakoutatari2600-turbo-native",
        "training_backend": {"id": "sb3.ppo", "config": {}},
        "timesteps": 10000,
        "trajectory_collection": {"enabled": True, **capture},
    }


def test_collection_resolves_finite_defaults_without_huggingface_target():
    resolved = validate_and_normalize_train_config(config(), validate_backend_config=False)
    capture = resolved["trajectory_collection"]
    assert capture["contribution_bytes"] == 10 * 1024**3
    assert capture["drain_seconds"] > 0
    assert capture["memory_bytes"] > capture["chunk_bytes"]
    assert capture["disk_bytes"] > capture["chunk_bytes"]
    assert not any("hf" in key for key in capture)


@pytest.mark.parametrize(
    "field,value",
    [
        ("contribution_bytes", 0),
        ("memory_bytes", 1),
        ("disk_bytes", 1),
        ("drain_seconds", float("inf")),
        ("sample_probability", -1),
        ("sample_probability", 0),
        ("enabled", "yes"),
        ("hf_repo", "user/data"),
    ],
)
def test_invalid_collection_is_rejected(field, value):
    with pytest.raises(ValueError):
        validate_and_normalize_train_config(config(**{field: value}), validate_backend_config=False)


def test_unverified_provider_is_rejected():
    value = config()
    value["env_provider"] = "env-stableretro-turbo"
    with pytest.raises(ValueError, match="native Breakout"):
        validate_and_normalize_train_config(value, validate_backend_config=False)
