"""Keep the strict native recording gate aligned with the verified dependency pin."""

from pathlib import Path
import tomllib
from unittest.mock import patch

import pytest

from gradlab import recording_contract
from gradlab.env import make_eval_vec_env, resolve_env_config
from gradlab.env_config import env_config_from_mapping
from gradlab.recipe_documents import compose_train_document


@pytest.fixture
def native_runtime():
    goal = Path('experiments/goals/Breakout-Atari2600-v0/FirstWall')
    train = compose_train_document(goal / '_goal.yaml', goal / 'recipes/ppo.yaml')['train_config']
    config = resolve_env_config(env_config_from_mapping(train))
    env = make_eval_vec_env(config, n_envs=1, seed=17)
    try:
        yield env.runtime
    finally:
        env.close()


def test_recording_contract_accepts_current_verified_native_provider(native_runtime):
    contract = recording_contract.verify_recording_provider(native_runtime)
    assert contract['version'] == recording_contract.PROVIDER_VERSION
    assert f"/v{contract['version']}/" in contract['source']
    assert contract['rgb_shape'] == [210, 160, 3]
    assert contract['native_action_meanings'] == ['noop', 'button', 'right', 'left']


@pytest.mark.parametrize('installed', ['0.5.13', '0.5.14', '0.5.16'])
def test_recording_contract_still_rejects_unverified_versions(native_runtime, installed):
    with patch.object(recording_contract.importlib.metadata, 'version', return_value=installed):
        with pytest.raises(ValueError, match='exact verified native Breakout provider'):
            recording_contract.verify_recording_provider(native_runtime)


def test_recording_contract_matches_all_native_dependency_pins():
    with (Path(__file__).resolve().parents[1] / 'pyproject.toml').open('rb') as handle:
        project = tomllib.load(handle)
    dependencies = project['project']['dependencies'] + project['dependency-groups']['train-runtime']
    pins = {
        dependency.split('==', 1)[1].split(';', 1)[0]
        for dependency in dependencies
        if dependency.startswith(recording_contract.PROVIDER + '==')
    }
    assert pins == {recording_contract.PROVIDER_VERSION}
