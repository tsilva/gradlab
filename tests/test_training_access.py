"""Domain dispatch does not create sessions or perform source-credential reads."""
import os
from unittest.mock import patch

import pytest

from gradlab import training_access


def test_unreviewed_operation_fails_before_transport_validation():
    with patch('gradlab.training_vault.validate_session', side_effect=AssertionError('not reached')):
        with pytest.raises(training_access.LaunchError):
            training_access.main(['unreviewed'])


def test_follow_requires_the_separate_long_lived_profile():
    with patch.dict(os.environ, {}, clear=True), pytest.raises(training_access.LaunchError):
        training_access.main(['experiment', 'follow', '--run', 'synthetic'])


def test_launch_keeps_json_admission_and_command_scoped_https_auth():
    with patch('gradlab.training_vault.validate_session'), patch.object(training_access.runpy, 'run_module') as run:
        with patch.dict(os.environ, {}, clear=True), patch.object(training_access.sys, 'argv', []):
            training_access.main(['experiment', 'launch', '--recipe-file', 'synthetic'])
            assert training_access.sys.argv[-1] == '--json'
            assert os.environ['GIT_CONFIG_KEY_0'] == 'url.https://github.com/.insteadOf'
            assert os.environ['GIT_CONFIG_VALUE_2'] == '!gh auth git-credential'
            run.assert_called_once_with('gradlab.main', run_name='__main__')
