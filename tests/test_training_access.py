"""Domain dispatch does not create sessions or perform source-credential reads."""
import os
from unittest.mock import patch

import pytest

from gradlab import training_access
from gradlab.training_vault import TrainingVaultCommandError


@pytest.mark.parametrize("text,category", [
    ("HTTP 401 Unauthorized", "authentication_failed"),
    ("HTTP 403 Client Error", "authentication_failed"),
    ("407 Proxy Authentication Required", "proxy_rejected"),
    ("certificate verify failed", "tls_failed"),
    ("ConnectionRefusedError", "connection_failed"),
    ("Unexpected upstream status", "nativecall_error"),
])
def test_native_failure_categories_never_include_provider_details(text, category):
    private = "http://x-agent-vault:agv_syntheticPrivate@127.0.0.1:28383/private-body"
    error = TrainingVaultCommandError(private, text + private)
    assert error.category == category
    assert str(error) == "Training inventory request failed: " + category
    assert "agv_" not in repr(error) and "private-body" not in repr(error)


def test_inventory_emits_only_fixed_error_category(capsys):
    error = TrainingVaultCommandError("private-body", "401 Unauthorized")
    with patch('gradlab.training_vault.validate_session'), patch.object(
        training_access, 'inventory', side_effect=error
    ), pytest.raises(SystemExit) as caught:
        training_access.main(['inventory'])
    assert caught.value.code == 1
    assert capsys.readouterr().out == '{"probe_error": "authentication_failed"}\n'


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


def test_retry_dispatch_matches_the_actual_cli_parser():
    from gradlab.experiment_cli import build_parser, cmd_retry

    run_id = 'gradlab-' + 'a' * 32
    def parse_dispatch(*args, **kwargs):
        parsed = build_parser().parse_args(training_access.sys.argv[2:])
        assert parsed.func is cmd_retry and parsed.run_id == run_id
        assert not parsed.repair_runtime
    with patch('gradlab.training_vault.validate_session'), patch.object(
        training_access.runpy, 'run_module', side_effect=parse_dispatch
    ) as run:
        with patch.dict(os.environ, {}, clear=True), patch.object(training_access.sys, 'argv', []):
            training_access.main(['experiment', 'retry', '--run', run_id])
            run.assert_called_once_with('gradlab.main', run_name='__main__')
