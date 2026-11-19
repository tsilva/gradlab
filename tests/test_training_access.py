"""Domain dispatch does not create sessions or perform source-credential reads."""
import json
import os
from types import SimpleNamespace
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
        with pytest.raises(training_access.LaunchError):
            training_access.main(['autoresearch', 'reserve-wave'])
        with pytest.raises(training_access.LaunchError):
            training_access.main(['reservations', '--fleet', 'synthetic'])


def test_reservations_uses_the_study_fleet_and_coordinator(tmp_path, capsys):
    from gradlab.autoresearch import SCHEMA_VERSION

    path = tmp_path / "study.json"
    path.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "repo_root": str(training_access.ROOT),
        "policy": {"compute_target": "synthetic-fleet", "dstack_coordinator_id": "selected"},
    }))
    before = path.read_bytes()
    fleet = SimpleNamespace(name="synthetic-fleet")
    backend = SimpleNamespace(active_training_reservations=lambda name: 2 if name == fleet.name else 0)
    with patch("gradlab.training_vault.validate_session") as validate, patch.object(
        training_access, "_fleet_backend", return_value=(fleet, backend)
    ) as route:
        with patch.dict(os.environ, {}, clear=True):
            training_access.main(["reservations", "--study", str(path)])
        validate.assert_called_once_with()
        route.assert_called_once_with("synthetic-fleet", "selected")

    assert json.loads(capsys.readouterr().out) == {"fleet": "synthetic-fleet", "active_reservations": 2}
    assert path.read_bytes() == before


@pytest.mark.parametrize("operation", ["reconcile", "collect-training-evidence"])
def test_training_dispatch_parses_evidence_operations(operation):
    from gradlab.autoresearch import build_parser

    options = ["--study", "synthetic-study"]
    if operation == "collect-training-evidence":
        options.extend(["--run-id", "gradlab-" + "a" * 32])

    def parse_dispatch(*args, **kwargs):
        assert training_access.sys.argv[1] == "autoresearch"
        parsed = build_parser().parse_args(training_access.sys.argv[2:])
        assert parsed.study == "synthetic-study"
        assert parsed.handler.__name__ == "command_" + operation.replace("-", "_")

    with patch('gradlab.training_vault.validate_session') as validate, patch.object(
        training_access.runpy, 'run_module', side_effect=parse_dispatch
    ) as run:
        with patch.dict(os.environ, {}, clear=True), patch.object(training_access.sys, 'argv', []):
            training_access.main(["autoresearch", operation, *options])
            validate.assert_called_once_with()
            run.assert_called_once_with('gradlab.main', run_name='__main__')


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


def test_reconcile_dispatch_preserves_the_actual_cli_evidence_and_stop_reason():
    from gradlab.experiment_cli import build_parser, cmd_reconcile

    run_id = 'gradlab-' + 'a' * 32
    evidence = 'f' * 64

    def parse_dispatch(*args, **kwargs):
        parsed = build_parser().parse_args(training_access.sys.argv[2:])
        assert parsed.func is cmd_reconcile and parsed.run_id == run_id
        assert parsed.stop_reason == 'forced_cancel_before_drain'
        assert parsed.evidence_sha256 == [evidence] and parsed.json

    with patch('gradlab.training_vault.validate_session') as validate, patch.object(
        training_access.runpy, 'run_module', side_effect=parse_dispatch
    ) as run:
        with patch.dict(os.environ, {}, clear=True), patch.object(training_access.sys, 'argv', []):
            training_access.main(['experiment', 'reconcile', '--run', run_id,
                                  '--stop-reason', 'forced_cancel_before_drain',
                                  '--evidence-sha256', evidence, '--json'])
            validate.assert_called_once_with()
            run.assert_called_once_with('gradlab.main', run_name='__main__')
