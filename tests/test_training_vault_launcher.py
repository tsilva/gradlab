"""Client exit must not revoke a worker after uncertain durable admission."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/training_vault.py"
    spec = importlib.util.spec_from_file_location("training_vault_launcher_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ca = tmp_path / "public-ca.pem"
    ca.write_text("synthetic public CA")
    profile = {"ca_file": str(ca), "ca_sha256": "synthetic-ca-digest", "ca_fingerprint": "synthetic-fingerprint",
               "proxy_local": "127.0.0.1:28383", "proxy_remote": "172.17.0.1:28383", "coordinator_id": "b3",
               "worker_bundle": "local-gradlab-training-worker", "operator_bundle": "local-gradlab-training-operator",
               "python": "/synthetic/python", "cli": "/synthetic/infisical"}
    monkeypatch.setattr(module, "profile", lambda: profile)
    monkeypatch.setattr(module, "forward", lambda _: None)
    worker = {"id": "00000000-0000-4000-8000-000000000001", "token": "agv_syntheticWorkerCapability",
              "expiresAt": "2099-01-01T00:00:00Z"}
    native = Mock(return_value=json.dumps(worker))
    monkeypatch.setattr(module, "native", native)
    return module, profile, native


def test_failed_child_without_admission_ack_keeps_worker_session(launcher, monkeypatch):
    module, _, native = launcher
    process = Mock(stdout=iter(()), returncode=1)
    monkeypatch.setattr(module.subprocess, "Popen", Mock(return_value=process))
    with pytest.raises(SystemExit):
        module.main(["experiment", "launch"])
    assert [call.args[1][:3] for call in native.call_args_list] == [["agent-vault", "sessions", "create"]]


def test_interrupted_child_start_keeps_worker_session(launcher, monkeypatch):
    module, _, native = launcher
    monkeypatch.setattr(module.subprocess, "Popen", Mock(side_effect=KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        module.main(["experiment", "launch"])
    assert native.call_count == 1


def test_failure_before_child_start_revokes_unused_worker_session(launcher):
    module, profile, native = launcher
    profile.pop("python")
    with pytest.raises(KeyError):
        module.main(["experiment", "launch"])
    assert native.call_args.args[1] == ["agent-vault", "sessions", "revoke", "00000000-0000-4000-8000-000000000001"]
