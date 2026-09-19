"""Failure injection at durable submission and real process ownership boundaries."""

from dataclasses import asdict
import os
import subprocess
import sys

import pytest

from gradlab.lifecycle_certification import CertificationFixture
from gradlab.monitor_backend import SameHostEvalBackend
from gradlab.monitor_config import MonitoringConfig
from gradlab.monitor_supervisor import MonitoringQueue


def test_lost_running_write_recovers_same_process_and_shutdown_without_r2(tmp_path, monkeypatch):
    fixture = CertificationFixture(tmp_path)
    prepared = fixture.prepare(run_number=91)
    owner = prepared.supervisor
    owner.evaluation_required = False
    fixture.record_checkpoint(prepared, step=100, kind="periodic")
    owner.active_iteration()
    owner.train_config["checkpoint_monitoring"] = asdict(MonitoringConfig())
    backend = SameHostEvalBackend(tmp_path / "workers", fixture.storage.models)
    queue = MonitoringQueue(owner, backend)
    # Substitute a long-running CPU process at the subprocess boundary; retain
    # the actual inherited flock, session and recoverable PID identity.
    popen = subprocess.Popen
    spawned = []

    def spawn(argv, **kwargs):
        if argv[0] == "/bin/ps":
            return popen(argv, **kwargs)
        code = """import os, time, json, pathlib
from gradlab.monitor_bootstrap import process_start
pathlib.Path(os.environ["TEST_PROCESS_FILE"]).write_text(json.dumps(dict(pid=os.getpid(),created=process_start(os.getpid()))))
time.sleep(60)
"""
        kwargs["env"]["TEST_PROCESS_FILE"] = str(
            __import__("pathlib").Path(argv[-1]).parent / "process.json"
        )
        process = popen([sys.executable, "-c", code], **kwargs)
        spawned.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", spawn)
    put = owner.authority.control.put_json

    def lost_write(key, value, **kwargs):
        if key.endswith("state.json") and value.get("status") == "running":
            raise OSError("lost state write")
        return put(key, value, **kwargs)

    monkeypatch.setattr(owner.authority.control, "put_json", lost_write)
    try:
        with pytest.raises(OSError, match="lost state"):
            queue.advance()
        assert len(spawned) == 1 and not backend.quiescent()
        monkeypatch.setattr(owner.authority.control, "put_json", put)
        # A fresh backend has no Popen handles. It must rediscover the old worker.
        recovered_backend = SameHostEvalBackend(backend.root, fixture.storage.models)
        recovered = MonitoringQueue(owner, recovered_backend)
        recovered.advance()
        assert len(spawned) == 1
        assert recovered.receipt["inventory"][0]["attempts"] == 1
        assert not recovered.receipt["workers_quiescent"]
        monkeypatch.setattr(
            owner.authority.control,
            "get_json_optional",
            lambda *_: (_ for _ in ()).throw(OSError("R2 down")),
        )
        recovered.close()
        spawned[0].wait(timeout=5)
        assert recovered_backend.quiescent()
    finally:
        backend.close()
        for process in spawned:
            process.wait(timeout=5)


def test_bootstrap_rejects_parent_lost_before_runtime_imports(tmp_path):
    request = tmp_path / "request.json"
    request.write_text("{}")
    with (tmp_path / "ownership.lock").open("a") as lock:
        env = dict(
            os.environ, GRADLAB_MONITOR_PARENT_PID="-1", GRADLAB_MONITOR_LOCK_FD=str(lock.fileno())
        )
        process = subprocess.Popen(
            [sys.executable, "-m", "gradlab.monitor_bootstrap", str(request)],
            env=env,
            pass_fds=(lock.fileno(),),
            start_new_session=True,
        )
        assert process.wait(timeout=10) != 0
    assert not (tmp_path / "process.json").exists()
    assert not (tmp_path / "result.json").exists()


def test_same_owner_recovers_committed_state_after_lost_ack(tmp_path, monkeypatch):
    from gradlab.eval_backend import EvalHandle, EvalPoll

    fixture = CertificationFixture(tmp_path)
    prepared = fixture.prepare(run_number=94)
    owner = prepared.supervisor
    owner.evaluation_required = False
    owner.eval_admission_closed = True
    fixture.record_checkpoint(prepared, step=100, kind="periodic")
    owner._publish_checkpoints()
    owner.train_config["checkpoint_monitoring"] = asdict(MonitoringConfig(enabled=True))

    class Backend:
        submissions = 0

        def submit(self, intent):
            self.submissions += 1
            return EvalHandle("cpu", "stable-handle")

        def poll(self, handle):
            assert handle.call_id == "stable-handle"
            return EvalPoll("running")

    backend = Backend()
    queue = MonitoringQueue(owner, backend)
    put = owner.authority.control.put_json
    lost = False

    def lost_ack(key, value, **kwargs):
        nonlocal lost
        result = put(key, value, **kwargs)
        if key.endswith("state.json") and value.get("status") == "running" and not lost:
            lost = True
            raise OSError("committed state but lost ACK")
        return result

    monkeypatch.setattr(owner.authority.control, "put_json", lost_ack)
    with pytest.raises(OSError, match="lost ACK"):
        queue.advance()
    assert queue.advance() is False
    assert backend.submissions == 1
    assert queue.receipt["inventory"][0]["attempts"] == 1

    # Immutable intent and owner-written state need no further remote reads.
    monkeypatch.setattr(owner.authority.control, "get_json_optional",
                        lambda key: (_ for _ in ()).throw(AssertionError(key)))
    queue.advance()
    assert backend.submissions == 1
