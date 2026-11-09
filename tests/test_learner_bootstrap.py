from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest


def _wait_for(predicate, *, timeout=8.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("owned subprocess did not reach its bounded lifecycle state")


def _identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    return fields[0], fields[19]


def _active(pid, started):
    current = _identity(pid)
    return current is not None and current[0] != "Z" and current[1] == started


def _supervisor(tmp_path, *, cooperative, native_block=False, shutdown_block=False, module=False, multiprocessing=None):
    descendant = """
import signal,time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
while True: time.sleep(0.05)
"""
    learner = """
import json,os,signal,subprocess,sys,time
from pathlib import Path
directory=Path(sys.argv[1])
def stop(_signal,_frame):
    (directory/'training-result.json').write_text(json.dumps({'status':'stopped'}))
    raise SystemExit(0)
signal.signal(signal.SIGUSR1,stop)
if not COOPERATIVE: signal.signal(signal.SIGTERM,signal.SIG_IGN)
child=None
if not COOPERATIVE:
    child=subprocess.Popen([sys.executable,'-c',DESCENDANT])
(directory/'learner-ready.json').write_text(json.dumps({'learner':os.getpid(),'descendant':None if child is None else child.pid}))
def native_wait():
    import ctypes
    libc=ctypes.CDLL(None)
    mutex=ctypes.create_string_buffer(128)
    assert libc.pthread_mutex_init(ctypes.byref(mutex),None)==0
    assert libc.pthread_mutex_lock(ctypes.byref(mutex))==0
    libc.pthread_mutex_lock(ctypes.byref(mutex))
if NATIVE_BLOCK:
    if SHUTDOWN_BLOCK:
        import atexit
        atexit.register(native_wait)
        raise SystemExit(0)
    native_wait()
while True: time.sleep(0.05)
""".replace("COOPERATIVE", repr(cooperative)).replace("DESCENDANT", repr(descendant)).replace("NATIVE_BLOCK", repr(native_block)).replace("SHUTDOWN_BLOCK", repr(shutdown_block))
    if multiprocessing:
        original = "child=subprocess.Popen([sys.executable,'-c'," + repr(descendant) + "])"
        assert learner.count(original) == 1
        learner = learner.replace(
            original,
            "import multiprocessing\n    child=multiprocessing.get_context(" + repr(multiprocessing) + ").Process(target=idle)\n    child.start()",
        )
    environment = os.environ.copy()
    if module:
        source = "def idle():\n" + textwrap.indent(descendant, "    ")
        source += "\ndef work():\n" + textwrap.indent(learner, "    ")
        source += "\nif __name__=='__main__': work()\n"
        (tmp_path / "learner_guard_fixture.py").write_text(source)
        environment["PYTHONPATH"] = str(tmp_path) + os.pathsep + environment.get("PYTHONPATH", "")
    learner_command = ["-m", "learner_guard_fixture"] if module else ["-c", learner]
    supervisor = """
import os,signal,sys,time
from pathlib import Path
from gradlab.supervisor_runtime import SupervisorRuntime
directory=Path(sys.argv[1]);runtime=SupervisorRuntime()
learner=runtime.start_learner([sys.executable,*COMMAND,str(directory)],log_path=directory/'learner.log',environment=os.environ)
def stop(_signal,_frame):
    runtime.request_learner_stop(learner)
    status=learner.wait(timeout=5)
    assert not runtime.learner_group_alive(learner)
    learner._gradlab_log.close()
    raise SystemExit(status)
signal.signal(signal.SIGTERM,stop)
while True: time.sleep(0.05)
""".replace("COMMAND", repr(learner_command))
    process = subprocess.Popen(
        [sys.executable, "-c", supervisor, str(tmp_path)],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=environment,
    )
    return process


def _check_learner_group(tmp_path, cooperative, *, native_block=False, shutdown_block=False, module=False, multiprocessing=None):
    parent = _supervisor(tmp_path, cooperative=cooperative, native_block=native_block, shutdown_block=shutdown_block, module=module, multiprocessing=multiprocessing)
    owned = {}
    try:
        ready = tmp_path / "learner-ready.json"
        _wait_for(ready.exists)
        values = json.loads(ready.read_text())
        owned = {pid: _identity(pid)[1] for pid in values.values() if pid is not None}
        if native_block:
            _wait_for(lambda: "futex" in Path(f"/proc/{values['learner']}/wchan").read_text())
        os.kill(parent.pid, signal.SIGTERM if cooperative else signal.SIGKILL)
        parent.wait(timeout=8)
        _wait_for(lambda: all(not _active(pid, started) for pid, started in owned.items()))
        result = tmp_path / "training-result.json"
        if cooperative:
            assert parent.returncode == 0
            assert json.loads(result.read_text()) == {"status": "stopped"}
        else:
            assert parent.returncode == -signal.SIGKILL
            assert not result.exists()
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=8)
        # Bound cleanup even when exercising the unguarded regression baseline.
        for pid, started in owned.items():
            if _active(pid, started):
                os.kill(pid, signal.SIGKILL)


def _check_parent_arm_race(tmp_path):
    marker = tmp_path / "unexpected-learner-start"
    environment = {**os.environ, "GRADLAB_LEARNER_PARENT_PID": "2147483647"}
    process = subprocess.Popen(
        [sys.executable, "-m", "gradlab.learner_bootstrap", "-c", f"open({str(marker)!r},'w').close()"],
        env=environment,
        start_new_session=True,
    )
    assert process.wait(timeout=8) == -signal.SIGKILL
    assert not marker.exists()


@unittest.skipUnless(sys.platform == "linux", "Linux parent-death ownership")
class LearnerBootstrapTests(unittest.TestCase):
    def test_parent_sigkill_with_ignored_sigterm(self):
        with tempfile.TemporaryDirectory() as directory:
            _check_learner_group(Path(directory), False)

    def test_graceful_stop_finalizes(self):
        with tempfile.TemporaryDirectory() as directory:
            _check_learner_group(Path(directory), True)

    def test_native_block_does_not_block_parent_death_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            _check_learner_group(Path(directory), False, native_block=True)

    def test_native_atexit_block_does_not_outlive_supervisor(self):
        with tempfile.TemporaryDirectory() as directory:
            _check_learner_group(Path(directory), False, native_block=True, shutdown_block=True)

    def test_python_module_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            _check_learner_group(Path(directory), True, module=True)

    def test_multiprocessing_descendants_are_stopped(self):
        for method in ("fork", "spawn"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                _check_learner_group(Path(directory), False, module=True, multiprocessing=method)

    def test_parent_loss_before_bootstrap_is_armed(self):
        with tempfile.TemporaryDirectory() as directory:
            _check_parent_arm_race(Path(directory))
