"""Bind a detached Linux learner group to its supervisor before runtime imports."""

import ctypes
import os
import runpy
import select
import signal
import sys


def _stop_group() -> None:
    os.killpg(os.getpgrp(), signal.SIGKILL)


def _guard_supervisor(expected_parent: int) -> int | None:
    if os.getpgrp() != os.getpid():
        raise RuntimeError("learner bootstrap requires its own process group")
    if os.getppid() != expected_parent:
        _stop_group()
    if sys.platform != "linux":
        return

    # This bootstrap is still single-threaded. Keep the sentinel independent of
    # learner imports, signal dispositions, and potentially blocked native calls.
    supervisor = os.pidfd_open(expected_parent)
    ready, acknowledge = os.pipe2(os.O_CLOEXEC)
    learner_pid = os.getpid()
    try:
        sentinel = os.fork()
    except BaseException:
        os.close(supervisor)
        os.close(ready)
        os.close(acknowledge)
        raise
    if sentinel == 0:
        os.close(ready)

        def stop_owned_group() -> None:
            os.killpg(learner_pid, signal.SIGKILL)

        try:
            # Keep even a transient sentinel zombie out of the learner group
            # used by the supervisor's post-exit quiescence checks.
            os.setsid()
            # Do not retain a sentinel after a normal learner exit. The
            # supervisor still owns ordinary post-exit group quiescence.
            libc = ctypes.CDLL(None, use_errno=True)
            if libc.prctl(1, int(signal.SIGKILL), 0, 0, 0) != 0:
                stop_owned_group()
            if os.getppid() != learner_pid:
                os._exit(0)
            watcher = select.poll()
            watcher.register(supervisor, select.POLLIN)
            os.write(acknowledge, b"1")
            os.close(acknowledge)
            watcher.poll()
            stop_owned_group()
        except BaseException:
            stop_owned_group()
        finally:
            os._exit(1)
    os.close(supervisor)
    os.close(acknowledge)
    try:
        if os.read(ready, 1) != b"1":
            _stop_group()
    finally:
        os.close(ready)
    if os.getppid() != expected_parent:
        _stop_group()
    return sentinel


def main() -> None:
    expected_parent = int(os.environ.pop("GRADLAB_LEARNER_PARENT_PID"))
    arguments = sys.argv[1:]
    if expected_parent <= 0 or len(arguments) < 2 or arguments[0] not in {"-m", "-c"}:
        raise ValueError("learner bootstrap requires a parent and Python -m/-c command")
    # Leave the sentinel armed through interpreter/atexit shutdown, which can
    # itself block in native code. Its kernel parent-death signal ends it only
    # when this learner process actually exits.
    _guard_supervisor(expected_parent)
    if arguments[0] == "-m":
        sys.argv = arguments[1:]
        runpy.run_module(arguments[1], run_name="__main__", alter_sys=True)
    else:
        sys.argv = ["-c", *arguments[2:]]
        exec(compile(arguments[1], "<string>", "exec"), {"__name__": "__main__"})


if __name__ == "__main__":
    main()
