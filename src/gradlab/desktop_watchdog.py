"""Standalone stdlib helper: an EOF on the parent's pipe closes its native viewer.

Run by absolute filename so it also works from wheels and editable installations.
The pipe closes on normal shutdown, crashes, and SIGKILL, without PID polling.
"""

from __future__ import annotations

import os
import argparse
import shutil
import select
import signal
import subprocess
import sys
import time


def supervise(command: list[str]) -> int:
    # Own the child group separately: only the watchdog survives to reap it.
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, start_new_session=True)
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while process.poll() is None and not stopping:
            readable, _, _ = select.select([sys.stdin.buffer], [], [], 0.2)
            if readable and not os.read(sys.stdin.fileno(), 1):
                break
        return process.returncode or 0
    finally:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 3
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        # Reap leftover children even if the main native process exited first.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleanup-directory")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        result = supervise(args.command)
    finally:
        if args.cleanup_directory:
            shutil.rmtree(args.cleanup_directory, ignore_errors=True)
    raise SystemExit(result)
