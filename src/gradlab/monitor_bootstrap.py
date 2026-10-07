"""Establish worker ownership before importing any training/runtime libraries."""

import ctypes
import json
import os
from pathlib import Path
import signal
import sys
import subprocess


def process_start(pid):
    """Disambiguate a PID from later reuse without importing runtime dependencies."""
    if sys.platform == "linux":
        try:
            # comm may contain spaces and parentheses; field 22 is starttime.
            return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
        except FileNotFoundError:
            return None
    result = subprocess.run(
        ["/bin/ps", "-p", str(pid), "-o", "lstart="], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def main():
    request = Path(sys.argv[1])
    expected_parent = int(os.environ["GRADLAB_MONITOR_PARENT_PID"])
    # The inherited descriptor keeps the supervisor-acquired ownership lock held
    # across spawn and imports. Children such as ffmpeg must not inherit it.
    descriptor = int(os.environ["GRADLAB_MONITOR_LOCK_FD"])
    os.set_inheritable(descriptor, False)

    def stop_group(_signal, _frame):
        os.killpg(os.getpgrp(), signal.SIGKILL)

    signal.signal(signal.SIGTERM, stop_group)
    if sys.platform == "linux":
        if ctypes.CDLL(None).prctl(1, signal.SIGTERM) != 0:
            stop_group(None, None)
    if os.getppid() != expected_parent:
        stop_group(None, None)
    identity = request.parent / "process.json"
    temporary = identity.with_suffix(".tmp")
    temporary.write_text(json.dumps(dict(pid=os.getpid(), created=process_start(os.getpid()))))
    temporary.replace(identity)
    from gradlab.monitor_worker import main as work

    return work()


if __name__ == "__main__":
    raise SystemExit(main())
