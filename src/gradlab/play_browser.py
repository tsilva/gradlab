"""Own a browser process without sharing the user's normal browser profile."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path


def browser_executable() -> str:
    if sys.platform == "darwin":
        for root in (Path("/Applications"), Path.home() / "Applications"):
            for name in ("Google Chrome", "Chromium", "Microsoft Edge", "Brave Browser"):
                path = root / f"{name}.app" / "Contents" / "MacOS" / name
                if path.is_file() and os.access(path, os.X_OK):
                    return str(path)
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        if executable := shutil.which(name):
            return executable
    raise RuntimeError(
        "Playback needs an installed Chrome or Chromium browser. "
        "Install one, or use --no-open and open the printed URL manually."
    )


class PlaybackBrowser:
    """A unique profile prevents reuse of any existing browser process."""

    def __init__(self) -> None:
        self.process: subprocess.Popen | None = None
        self.profile: tempfile.TemporaryDirectory | None = None

    def open(self, url: str) -> None:
        executable = browser_executable()
        self.profile = tempfile.TemporaryDirectory(prefix="gradlab-browser-")
        try:
            self.process = subprocess.Popen(
                [
                    executable,
                    f"--user-data-dir={self.profile.name}",
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--disable-background-mode",
                    "--new-window",
                    url,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        process = self.process
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Only the process group created for this isolated instance.
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
            self.process = None
        if self.profile is not None:
            self.profile.cleanup()
            self.profile = None
