import asyncio
import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from gradlab import desktop_runtime, desktop_watchdog
from gradlab.play_browser import DesktopWindow, PlaybackBrowser


def test_window_launch_has_no_page_native_capability_and_owned_pipe(tmp_path):
    process = Mock()
    process.stdin.closed = False
    with patch("gradlab.play_browser.subprocess.Popen", return_value=process) as popen:
        window = DesktopWindow(Path("/viewer"), "http://127.0.0.1:1234/", "GradLab")
        root = Path(window.profile.name)
        config = json.loads((root / "neutralino.config.json").read_text())
        assert config["modes"]["window"]["injectGlobals"] is False
        assert config["modes"]["window"]["injectClientLibrary"] is False
        assert config["nativeAllowList"] == ["window.show", "window.unminimize", "window.focus"]
        assert (root / "icon.png").read_bytes() == (
            desktop_runtime.ASSETS / "player.png"
        ).read_bytes()
        assert popen.call_args.kwargs["stdin"] == subprocess.PIPE
        assert popen.call_args.kwargs["start_new_session"] is True
        window.close()
        process.stdin.close.assert_called_once()
        process.wait.assert_called_once()
        assert not root.exists()


def test_window_launch_failure_cleans_resources():
    roots = []

    def fail(command, **kwargs):
        roots.append(Path(command[-1].removeprefix("--path=")))
        raise OSError("failed")

    with patch("gradlab.play_browser.subprocess.Popen", side_effect=fail):
        with pytest.raises(OSError, match="failed"):
            DesktopWindow(Path("/viewer"), "http://127.0.0.1:1234/", "GradLab")
    assert not roots[0].exists()


def test_open_focuses_existing_window_and_reopens_closed_window():
    async def scenario():
        first, second = Mock(), Mock()
        for window in (first, second):
            window.focus = AsyncMock()
            window.process.poll.return_value = None
        with (
            patch("gradlab.play_browser.viewer_executable", return_value=Path("/viewer")),
            patch("gradlab.play_browser.DesktopWindow", side_effect=[first, second]) as factory,
        ):
            browser = PlaybackBrowser()
            player = Mock()
            player.process.poll.return_value = None
            browser.windows["main"] = player
            await browser.open("http://127.0.0.1:1/?workspace=paired#token=secret", "stats")
            await browser.open("http://127.0.0.1:1/", "stats")
            assert factory.call_count == 1
            assert first.focus.await_count == 2
            url = factory.call_args.args[1]
            assert f"desktop={browser.workspace_id}" in url and url.endswith("#token=secret")
            first.process.poll.return_value = 0
            await browser.open("http://127.0.0.1:1/", "stats")
            assert factory.call_count == 2
            first.close.assert_called_once()
            player.close.assert_not_called()
            browser.close()
            second.close.assert_called_once()

    asyncio.run(scenario())


@pytest.mark.parametrize("window_name", ["main", "stats"])
def test_companion_requests_cannot_reopen_a_closed_player(window_name):
    async def scenario():
        browser = PlaybackBrowser()
        main, stats = Mock(), Mock()
        main.process.poll.return_value = 0
        stats.process.poll.return_value = None
        browser.windows = {"main": main, "stats": stats}
        with patch("gradlab.play_browser.DesktopWindow") as factory:
            with pytest.raises(RuntimeError, match="Player.*closed"):
                await browser.open("http://127.0.0.1:1/", window_name)
            factory.assert_not_called()
            stats.focus.assert_not_called()
        await asyncio.wait_for(browser.wait_closed(), 1)
        browser.close()
        main.close.assert_called_once()
        stats.close.assert_called_once()
        with pytest.raises(RuntimeError, match="Player.*closed"):
            await browser.open("http://127.0.0.1:1/", window_name)

    asyncio.run(scenario())


def test_player_exiting_during_focus_still_ends_the_session():
    async def scenario():
        browser = PlaybackBrowser()
        player, stats = Mock(), Mock()
        player.process.poll.return_value = None
        stats.process.poll.return_value = None
        player.focus = AsyncMock(side_effect=RuntimeError("viewer disconnected"))
        browser.windows = {"main": player, "stats": stats}
        with pytest.raises(RuntimeError, match="viewer disconnected"):
            await browser.open("http://127.0.0.1:1/", "main")
        await asyncio.wait_for(browser.wait_closed(), 1)
        browser.close()
        player.close.assert_called_once()
        stats.close.assert_called_once()

    asyncio.run(scenario())


def test_close_listener_clearing_player_process_ends_session_without_error():
    async def scenario():
        browser = PlaybackBrowser()
        player = Mock()
        player.process = None  # DesktopWindow.close() has already reaped it.
        browser.windows["main"] = player

        await asyncio.wait_for(browser.wait_closed(), 1)
        with pytest.raises(RuntimeError, match="Player.*closed"):
            await browser.open("http://127.0.0.1:1/", "stats")
        browser.close()
        player.close.assert_called_once()

    asyncio.run(scenario())


def test_closed_stats_window_can_reopen_after_listener_clears_process():
    async def scenario():
        browser = PlaybackBrowser()
        player, stats, replacement = Mock(), Mock(), Mock()
        player.process.poll.return_value = None
        stats.process = None
        replacement.focus = AsyncMock()
        browser.windows = {"main": player, "stats": stats}
        with (
            patch("gradlab.play_browser.viewer_executable", return_value=Path("/viewer")),
            patch("gradlab.play_browser.DesktopWindow", return_value=replacement),
        ):
            await browser.open("http://127.0.0.1:1/", "stats")
        stats.close.assert_called_once()
        replacement.focus.assert_awaited_once()
        browser.close()

    asyncio.run(scenario())


def test_runtime_hash_verification_and_cached_launch(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    data = b"example executable"
    digest = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(
        desktop_runtime,
        "BINARIES",
        {(sys.platform, desktop_runtime.platform.machine().lower()): ("test", digest)},
    )
    with patch.object(desktop_runtime, "_download_binary", return_value=data) as download:
        executable = desktop_runtime.runtime_executable()
        assert executable.read_bytes() == data
        assert desktop_runtime.runtime_executable() == executable
        download.assert_called_once()
        executable.write_bytes(b"tampered")
        with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
            desktop_runtime.runtime_executable()


def test_download_rejects_modified_archive_before_extraction(monkeypatch):
    with patch("urllib.request.urlopen") as download:
        download.return_value.__enter__.return_value.read.return_value = b"tampered archive"
        with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
            desktop_runtime._download_binary("runtime", "unused")


def test_download_extracts_only_verified_selected_binary(monkeypatch):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("runtime", b"executable")
        bundle.writestr("../escape", b"never extracted")
    data = archive.getvalue()
    monkeypatch.setattr(desktop_runtime, "ARCHIVE_SHA256", hashlib.sha256(data).hexdigest())
    with patch("urllib.request.urlopen") as download:
        download.return_value.__enter__.return_value.read.return_value = data
        assert (
            desktop_runtime._download_binary("runtime", hashlib.sha256(b"executable").hexdigest())
            == b"executable"
        )


def test_watchdog_stops_native_process_after_parent_sigkill(tmp_path):
    child_pid = tmp_path / "child.pid"
    child_code = (
        "import os,time,pathlib; pathlib.Path(%r).write_text(str(os.getpid())); time.sleep(60)"
        % str(child_pid)
    )
    watchdog = str(Path(desktop_watchdog.__file__))
    parent_code = (
        "import subprocess,sys,time; "
        f'p=subprocess.Popen([sys.executable,{watchdog!r},sys.executable,"-c",{child_code!r}],stdin=subprocess.PIPE); '
        "time.sleep(60)"
    )
    parent = subprocess.Popen([sys.executable, "-c", parent_code])
    try:
        deadline = time.monotonic() + 5
        while not child_pid.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert child_pid.exists()
        pid = int(child_pid.read_text())
        parent.kill()
        parent.wait(timeout=5)
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            pytest.fail("viewer survived its parent being killed")
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        if child_pid.exists():
            try:
                os.kill(int(child_pid.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
