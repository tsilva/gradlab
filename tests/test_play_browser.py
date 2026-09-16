from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from gradlab.play_browser import PlaybackBrowser


def test_browser_profiles_are_isolated_and_cleanup_is_owned() -> None:
    processes = [Mock(), Mock()]
    for process in processes:
        process.poll.return_value = None
    with (
        patch("gradlab.play_browser.browser_executable", return_value="/browser"),
        patch("gradlab.play_browser.subprocess.Popen", side_effect=processes) as popen,
    ):
        first, second = PlaybackBrowser(), PlaybackBrowser()
        try:
            first.open("http://127.0.0.1:1234/")
            second.open("http://127.0.0.1:5678/")
            profiles = [Path(first.profile.name), Path(second.profile.name)]
            assert profiles[0] != profiles[1]
            for call, profile in zip(popen.call_args_list, profiles, strict=True):
                assert f"--user-data-dir={profile}" in call.args[0]
                assert call.kwargs["start_new_session"] is True
                assert profile.is_dir()
            first.close()
            processes[0].terminate.assert_called_once()
            processes[1].terminate.assert_not_called()
            assert not profiles[0].exists()
            assert profiles[1].exists()
            first.close()
            processes[0].terminate.assert_called_once()
        finally:
            first.close()
            second.close()
        assert not profiles[1].exists()


def test_failed_launch_removes_profile() -> None:
    browser = PlaybackBrowser()
    profiles = []

    def fail(command, **kwargs):
        profiles.append(Path(command[1].split("=", 1)[1]))
        raise OSError("launch failed")

    with (
        patch("gradlab.play_browser.browser_executable", return_value="/browser"),
        patch("gradlab.play_browser.subprocess.Popen", side_effect=fail),
        pytest.raises(OSError, match="launch failed"),
    ):
        browser.open("http://127.0.0.1:1234/")
    assert not profiles[0].exists()
    assert browser.process is None
    assert browser.profile is None
