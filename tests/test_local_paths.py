from pathlib import Path

from gradlab.local_paths import (
    default_gradlab_config_dir,
    default_runs_dir,
)


def test_default_runs_dir_uses_home_config_directory() -> None:
    environment = {"HOME": "/tmp/gradlab-home"}

    assert default_gradlab_config_dir(environment) == Path(
        "/tmp/gradlab-home/.config/gradlab"
    ).resolve()
    assert default_runs_dir(environment) == Path(
        "/tmp/gradlab-home/.config/gradlab/runs"
    ).resolve()


def test_default_runs_dir_honors_xdg_config_home() -> None:
    environment = {
        "HOME": "/tmp/gradlab-home",
        "XDG_CONFIG_HOME": "/tmp/gradlab-xdg",
    }

    assert default_runs_dir(environment) == Path("/tmp/gradlab-xdg/gradlab/runs").resolve()
