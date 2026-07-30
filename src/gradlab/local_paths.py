from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path


PORTABLE_DEFAULT_RUNS_DIR = "~/.config/gradlab/runs"


def default_gradlab_config_dir(
    environment: Mapping[str, str] | None = None,
) -> Path:
    values = os.environ if environment is None else environment
    xdg_root = str(values.get("XDG_CONFIG_HOME") or "").strip()
    if xdg_root:
        root = Path(xdg_root).expanduser()
    else:
        configured_home = str(values.get("HOME") or "").strip()
        home = Path(configured_home).expanduser() if configured_home else Path.home()
        root = home / ".config"
    return (root / "gradlab").resolve()


def default_runs_dir(
    environment: Mapping[str, str] | None = None,
) -> Path:
    return default_gradlab_config_dir(environment) / "runs"
