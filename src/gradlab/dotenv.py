from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path


def load_env_file(
    path: str | Path = ".env",
    *,
    key_filter: Callable[[str], bool] | None = None,
) -> None:
    env_path = Path(path)
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key_filter is not None and not key_filter(key):
            continue
        os.environ.setdefault(key, value.strip().strip("'\""))
