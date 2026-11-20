"""Render a dashboard with the same renderer used by release jobs."""

from pathlib import Path
from runpy import run_path


if __name__ == "__main__":
    module = Path(__file__).resolve().parents[4] / "src" / "gradlab" / "game_dashboard.py"
    run_path(str(module), run_name="__main__")
