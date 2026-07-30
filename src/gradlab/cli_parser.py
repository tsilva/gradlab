from __future__ import annotations

import argparse
from typing import Any


class ExactArgumentParser(argparse.ArgumentParser):
    """Argument parser that accepts only declared option names."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["allow_abbrev"] = False
        super().__init__(*args, **kwargs)


__all__ = ["ExactArgumentParser"]
