from __future__ import annotations

from collections.abc import Collection
from pathlib import Path

from gradlab.dotenv import load_env_file
from gradlab.operator_credentials import (
    OperatorEnvironmentReport,
    load_operator_environment,
    reject_protected_dotenv,
)


def load_repository_operator_environment(
    root: str | Path,
    *,
    requested_names: Collection[str] | None = None,
) -> OperatorEnvironmentReport:
    """Load safe repository metadata and the private operator configuration."""
    dotenv_path = Path(root).resolve() / ".env"
    reject_protected_dotenv(dotenv_path)
    load_env_file(
        dotenv_path,
        key_filter=(
            None
            if requested_names is None
            else lambda name: name in requested_names
        ),
    )
    return load_operator_environment(requested_names=requested_names)
