from __future__ import annotations

from pathlib import Path

from gradlab.dotenv import load_env_file
from gradlab.operator_credentials import (
    OperatorEnvironmentReport,
    load_operator_environment,
    reject_protected_dotenv,
)


def load_repository_operator_environment(
    root: str | Path,
) -> OperatorEnvironmentReport:
    """Load safe repository metadata and the private operator configuration."""
    dotenv_path = Path(root).resolve() / ".env"
    reject_protected_dotenv(dotenv_path)
    load_env_file(dotenv_path)
    return load_operator_environment()
