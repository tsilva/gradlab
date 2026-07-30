from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[1]
CLEANUP = ROOT / "ops" / "dstack" / "gradlab-dstack-image-cleanup"
SERVICE = ROOT / "ops" / "dstack" / "gradlab-dstack-image-cleanup.service"
PORTABLE_COMPUTE_FILES = (
    ROOT / "COMPUTE.md",
    ROOT / "ops" / "operator.example.toml",
    ROOT / "ops" / "dstack" / "README.md",
    ROOT / "ops" / "dstack" / "fleet.example.dstack.yml",
    ROOT / "ops" / "dstack" / "local-smoke.task.dstack.example.yml",
)


def _run_cleanup(*, project: str, repository: str) -> subprocess.CompletedProcess[str]:
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "DSTACK_PROJECT": project,
        "GRADLAB_IMAGE_REPOSITORY": repository,
        "DSTACK_SERVER_ADMIN_TOKEN": "test-token",
        "DSTACK_BIN": "/definitely/missing/dstack",
    }
    return subprocess.run(
        ["/bin/sh", str(CLEANUP)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_image_cleanup_requires_explicit_dstack_project() -> None:
    completed = _run_cleanup(project="", repository="registry.example/gradlab/train")

    assert completed.returncode != 0
    assert "DSTACK_PROJECT is required" in completed.stderr


def test_image_cleanup_requires_explicit_image_repository() -> None:
    completed = _run_cleanup(project="research", repository="")

    assert completed.returncode != 0
    assert "GRADLAB_IMAGE_REPOSITORY is required" in completed.stderr


def test_cleanup_service_does_not_override_host_owned_project() -> None:
    unit = SERVICE.read_text(encoding="utf-8")

    assert "EnvironmentFile=/etc/gradlab/dstack/server.env" in unit
    assert "Environment=DSTACK_PROJECT=" not in unit


def test_checked_in_compute_configuration_is_portable_or_templated() -> None:
    assert not (ROOT / "INSTANCES.md").exists()
    assert not (ROOT / "ops" / "dstack" / "b3.fleet.dstack.yml").exists()
    combined = "\n".join(path.read_text(encoding="utf-8") for path in PORTABLE_COMPUTE_FILES)

    for private_value in (
        "beast-3",
        "tsilva@",
        "name: b3",
        "DSTACK_PROJECT=main",
    ):
        assert private_value not in combined
    assert "<dstack-project>" in combined
    assert "<local-fleet>" in combined
    assert "<ssh-user>" in combined
