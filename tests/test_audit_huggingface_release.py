from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from gradlab.publication import HUGGINGFACE_RELEASE_FILES


SCRIPT = Path(__file__).parents[1] / "scripts/audit_huggingface_release.py"
SPEC = importlib.util.spec_from_file_location("audit_huggingface_release", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeApi:
    def __init__(self, *, main_sha: str = "a" * 40, tag_sha: str = "a" * 40) -> None:
        self.main_sha = main_sha
        self.tag_sha = tag_sha

    def model_info(self, repo_id: str, **kwargs: object) -> SimpleNamespace:
        sha = self.tag_sha if kwargs.get("revision") else self.main_sha
        return SimpleNamespace(sha=sha, private=False)

    def list_repo_refs(self, repo_id: str, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            tags=[
                SimpleNamespace(name="v3", target_commit=self.tag_sha),
                SimpleNamespace(name="checkpoint-10", target_commit=self.tag_sha),
            ]
        )

    def list_repo_files(self, repo_id: str, **kwargs: object) -> list[str]:
        return sorted(HUGGINGFACE_RELEASE_FILES)

    def list_collections(self, **kwargs: object) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                title="GradLab — VizdoomDeathmatch-v1",
                slug="tsilva/gradlab-vizdoom-id",
            )
        ]

    def get_collection(self, slug: str) -> SimpleNamespace:
        return SimpleNamespace(
                title="GradLab — VizdoomDeathmatch-v1",
                slug=slug,
                private=False,
                items=[
                    SimpleNamespace(
                        item_id="tsilva/model",
                        item_type="model",
                        note="Immutable research release: https://huggingface.co/tsilva/model/tree/v3",
                    )
                ],
        )


def test_remote_release_audit_checks_commit_files_collection_and_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for filename in HUGGINGFACE_RELEASE_FILES:
        (tmp_path / filename).write_bytes(filename.encode())
    monkeypatch.setattr(
        MODULE,
        "hf_hub_download",
        lambda repo_id, filename, **kwargs: str(tmp_path / filename),
    )
    monkeypatch.setattr(
        MODULE,
        "validate_release_bundle",
        lambda root: {
            "repository": {
                "repo_id": "tsilva/model",
                "canonical_environment_id": "VizdoomDeathmatch-v1",
            },
            "release": {"version": "v3", "checkpoint_tag": "checkpoint-10"},
            "format_version": 3,
        },
    )
    monkeypatch.setattr(
        MODULE,
        "verify_replay",
        lambda path: {"codec_name": "h264", "frames": 100},
    )

    result = MODULE.audit_huggingface_release("tsilva/model", "v3", api=FakeApi())

    assert result["status"] == "passed"
    assert result["collection"] == "tsilva/gradlab-vizdoom-id"


def test_remote_release_audit_rejects_main_tag_drift() -> None:
    with pytest.raises(ValueError, match="do not point to the same commit"):
        MODULE.audit_huggingface_release(
            "tsilva/model",
            "v3",
            api=FakeApi(main_sha="a" * 40, tag_sha="b" * 40),
        )
