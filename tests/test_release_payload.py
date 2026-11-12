"""Release publication must preserve the reviewed body and exact source identity."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/release_payload.py"


def run(*args, cwd, check=True, input=None):
    return subprocess.run(args, cwd=cwd, input=input, capture_output=True, text=True, check=check)


@pytest.fixture
def prepared(tmp_path):
    run("git", "init", "--quiet", cwd=tmp_path)
    tree = run("git", "mktree", cwd=tmp_path, input="").stdout.strip()
    sha = run(
        "git",
        "-c",
        "user.name=Release Test",
        "-c",
        "user.email=release@example.test",
        "-c",
        "commit.gpgSign=false",
        "commit-tree",
        tree,
        "-m",
        "Fixture",
        cwd=tmp_path,
    ).stdout.strip()
    notes = tmp_path / "reviewed.md"
    body = "Fix installed playback.\n\n- Preserve exact provenance.\n- Keep Unicode: 🍄.\n"
    notes.write_text(body)
    message = tmp_path / "tag-message"
    run(
        sys.executable,
        str(SCRIPT),
        "prepare",
        "--version",
        "1.2.3",
        "--source",
        sha,
        "--notes-file",
        str(notes),
        "--output",
        str(message),
        cwd=tmp_path,
    )
    return tmp_path, sha, message, body


def tag(root, sha, message):
    run(
        "git",
        "-c",
        "user.name=Release Test",
        "-c",
        "user.email=release@example.test",
        "-c",
        "tag.gpgSign=false",
        "tag",
        "-a",
        "v1.2.3",
        sha,
        "-F",
        str(message),
        cwd=root,
    )


def extract(root, sha):
    return run(
        sys.executable,
        str(SCRIPT),
        "extract",
        "--version",
        "1.2.3",
        "--source",
        sha,
        "--tag",
        "v1.2.3",
        "--output",
        str(root / "published.md"),
        cwd=root,
        check=False,
    )


def test_reviewed_notes_survive_annotated_tag_exactly(prepared):
    root, sha, message, body = prepared
    tag(root, sha, message)
    assert extract(root, sha).returncode == 0
    assert (root / "published.md").read_text() == body
    assert (root / "published.md.title").read_text() == "v1.2.3"


@pytest.mark.parametrize("field,value", [("source_sha", "b" * 40), ("version", "9.9.9")])
def test_notes_with_a_different_source_or_version_are_rejected(prepared, field, value):
    root, sha, message, _ = prepared
    subject, body = message.read_text().split("\n\n", 1)
    payload = json.loads(body)
    payload[field] = value
    message.write_text(subject + "\n\n" + json.dumps(payload))
    tag(root, sha, message)
    assert extract(root, sha).returncode != 0
    assert not (root / "published.md").exists()


def test_lightweight_tags_cannot_supply_reviewed_notes(prepared):
    root, sha, _, _ = prepared
    run("git", "-c", "tag.gpgSign=false", "tag", "v1.2.3", sha, cwd=root)
    assert extract(root, sha).returncode != 0


def test_extract_rejects_a_different_commit(prepared):
    root, sha, message, _ = prepared
    tag(root, sha, message)
    assert extract(root, "b" * 40).returncode != 0
