#!/usr/bin/env python3
"""Generate release-note drafts from Git without a checked-in changelog.

Canonical helper: private-skills_release-workflow/scripts/release_notes.py. Portable copies
may be bundled by projects whose CI needs commit-based notes before publishing.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


def capture(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def validate_notes(notes: str) -> str:
    visible = re.sub(r"<!--.*?-->", "", notes, flags=re.DOTALL)
    prose = [
        line.strip()
        for line in visible.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not any(character.isalnum() for line in prose for character in line):
        raise ValueError("release notes must contain meaningful text")
    return notes.strip() + "\n"


def generate_notes(
    root: Path, version: str, *, ref: str = "HEAD", tag_prefix: str = "v"
) -> str:
    if capture(root, "rev-parse", "--is-shallow-repository") == "true":
        raise ValueError("release notes require full Git history and release tags")
    commit = capture(root, "rev-parse", "--verify", f"{ref}^{{commit}}")
    try:
        previous = capture(
            root,
            "describe",
            "--tags",
            "--abbrev=0",
            "--match",
            f"{tag_prefix}[0-9]*",
            "--exclude",
            f"{tag_prefix}{version}",
            commit,
        )
    except subprocess.CalledProcessError:
        previous = None
    revision = f"{previous}..{commit}" if previous else commit
    subjects = capture(root, "log", "--reverse", "--format=%s", revision).splitlines()
    changes = list(
        dict.fromkeys(
            subject.strip()
            for subject in subjects
            if subject.strip()
            and not subject.startswith(("Release ", "Bump version to "))
        )
    )
    if not changes:
        raise ValueError(
            "no releasable commits found; supply reviewed notes with --notes-file"
        )
    return validate_notes(
        "## Changes\n\n" + "\n".join(f"- {subject}" for subject in changes)
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-path", type=Path, default=Path.cwd())
    parser.add_argument("--version", required=True)
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--tag-prefix", default="v")
    parser.add_argument(
        "--notes-file",
        type=Path,
        help="Use reviewed notes instead of generating a draft",
    )
    args = parser.parse_args(argv)
    try:
        notes = (
            validate_notes(args.notes_file.read_text(encoding="utf-8"))
            if args.notes_file
            else generate_notes(
                args.repo_path,
                args.version,
                ref=args.ref,
                tag_prefix=args.tag_prefix,
            )
        )
    except ValueError as error:
        parser.error(str(error))
    print(notes, end="")


if __name__ == "__main__":
    main()
