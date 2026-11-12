"""Bind reviewed release notes to an annotated tag and its exact source commit."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from release_notes import validate_notes


SCHEMA = "gradlab.release-notes.v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "extract"))
    parser.add_argument("--version", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--notes-file", type=Path)
    parser.add_argument("--tag")
    args = parser.parse_args()
    if not re.fullmatch(r"\d+\.\d+\.\d+", args.version):
        parser.error("expected a final release version")
    if not re.fullmatch(r"[0-9a-f]{40}", args.source):
        parser.error("expected the full source commit SHA")
    tag = f"v{args.version}"
    if args.operation == "prepare":
        if args.notes_file is None:
            parser.error("prepare requires --notes-file")
        payload = {
            "schema": SCHEMA,
            "version": args.version,
            "source_sha": args.source,
            "title": tag,
            "body": validate_notes(args.notes_file.read_text(encoding="utf-8")),
        }
        args.output.write_text(
            f"GradLab {tag}\n\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return
    if args.tag != tag:
        parser.error("tag must match the release version")
    ref = f"refs/tags/{tag}"
    kind = subprocess.check_output(["git", "cat-file", "-t", ref], text=True).strip()
    source = subprocess.check_output(["git", "rev-parse", f"{ref}^{{commit}}"], text=True).strip()
    if kind != "tag" or source != args.source:
        parser.error("release requires an annotated tag at the exact source SHA")
    body = subprocess.check_output(
        ["git", "for-each-ref", "--format=%(contents:body)", ref], text=True
    )
    payload = json.loads(body)
    expected = {
        "schema": SCHEMA,
        "version": args.version,
        "source_sha": args.source,
        "title": tag,
    }
    if set(payload) != {*expected, "body"} or any(
        payload.get(key) != value for key, value in expected.items()
    ):
        parser.error("release notes do not match the tag, version and source SHA")
    notes = validate_notes(payload["body"])
    if notes != payload["body"]:
        parser.error("release notes must retain their prepared bytes")
    args.output.write_text(notes, encoding="utf-8")
    Path(str(args.output) + ".title").write_text(payload["title"], encoding="utf-8")


if __name__ == "__main__":
    main()
