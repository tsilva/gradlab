"""Current-episode data-only annotations and archive validation."""

from __future__ import annotations

import base64
from copy import deepcopy
import io
import re
import uuid

from PIL import Image

MAX_BOOKMARKS = 128


def bookmark_name(value):
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= 120:
        raise ValueError("bookmark name must contain 1–120 characters")
    return value.strip()


def create_bookmark(recording, step, name):
    if len(recording.metadata["bookmarks"]) >= MAX_BOOKMARKS:
        raise ValueError("current episode has 128 bookmarks; delete one first")
    if type(step) is not int:
        raise ValueError("bookmark step must be an integer")
    if step == recording.metadata["first_step"] - 1:
        row = (
            recording.transition(recording.metadata["first_step"])
            if recording.status()["transitions"]
            else None
        )
        frame = row["before_image"] if row else None
    else:
        row = recording.transition(step)
        frame = row["after_image"]
    thumbnail = None
    if frame is not None:
        image = Image.fromarray(frame)
        image.thumbnail((160, 120))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        thumbnail = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {
        "id": uuid.uuid4().hex,
        "name": bookmark_name(name),
        "step": step,
        "episode": recording.metadata["episode"],
        "thumbnail": thumbnail,
    }


def validate_annotations(metadata):
    revision = metadata.get("trajectory_revision")
    if type(revision) is not int or revision < 0:
        raise ValueError("invalid trajectory revision")
    bookmarks = metadata.get("bookmarks")
    if not isinstance(bookmarks, list) or len(bookmarks) > MAX_BOOKMARKS:
        raise ValueError("invalid bookmark inventory")
    first = metadata["first_step"] - 1
    last = first + metadata["transition_count"]
    ids = set()
    for bookmark in bookmarks:
        if not isinstance(bookmark, dict) or set(bookmark) != {
            "id",
            "name",
            "step",
            "episode",
            "thumbnail",
        }:
            raise ValueError("invalid bookmark annotation")
        if not re.fullmatch(r"[0-9a-f]{32}", str(bookmark["id"])) or bookmark["id"] in ids:
            raise ValueError("invalid or duplicate bookmark identity")
        ids.add(bookmark["id"])
        bookmark_name(bookmark["name"])
        if (
            type(bookmark["step"]) is not int
            or not first <= bookmark["step"] <= last
            or bookmark["episode"] != metadata["episode"]
        ):
            raise ValueError("bookmark is outside the recorded episode")
        thumbnail = bookmark["thumbnail"]
        if thumbnail is not None:
            if not isinstance(thumbnail, str) or len(thumbnail) > 128 * 1024:
                raise ValueError("invalid bookmark thumbnail")
            data = base64.b64decode(thumbnail, validate=True)
            with Image.open(io.BytesIO(data)) as image:
                if image.format != "PNG" or image.width > 160 or image.height > 120:
                    raise ValueError("invalid bookmark thumbnail dimensions")
                image.verify()
    cuts = metadata.get("resampling")
    if not isinstance(cuts, list) or len(cuts) != revision or len(cuts) > 10000:
        raise ValueError("invalid resampling provenance")
    for index, cut in enumerate(cuts, 1):
        if (
            not isinstance(cut, dict)
            or cut.get("revision") != index
            or type(cut.get("cut_step")) is not int
            or cut["cut_step"] < first
            or type(cut.get("next_sequence")) is not int
            or cut["next_sequence"] < 1
            or cut.get("sampling_stream") != "continued"
            or not isinstance(cut.get("sampling_mode"), str)
        ):
            raise ValueError("invalid resampling cut")
    if cuts and metadata["classification"] != "counterfactual":
        raise ValueError("resampling must be Counterfactual Playback")


def annotations_payload(metadata):
    return {
        key: deepcopy(metadata[key]) for key in ("bookmarks", "trajectory_revision", "resampling")
    }
