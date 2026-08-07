from __future__ import annotations

import hashlib
import json
import os
import queue
import shutil
import subprocess
import threading
import uuid
from collections.abc import Mapping
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from gradlab.file_utils import atomic_write_json, file_sha256, fsync_path
from gradlab import __version__
from gradlab.json_utils import canonical_json_sha256, json_safe
from gradlab.local_paths import default_runs_dir
from gradlab.publication import verify_replay


CAPTURE_DOCUMENT_TYPE = "gradlab.player_episode_capture"
CAPTURE_FORMAT_VERSION = 1
CAPTURE_FPS = 30
CAPTURE_QUEUE_LIMIT = 256
CAPTURE_MAX_COUNT = 32
CAPTURE_MAX_TOTAL_BYTES = 10 * 1024**3
CAPTURE_MAX_FILE_BYTES = 2 * 1024**3
CAPTURE_MAX_WIDTH = 1280
CAPTURE_MAX_HEIGHT = 960


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def default_capture_root() -> Path:
    return default_runs_dir() / "player_publications" / "captures"


def player_source_provenance(repo_root: Path | str) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    commit = ""
    dirty = False
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        commit = commit_result.stdout.strip().lower()
        dirty_result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no", "--", "src/gradlab"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        dirty = bool(dirty_result.stdout.strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    digest = hashlib.sha256()
    source_root = root / "src" / "gradlab"
    if source_root.is_dir():
        for path in sorted(source_root.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(root).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(4, "big"))
            digest.update(relative)
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
    source_digest = digest.hexdigest()
    if not source_digest or source_digest == hashlib.sha256().hexdigest():
        raise ValueError("player source tree identity is unavailable")
    result: dict[str, Any] = {
        "kind": "checkout" if commit else "installed_distribution",
        "distribution": "gradlab",
        "version": __version__,
        "source_tree_sha256": source_digest,
    }
    if commit:
        result.update(git_commit=commit, working_tree_dirty=dirty)
    return result


def _safe_key(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("capture checkpoint identity is required")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _rgb_frame(value: object) -> np.ndarray:
    frame = np.asarray(value)
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError(f"capture frames must have HxWx3 shape; got {frame.shape}")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame[..., :3])


def capture_output_size(height: int, width: int) -> tuple[int, int, str]:
    if height < 1 or width < 1:
        raise ValueError("capture frame dimensions must be positive")
    if width <= CAPTURE_MAX_WIDTH and height <= CAPTURE_MAX_HEIGHT:
        scale = max(
            1,
            min(
                4,
                CAPTURE_MAX_WIDTH // width,
                CAPTURE_MAX_HEIGHT // height,
            ),
        )
        output_width = width * scale
        output_height = height * scale
        interpolation = "nearest"
    else:
        ratio = min(CAPTURE_MAX_WIDTH / width, CAPTURE_MAX_HEIGHT / height)
        output_width = max(2, int(width * ratio) // 2 * 2)
        output_height = max(2, int(height * ratio) // 2 * 2)
        interpolation = "area"
    output_width -= output_width % 2
    output_height -= output_height % 2
    if output_width < 2 or output_height < 2:
        raise ValueError("capture output dimensions must be at least 2x2")
    return output_width, output_height, interpolation


class StreamingReplayWriter:
    """Bounded, lossless raw-frame pipe into one browser-safe MP4."""

    _SENTINEL = object()

    def __init__(
        self,
        output: Path,
        first_frame: object,
        *,
        fps: int = CAPTURE_FPS,
        queue_limit: int = CAPTURE_QUEUE_LIMIT,
        max_bytes: int = CAPTURE_MAX_FILE_BYTES,
    ) -> None:
        self.output = Path(output)
        self.fps = int(fps)
        self.max_bytes = int(max_bytes)
        frame = _rgb_frame(first_frame)
        self.input_shape = tuple(int(value) for value in frame.shape)
        self.width, self.height, self.interpolation = capture_output_size(
            self.input_shape[0], self.input_shape[1]
        )
        self._queue: queue.Queue[object] = queue.Queue(max(1, int(queue_limit)))
        self._thread = threading.Thread(
            target=self._run,
            name="gradlab-player-replay-encoder",
            daemon=True,
        )
        self._error: BaseException | None = None
        self._closed = False
        self.frames = 0
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self._thread.start()
        self.write(frame)

    def _scaled(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape != self.input_shape:
            raise ValueError(
                f"capture frame shape changed: expected {self.input_shape}, got {frame.shape}"
            )
        if (frame.shape[1], frame.shape[0]) == (self.width, self.height):
            return frame
        interpolation = cv2.INTER_NEAREST if self.interpolation == "nearest" else cv2.INTER_AREA
        return np.ascontiguousarray(
            cv2.resize(frame, (self.width, self.height), interpolation=interpolation)
        )

    def _raise_error(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"replay encoder failed: {self._error}") from self._error

    def write(self, frame: object) -> None:
        if self._closed:
            raise RuntimeError("replay encoder is closed")
        self._raise_error()
        converted = self._scaled(_rgb_frame(frame)).copy()
        while True:
            self._raise_error()
            try:
                self._queue.put(converted, timeout=0.25)
                self.frames += 1
                return
            except queue.Full:
                continue

    def _run(self) -> None:
        process: subprocess.Popen[bytes] | None = None
        try:
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg is None:
                raise RuntimeError("ffmpeg is required to capture player episodes")
            command = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{self.width}x{self.height}",
                "-r",
                str(self.fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-profile:v",
                "high",
                "-level",
                "4.0",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(self.output),
            ]
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            assert process.stdin is not None
            while True:
                item = self._queue.get()
                if item is self._SENTINEL:
                    break
                process.stdin.write(np.asarray(item).tobytes())
            process.stdin.close()
            stderr = process.stderr.read() if process.stderr is not None else b""
            return_code = process.wait()
            if return_code:
                message = stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"ffmpeg exited with {return_code}: {message[-1000:]}")
            size = self.output.stat().st_size
            if size < 1:
                raise RuntimeError("replay encoder produced an empty file")
            if size > self.max_bytes:
                raise RuntimeError(
                    f"replay exceeds capture limit: {size} > {self.max_bytes} bytes"
                )
        except BaseException as exc:
            self._error = exc
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()

    def close(self) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("replay encoder is already closed")
        self._closed = True
        while self._thread.is_alive():
            self._raise_error()
            try:
                self._queue.put(self._SENTINEL, timeout=0.25)
                break
            except queue.Full:
                continue
        self._thread.join(timeout=60.0)
        if self._thread.is_alive():
            raise RuntimeError("replay encoder did not finish within 60 seconds")
        self._raise_error()
        return {
            "frames": self.frames,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "size_bytes": self.output.stat().st_size,
        }

    def abort(self) -> None:
        self._closed = True
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        try:
            self._queue.put_nowait(self._SENTINEL)
        except queue.Full:
            pass
        self._thread.join(timeout=5.0)
        self.output.unlink(missing_ok=True)


def validate_capture_document(document: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(document))
    if value.get("document_type") != CAPTURE_DOCUMENT_TYPE:
        raise ValueError("capture document type is invalid")
    if value.get("format_version") != CAPTURE_FORMAT_VERSION:
        raise ValueError("capture document version is invalid")
    capture_id = str(value.get("capture_id") or "")
    fence = str(value.get("capture_fence_sha256") or "")
    if not capture_id.startswith("capture-") or len(capture_id) != 40:
        raise ValueError("capture id is invalid")
    if len(fence) != 64 or any(character not in "0123456789abcdef" for character in fence):
        raise ValueError("capture fence is invalid")
    portable = deepcopy(value)
    portable.pop("capture_id", None)
    portable.pop("capture_fence_sha256", None)
    if canonical_json_sha256(portable) != fence or capture_id != f"capture-{fence[:32]}":
        raise ValueError("capture identity does not match its contents")
    replay = value.get("replay")
    if not isinstance(replay, Mapping) or int(replay.get("frames") or 0) != int(
        value.get("steps") or -1
    ) + 1:
        raise ValueError("capture replay frames must equal steps + 1")
    if value.get("boundary_role") != "terminal_observation":
        raise ValueError("capture must end on a terminal observation")
    execution = value.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("capture execution provenance is required")
    required_execution = {
        "source",
        "qualified_environment_id",
        "environment_hash",
        "runtime_versions",
        "execution_target",
        "device_type",
    }
    missing = sorted(required_execution - set(execution))
    if missing:
        raise ValueError(f"capture execution provenance is missing {missing[0]}")
    return value


class EpisodeCaptureStore:
    def __init__(
        self,
        root: Path | str | None = None,
        *,
        max_count: int = CAPTURE_MAX_COUNT,
        max_total_bytes: int = CAPTURE_MAX_TOTAL_BYTES,
    ) -> None:
        self.root = Path(root or default_capture_root()).expanduser().resolve()
        self.max_count = max(1, int(max_count))
        self.max_total_bytes = max(1, int(max_total_bytes))
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)

    @property
    def latest_path(self) -> Path:
        return self.root / "latest.json"

    def _latest(self) -> dict[str, str]:
        if not self.latest_path.is_file():
            return {}
        value = json.loads(self.latest_path.read_text(encoding="utf-8"))
        return {str(key): str(item) for key, item in value.items()} if isinstance(value, dict) else {}

    def capture_dir(self, capture_id: str) -> Path:
        return self.root / str(capture_id)

    def load(self, capture_id: str) -> dict[str, Any]:
        directory = self.capture_dir(capture_id)
        resolved = directory.resolve(strict=True)
        if resolved.parent != self.root or directory.is_symlink():
            raise ValueError("capture path escapes the capture root")
        document = validate_capture_document(
            json.loads((resolved / "capture.json").read_text(encoding="utf-8"))
        )
        replay = resolved / "replay.mp4"
        if file_sha256(replay) != document["replay"]["sha256"]:
            raise ValueError("capture replay hash mismatch")
        os.utime(resolved)
        return document

    def latest_for(self, checkpoint_identity: object) -> dict[str, Any] | None:
        capture_id = self._latest().get(_safe_key(checkpoint_identity))
        if not capture_id:
            return None
        try:
            return self.load(capture_id)
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            return None

    def commit(
        self,
        *,
        checkpoint_identity: object,
        temporary_replay: Path,
        document_without_identity: Mapping[str, Any],
    ) -> dict[str, Any]:
        portable = deepcopy(dict(document_without_identity))
        fence = canonical_json_sha256(portable)
        document = {
            **portable,
            "capture_id": f"capture-{fence[:32]}",
            "capture_fence_sha256": fence,
        }
        validate_capture_document(document)
        capture_id = str(document["capture_id"])
        destination = self.capture_dir(capture_id)
        temporary_dir = self.root / f".{capture_id}.{uuid.uuid4().hex}.tmp"
        if destination.exists():
            temporary_replay.unlink(missing_ok=True)
        else:
            temporary_dir.mkdir(mode=0o700)
            try:
                os.replace(temporary_replay, temporary_dir / "replay.mp4")
                atomic_write_json(temporary_dir / "capture.json", document)
                fsync_path(temporary_dir / "replay.mp4")
                fsync_path(temporary_dir)
                os.replace(temporary_dir, destination)
                fsync_path(self.root)
            except BaseException:
                shutil.rmtree(temporary_dir, ignore_errors=True)
                raise
        latest = self._latest()
        latest[_safe_key(checkpoint_identity)] = capture_id
        atomic_write_json(self.latest_path, latest)
        self.prune()
        return document

    def _protected_capture_ids(self) -> set[str]:
        requests = self.root.parent / "requests"
        protected: set[str] = set()
        if not requests.is_dir():
            return protected
        for request in requests.iterdir():
            manifest = request / "request.json"
            if not request.is_dir() or request.is_symlink() or not manifest.is_file():
                continue
            try:
                value = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            capture_id = str(value.get("capture_id") or "") if isinstance(value, Mapping) else ""
            if capture_id:
                protected.add(capture_id)
        return protected

    def prune(self) -> None:
        latest = self._latest()
        protected = self._protected_capture_ids()
        entries: list[tuple[float, int, Path]] = []
        for path in self.root.iterdir():
            if not path.is_dir() or path.is_symlink() or not path.name.startswith("capture-"):
                continue
            size = sum(item.stat().st_size for item in path.iterdir() if item.is_file())
            entries.append((path.stat().st_mtime, size, path))
        total = sum(item[1] for item in entries)
        count = len(entries)
        for _mtime, size, path in sorted(entries):
            if count <= self.max_count and total <= self.max_total_bytes:
                break
            if path.name in protected:
                continue
            # A latest capture is evictable only when the global bounds require it.
            shutil.rmtree(path)
            count -= 1
            total -= size
            for key, capture_id in tuple(latest.items()):
                if capture_id == path.name:
                    latest.pop(key, None)
        atomic_write_json(self.latest_path, latest)


class EpisodeCaptureManager:
    """Capture exactly one current faithful policy episode at a time."""

    def __init__(
        self,
        context: Mapping[str, Any] | None,
        *,
        store: EpisodeCaptureStore | None = None,
    ) -> None:
        self.context = deepcopy(dict(context)) if isinstance(context, Mapping) else None
        self.store = store or EpisodeCaptureStore()
        self.writer: StreamingReplayWriter | None = None
        self.temporary_replay: Path | None = None
        self.episode: int | None = None
        self.seed: int | None = None
        self.start_id: str | None = None
        self.sampling_mode: str | None = None
        self.error: str | None = None
        self._episode_in_progress = False
        self._last: dict[str, Any] | None = (
            self.store.latest_for(self.context["checkpoint_identity"])
            if self.context is not None and self.context.get("checkpoint_identity")
            else None
        )

    @property
    def enabled(self) -> bool:
        if not self.context:
            return False
        return (
            self.context.get("source_kind") == "public_run"
            and self.context.get("contract_mode") in {"training", "evaluation"}
            and self.context.get("matches_contract") is True
        )

    def begin(
        self,
        frame: object,
        *,
        episode: int,
        seed: int,
        sampling_mode: str,
    ) -> None:
        self.abort(None)
        self.error = None
        self._episode_in_progress = self.enabled
        if not self.enabled or frame is None:
            return
        temporary_root = self.store.root / ".encoding"
        temporary_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = temporary_root / f"{uuid.uuid4().hex}.mp4"
        try:
            self.writer = StreamingReplayWriter(path, frame)
        except Exception as exc:
            self.error = str(exc)
            path.unlink(missing_ok=True)
            return
        self.temporary_replay = path
        self.episode = int(episode)
        self.seed = int(seed)
        self.sampling_mode = str(sampling_mode)
        self.start_id = None

    def record_transition(self, transition: Any) -> dict[str, Any] | None:
        writer = self.writer
        if writer is None:
            return None
        if str(getattr(transition, "action_source", "")) != "policy":
            self.abort("human or recorded actions are not publishable")
            return None
        frame = getattr(transition, "after_frame", None)
        if frame is None:
            self.abort("transition has no display frame")
            return None
        try:
            writer.write(frame)
        except Exception as exc:
            self.abort(str(exc))
            return None
        start_id = getattr(transition, "start_id", None)
        if start_id is not None:
            self.start_id = str(start_id)
        if not bool(getattr(transition, "boundary", False)):
            return None
        if getattr(transition, "after_frame_role", None) != "terminal_observation":
            self.abort("episode boundary did not expose a terminal observation")
            return None
        try:
            encoding = writer.close()
            self.writer = None
            assert self.temporary_replay is not None
            probe = verify_replay(self.temporary_replay)
            expected_frames = int(getattr(transition, "step")) + 1
            if int(probe["frames"]) != expected_frames or encoding["frames"] != expected_frames:
                raise ValueError(
                    f"capture frame count mismatch: expected {expected_frames}, "
                    f"encoded {probe['frames']}"
                )
            replay = {
                **encoding,
                **probe,
                "sha256": file_sha256(self.temporary_replay),
                "size_bytes": self.temporary_replay.stat().st_size,
            }
            execution = deepcopy(dict(self.context.get("execution") or {}))
            document = {
                "document_type": CAPTURE_DOCUMENT_TYPE,
                "format_version": CAPTURE_FORMAT_VERSION,
                "created_at": _utc_now(),
                "checkpoint_identity": str(self.context["checkpoint_identity"]),
                "source": deepcopy(dict(self.context.get("source") or {})),
                "run_id": str(self.context.get("run_id") or ""),
                "checkpoint_id": str(self.context.get("checkpoint_id") or ""),
                "checkpoint_sha256": str(self.context.get("checkpoint_sha256") or ""),
                "recipe_sha256": str(self.context.get("recipe_sha256") or ""),
                "goal": deepcopy(self.context.get("goal")),
                "contract": deepcopy(dict(self.context.get("contract") or {})),
                "execution": execution,
                "episode": int(self.episode or getattr(transition, "episode", 0)),
                "seed": int(self.seed if self.seed is not None else getattr(transition, "seed")),
                "start_id": self.start_id,
                "sampling_mode": self.sampling_mode,
                "steps": int(getattr(transition, "step")),
                "return": float(getattr(transition, "total_reward")),
                "max_x_pos": int(getattr(transition, "max_x_pos")),
                "terminated": bool(getattr(transition, "terminated")),
                "truncated": bool(getattr(transition, "truncated")),
                "success": bool(getattr(transition, "completed")),
                "outcome": (
                    "success"
                    if bool(getattr(transition, "completed"))
                    else "truncated"
                    if bool(getattr(transition, "truncated"))
                    else "terminated"
                ),
                "boundary_role": "terminal_observation",
                "replay": json_safe(replay),
            }
            result = self.store.commit(
                checkpoint_identity=self.context["checkpoint_identity"],
                temporary_replay=self.temporary_replay,
                document_without_identity=document,
            )
            self._last = result
            self._episode_in_progress = False
            self.temporary_replay = None
            return result
        except Exception as exc:
            self.abort(str(exc))
            return None

    def abort(self, reason: str | None) -> None:
        writer, self.writer = self.writer, None
        if writer is not None:
            writer.abort()
        if self.temporary_replay is not None:
            self.temporary_replay.unlink(missing_ok=True)
            self.temporary_replay = None
        if reason:
            self.error = str(reason)
            self._episode_in_progress = self.enabled

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "recording": self.writer is not None,
            "episode_in_progress": self._episode_in_progress,
            "ready": bool(
                self.enabled
                and not self._episode_in_progress
                and self.writer is None
                and self._last is not None
            ),
            "error": self.error,
            "latest": deepcopy(self._last),
        }


__all__ = [
    "CAPTURE_DOCUMENT_TYPE",
    "CAPTURE_FORMAT_VERSION",
    "EpisodeCaptureManager",
    "EpisodeCaptureStore",
    "StreamingReplayWriter",
    "capture_output_size",
    "validate_capture_document",
]
