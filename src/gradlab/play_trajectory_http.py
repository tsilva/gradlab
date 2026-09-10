"""Authenticated, disk-backed trajectory uploads and single-use downloads."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
import secrets
import shutil
import tempfile
import time

from aiohttp import web

from gradlab.play_trajectory import MAX_ARCHIVE_BYTES, export_trajectory, file_sha256
from gradlab.policy_bundle import load_policy_bundle


def prepare_archive(frozen: Path, root: Path) -> Path:
    """Name an archive by its environment, checkpoint, and exact byte content."""
    frozen = Path(frozen)
    metadata = json.loads((frozen / "metadata.json").read_bytes())
    environment = metadata["initial_snapshot"]["session"]["env_id"]
    environment = re.sub(r"[^A-Za-z0-9_-]+", "-", environment).strip("-")[:64] or "environment"
    bundle = load_policy_bundle(frozen / "checkpoint")
    checkpoint = bundle.checkpoint_sha256[:16]
    path = export_trajectory(frozen, root / "episode.trj")
    named = root / f"{environment}-checkpoint-{checkpoint}-{file_sha256(path)}.trj"
    path.rename(named)
    return named


async def finish_thread(function, *args, cancelled_result=None):
    """Keep temporary files alive until background I/O stops, including cancellation."""
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        result = await task
        if cancelled_result is not None:
            cancelled_result(result)
        raise


class TrajectoryTransfers:
    def __init__(self, runner, authorize, authorize_control):
        self.runner = runner
        self.authorize = authorize
        self.authorize_control = authorize_control
        self.downloads: dict[str, tuple[Path, float]] = {}
        self.preparing = 0

    def routes(self):
        return [
            web.post("/api/trajectory/download", self.prepare),
            web.get("/api/trajectory/download/{ticket}", self.download),
            web.post("/api/trajectory/import", self.import_episode),
        ]

    async def prepare(self, request):
        self.authorize(request)
        self.expire()
        if self.preparing + len(self.downloads) >= 2:
            raise web.HTTPTooManyRequests(text="Finish the current trajectory downloads first")
        self.preparing += 1
        root = Path(tempfile.mkdtemp(prefix="gradlab-trajectory-transfer-"))
        frozen = None
        try:
            # The prefix is fixed before encoding. Playback never waits for Parquet.
            frozen = await finish_thread(
                self.runner.freeze_trajectory,
                cancelled_result=lambda path: shutil.rmtree(path, ignore_errors=True),
            )
            path = await finish_thread(prepare_archive, frozen, root)
            ticket = secrets.token_urlsafe(32)
            self.downloads[ticket] = (path, time.monotonic())
            return web.json_response({"url": f"/api/trajectory/download/{ticket}", "filename": path.name})
        except (ValueError, OSError, RuntimeError) as exc:
            shutil.rmtree(root, ignore_errors=True)
            return web.json_response({"error": str(exc)}, status=400)
        except BaseException:
            shutil.rmtree(root, ignore_errors=True)
            raise
        finally:
            if frozen is not None:
                shutil.rmtree(frozen, ignore_errors=True)
            self.preparing -= 1

    async def download(self, request):
        # An unguessable single-use capability lets the browser stream directly to disk.
        item = self.downloads.pop(request.match_info["ticket"], None)
        if item is None:
            raise web.HTTPNotFound()
        path, _ = item
        root = path.parent
        try:
            response = web.StreamResponse(
                headers={
                    "Content-Type": "application/zip",
                    "Content-Disposition": f'attachment; filename="{path.name}"',
                    "Content-Length": str(path.stat().st_size),
                    "Cache-Control": "no-store",
                }
            )
            await response.prepare(request)
            with path.open("rb") as stream:
                while chunk := await finish_thread(stream.read, 1024**2):
                    await response.write(chunk)
            await response.write_eof()
            return response
        finally:
            shutil.rmtree(root, ignore_errors=True)

    async def import_episode(self, request):
        self.authorize_control(request)
        root = Path(tempfile.mkdtemp(prefix="gradlab-trajectory-upload-"))
        try:
            path = root / "episode.trj"
            size = 0
            with path.open("wb") as stream:
                async for chunk in request.content.iter_chunked(1024**2):
                    size += len(chunk)
                    if size > MAX_ARCHIVE_BYTES:
                        raise web.HTTPRequestEntityTooLarge(
                            max_size=MAX_ARCHIVE_BYTES, actual_size=size
                        )
                    await finish_thread(stream.write, chunk)
            # Recheck the control lease after a potentially long upload.
            self.authorize_control(request)
            await finish_thread(self.runner.import_trajectory, str(path))
            return web.json_response({"ok": True})
        except (ValueError, OSError, RuntimeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def expire(self):
        for ticket, (path, created) in list(self.downloads.items()):
            if time.monotonic() - created > 300:
                del self.downloads[ticket]
                shutil.rmtree(path.parent, ignore_errors=True)

    def close(self):
        for path, _ in self.downloads.values():
            shutil.rmtree(path.parent, ignore_errors=True)
        self.downloads.clear()
