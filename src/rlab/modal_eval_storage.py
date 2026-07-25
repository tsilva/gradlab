from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

from rlab.r2_store import public_object_request


def write_downloaded_file(url: str, destination: Path) -> Path:
    parsed = urlparse(url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        try:
            with os.fdopen(fd, "wb") as handle:
                if parsed.scheme == "file":
                    with Path(unquote(parsed.path)).open("rb") as source:
                        shutil.copyfileobj(source, handle, length=1024 * 1024)
                else:
                    import urllib.request

                    with urllib.request.urlopen(
                        public_object_request(url),
                        timeout=60,
                    ) as response:
                        shutil.copyfileobj(response, handle, length=1024 * 1024)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception as exc:
            raise RuntimeError(f"object download failed: {type(exc).__name__}") from exc
        os.replace(name, destination)
    finally:
        Path(name).unlink(missing_ok=True)
    return destination
