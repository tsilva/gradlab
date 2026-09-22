"""Pinned, cached Neutralinojs runtime; no Node installation or application build."""

from __future__ import annotations

import hashlib
import io
import os
import platform
import plistlib
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

VERSION = "6.9.0"
ARCHIVE_SHA256 = "ff33dd68979cffc36e4e7e73d1bab4cbcdf2c036aad72ca32876fde51a32df14"
ARCHIVE_URL = (
    f"https://github.com/neutralinojs/neutralinojs/releases/download/v{VERSION}/"
    f"neutralinojs-v{VERSION}.zip"
)
BINARIES = {
    ("darwin", "arm64"): (
        "neutralino-mac_arm64",
        "ef5e8cfd96df35d811d47a127f5be54db50114f1f6b5c075218ca78a07e7a66f",
    ),
    ("darwin", "x86_64"): (
        "neutralino-mac_x64",
        "02616220c49f0e7063145d1d606a27f12d2b7951e23efdd077e5300947481099",
    ),
    ("linux", "x86_64"): (
        "neutralino-linux_x64",
        "47cb0b6e60706f6dff83ed98e198c753580f21f210e3dd87e00719701247fc9d",
    ),
    ("linux", "aarch64"): (
        "neutralino-linux_arm64",
        "e5dfc6a2b2eeb1e7011e3bb987a1e85d5331ec87eadc04d57beb5b02dcc92b67",
    ),
}
ASSETS = Path(__file__).with_name("desktop")


def viewer_executable(role: str) -> Path:
    """Give macOS each viewer its own app identity and content-versioned icon."""
    if role not in {"player", "stats"}:
        raise ValueError(f"Unknown desktop viewer role: {role}")
    executable = runtime_executable()
    if sys.platform != "darwin":
        return executable
    import fcntl

    icon_path = ASSETS / f"{role}.png"
    revision = hashlib.sha256(icon_path.read_bytes()).hexdigest()[:16]
    app_name = f"GradLab {role.title()}"
    root = executable.parents[3] / "viewers"
    destination = root / f"{role}-{revision}" / f"{app_name}.app"
    output = destination / "Contents/MacOS/GradLab"
    root.mkdir(exist_ok=True)
    with (root / "install.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output.is_file():
            _verified(output.read_bytes(), hashlib.sha256(executable.read_bytes()).hexdigest())
            return output
        with tempfile.TemporaryDirectory(dir=root, prefix="install-") as temporary:
            staged = Path(temporary) / destination.name
            shutil.copytree(executable.parents[2], staged)
            from PIL import Image

            with Image.open(icon_path) as icon:
                icon.save(staged / "Contents/Resources/GradLab.icns", format="ICNS")
            info_path = staged / "Contents/Info.plist"
            info = plistlib.loads(info_path.read_bytes())
            info.update(
                {
                    "CFBundleIdentifier": f"org.gradlab.viewer.{role}",
                    "CFBundleName": app_name,
                    "CFBundleDisplayName": app_name,
                }
            )
            info_path.write_bytes(plistlib.dumps(info))
            destination.parent.mkdir(exist_ok=True)
            staged.rename(destination)
    return output


def _verified(data: bytes, digest: str) -> bytes:
    if hashlib.sha256(data).hexdigest() != digest:
        raise RuntimeError("Neutralinojs SHA-256 mismatch; refusing to execute the runtime")
    return data


def _download_binary(name: str, digest: str) -> bytes:
    print(
        f"Installing Neutralinojs {VERSION} desktop viewer (cached after first launch)…", flush=True
    )
    try:
        with urllib.request.urlopen(ARCHIVE_URL, timeout=60) as response:
            data = response.read(32 * 1024 * 1024 + 1)
    except OSError as exc:
        raise RuntimeError(
            "Could not download the GradLab desktop viewer. "
            "Check your connection, or use --no-open and open the printed URL manually."
        ) from exc
    with zipfile.ZipFile(io.BytesIO(_verified(data, ARCHIVE_SHA256))) as archive:
        return _verified(archive.read(name), digest)


def runtime_executable() -> Path:
    """Install once under an exclusive lock, verify executable bytes on every launch."""
    target = (sys.platform, platform.machine().lower())
    if target not in BINARIES:
        raise RuntimeError(f"No GradLab desktop viewer for {target}; use --no-open")
    import fcntl

    name, digest = BINARIES[target]
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "gradlab/neutralino"
    root = cache / VERSION / name
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / "install.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        relative = (
            Path("GradLab.app/Contents/MacOS/GradLab")
            if sys.platform == "darwin"
            else Path("GradLab")
        )
        executable = root / relative
        if executable.is_file():
            _verified(executable.read_bytes(), digest)
            return executable
        binary = _download_binary(name, digest)
        root.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root.parent, prefix="install-") as temporary:
            staged = Path(temporary) / "runtime"
            output = staged / relative
            output.parent.mkdir(parents=True)
            output.write_bytes(binary)
            output.chmod(0o755)
            shutil.copyfile(ASSETS / "NEUTRALINO_LICENSE.txt", staged / "LICENSE.txt")
            if sys.platform == "darwin":
                from PIL import Image

                contents = output.parent.parent
                resources = contents / "Resources"
                resources.mkdir()
                with Image.open(ASSETS / "icon.png") as icon:
                    icon.save(resources / "GradLab.icns", format="ICNS")
                (contents / "Info.plist").write_bytes(
                    plistlib.dumps(
                        {
                            "CFBundleExecutable": "GradLab",
                            "CFBundleIdentifier": "org.gradlab.viewer",
                            "CFBundleName": "GradLab",
                            "CFBundleDisplayName": "GradLab",
                            "CFBundleIconFile": "GradLab.icns",
                            "CFBundlePackageType": "APPL",
                            "CFBundleVersion": VERSION,
                            "NSHighResolutionCapable": True,
                        }
                    )
                )
            staged.rename(root)
        return executable
