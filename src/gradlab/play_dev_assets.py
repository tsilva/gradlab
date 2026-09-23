"""Source-checkout player assets served by a short-lived Vite process."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path


def source_checkout_root() -> Path | None:
    root = Path(__file__).resolve().parents[2]
    required = ("vite.config.ts", "frontend/main.ts", "scripts/player-dev-server.mjs")
    return root if all((root / name).is_file() for name in required) else None


class PlayerDevAssets:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.process: asyncio.subprocess.Process | None = None
        self.url: str | None = None

    async def start(self) -> str:
        node = shutil.which("node")
        vite = self.root / "node_modules/vite"
        if node is None or not vite.is_dir():
            raise RuntimeError(
                "Hot reload requires Node and checkout dependencies. "
                "Run pnpm install --frozen-lockfile, or use --no-hot-reload."
            )
        self.process = await asyncio.create_subprocess_exec(
            node,
            str(self.root / "scripts/player-dev-server.mjs"),
            cwd=self.root,
            stdout=asyncio.subprocess.PIPE,
        )
        try:
            assert self.process.stdout is not None
            line = await asyncio.wait_for(self.process.stdout.readline(), timeout=20)
            prefix = b"GRADLAB_VITE_URL="
            if not line.startswith(prefix):
                raise RuntimeError("Vite did not report its development URL")
            self.url = line[len(prefix):].decode().strip()
            return self.url
        except BaseException:
            await self.stop()
            raise

    async def stop(self) -> None:
        process = self.process
        self.process = None
        self.url = None
        if process is None:
            return
        if process.returncode is None:
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()


def development_page(markup: str, vite_url: str) -> str:
    return (
        markup.replace(
            '<link rel="stylesheet" href="/assets/styles.css">',
            f'<link rel="stylesheet" href="{vite_url}/src/gradlab/web_player/styles.css">',
        ).replace(
            '<script type="module" src="/assets/app.js"></script>',
            f'<script type="module" src="{vite_url}/@vite/client"></script>\n'
            f'    <script type="module" src="{vite_url}/frontend/main.ts"></script>',
        )
    )
