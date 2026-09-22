"""Require fresh, prebuilt player assets in distributions; never invoke Node."""

import hashlib
import json
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        if version == "editable":
            # Editable installs use the source tree, including assets built later.
            # Hatch otherwise force-includes dist before it exists on clean clones.
            assets = Path(self.root) / "src/gradlab/web_player/dist"
            build_data["force_include_editable"] = {
                source: target
                for source, target in self.build_config.get_force_include().items()
                if Path(source) != assets
            }
            return
        root = Path(self.root)
        manifest = root / "src/gradlab/web_player/dist/build-manifest.json"
        message = "Build the player first with pnpm install --frozen-lockfile && pnpm build:web."
        if not manifest.is_file():
            raise RuntimeError(message)
        build = json.loads(manifest.read_text())
        inputs = build["inputs"]
        expected_paths = {
            path.relative_to(root).as_posix()
            for directory in ("frontend", "src/gradlab/web_player")
            for path in (root / directory).rglob("*")
            if path.is_file() and "dist" not in path.relative_to(root / directory).parts
        }
        expected_paths.update(("package.json", "pnpm-lock.yaml", "pnpm-workspace.yaml", ".npmrc", "vite.config.ts", "svelte.config.js", "tsconfig.json", "scripts/player-build-inputs.mjs"))
        if set(inputs) != expected_paths:
            raise RuntimeError(f"Player build inputs changed. {message}")
        for name, expected in inputs.items():
            path = root / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Stale player build: {name}. {message}")
        assets = manifest.parent
        for name, expected in build["outputs"].items():
            path = assets / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Missing or altered player asset: {name}. {message}")
        for name in ("index.html", "app.js", "styles.css", "sources/browser.js"):
            if not (assets / name).is_file():
                raise RuntimeError(f"Missing player asset: {name}. {message}")
