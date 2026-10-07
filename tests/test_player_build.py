"""Exercise the real editable build without pre-existing frontend artifacts."""

from pathlib import Path
import shutil
import subprocess
import sys
import tomllib


def test_editable_install_before_frontend_build(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    project = tomllib.loads((repository / "pyproject.toml").read_text())
    for name in ("pyproject.toml", "hatch_build.py", "README.md", "LICENSE"):
        shutil.copy2(repository / name, checkout / name)
    for name in project["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]:
        if name == "src/gradlab/web_player/dist":
            continue  # Generated assets are absent from a fresh checkout.
        source, destination = repository / name, checkout / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
    package = checkout / "src/gradlab"
    package.mkdir(parents=True, exist_ok=True)
    shutil.copy2(repository / "src/gradlab/__init__.py", package / "__init__.py")
    installed = tmp_path / "installed"
    result = subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--no-deps",
            "--target",
            str(installed),
            "--editable",
            str(checkout),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    # Distribution builds must still reject this same checkout without assets.
    result = subprocess.run(
        ["uv", "build", "--wheel", str(checkout), "--out-dir", str(tmp_path / "wheels")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Build the player first" in result.stderr

    # A subsequent frontend build must be visible through the editable source path.
    assets = package / "web_player/dist"
    assets.mkdir(parents=True)
    (assets / "app.js").write_text("built after installation")
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            "import pathlib, site, sys; site.addsitedir(sys.argv[1]); import gradlab; "
            "assert pathlib.Path(gradlab.__file__).parent == pathlib.Path(sys.argv[2]); "
            "assert (pathlib.Path(gradlab.__file__).parent / 'web_player/dist/app.js')"
            ".read_text() == 'built after installation'",
            str(installed),
            str(package),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
