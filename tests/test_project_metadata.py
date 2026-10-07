from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_project_declares_and_distributes_mit_license() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert project["project"]["license"] == "MIT"
    assert project["project"]["license-files"] == ["LICENSE"]
    assert "/LICENSE" in project["tool"]["hatch"]["build"]["targets"]["sdist"][
        "include"
    ]
    assert license_text.startswith("MIT License\n\nCopyright (c) 2026 Tiago Silva")
    assert "[MIT License](LICENSE)" in readme


def test_uv_tool_config_matches_project_resolution_policy() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    tool_config = tomllib.loads((ROOT / "uv-tool.toml").read_text(encoding="utf-8"))
    project_uv = project["tool"]["uv"]

    assert tool_config["exclude-newer"] == project_uv["exclude-newer"]
    assert tool_config["exclude-newer-package"] == project_uv["exclude-newer-package"]
    assert "dependency-metadata" not in tool_config
    assert "dependency-metadata" not in project_uv


def test_install_script_uses_repository_uv_tool_config() -> None:
    install_script = (ROOT / "install.sh").read_text(encoding="utf-8")

    assert install_script.count('--config-file "$ROOT/uv-tool.toml"') == 2
