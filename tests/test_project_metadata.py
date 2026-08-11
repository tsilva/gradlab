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
