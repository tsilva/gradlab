from types import SimpleNamespace
import hashlib
import pytest
from gradlab import play_capture


def installed_package(tmp_path, monkeypatch, name="site-packages"):
    package = tmp_path / name / "gradlab"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('version = "test"\n')
    (package / "play_capture.py").write_text("# player implementation\n")
    monkeypatch.setattr(play_capture, "__file__", str(package / "play_capture.py"))
    return package


def test_installed_identity_is_portable_and_changes_with_implementation(tmp_path, monkeypatch):
    first = installed_package(tmp_path, monkeypatch, "one")
    original = play_capture.player_source_provenance(first)
    assert play_capture.player_source_provenance() == original
    second = installed_package(tmp_path, monkeypatch, "two")
    assert play_capture.player_source_provenance(second) == original
    assert original["kind"] == "installed_distribution"
    assert original["source_tree_sha256"] != hashlib.sha256().hexdigest()
    (second / "play_capture.py").write_text("# changed implementation\n")
    assert play_capture.player_source_provenance(second) != original


def test_installed_identity_ignores_generated_bytecode(tmp_path, monkeypatch):
    package = installed_package(tmp_path, monkeypatch)
    before = play_capture.player_source_provenance(package)
    (package / "__pycache__").mkdir()
    (package / "__pycache__/play_capture.cpython-314.pyc").write_bytes(b"cache")
    (package / "old.pyc").write_bytes(b"old cache")
    assert play_capture.player_source_provenance(package) == before


def test_installed_identity_does_not_borrow_enclosing_git_checkout(tmp_path, monkeypatch):
    package = installed_package(tmp_path, monkeypatch)

    def no_git(*args, **kwargs):
        raise AssertionError("installed provenance must not probe ancestor Git identity")

    monkeypatch.setattr(play_capture.subprocess, "run", no_git)
    result = play_capture.player_source_provenance(package)
    assert result["kind"] == "installed_distribution"
    assert "git_commit" not in result


def test_missing_source_still_fails_closed(tmp_path, monkeypatch):
    installed_package(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="player source tree identity is unavailable"):
        play_capture.player_source_provenance(tmp_path / "missing")


def test_checkout_preserves_git_commit_and_dirty_state(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    package = root / "src/gradlab"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# source\n")
    monkeypatch.setattr(play_capture, "__file__", str(package / "play_capture.py"))

    def git(command, **kwargs):
        assert kwargs["cwd"] == root
        return SimpleNamespace(
            stdout=("a" * 40 + "\n")
            if command[1] == "rev-parse"
            else " M src/gradlab/__init__.py\n"
        )

    monkeypatch.setattr(play_capture.subprocess, "run", git)
    result = play_capture.player_source_provenance(root)
    assert result["kind"] == "checkout"
    assert result["git_commit"] == "a" * 40
    assert result["working_tree_dirty"] is True
    assert play_capture.player_source_provenance() == result
