from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

import pytest

from gradlab.local_train import (
    LOCAL_ROM_CACHE_ENV,
    _play_uvx_launcher,
    _should_use_training_tui,
    build_parser as build_local_train_parser,
    main,
)
from gradlab.play import main as play_main
from gradlab.play_runtime import (
    resolve_playback_rom_binding,
    resolve_shared_playback_rom_binding,
)
from gradlab.recipe_catalog import (
    LOCAL_RUN_RECEIPT,
    latest_local_recipe_model,
    resolve_recipe_source,
)
from gradlab.train import INTERNAL_LEARNER_ENV
from gradlab.training_lifecycle import TRAINING_RESULT_FILENAME
from gradlab.rom_runtime import RomRuntimeBinding


def _write_training_result(
    run_dir: Path,
    *,
    status: str = "completed",
    terminal_reason: str = "resource_exhaustion",
    model_kind: str = "final",
    step: int = 64,
) -> None:
    (run_dir / TRAINING_RESULT_FILENAME).write_text(
        json.dumps(
            {
                "document_type": "gradlab.training-result",
                "format_version": 2,
                "status": status,
                "terminal_reason": terminal_reason,
                "execution_mode": "local-demo",
                "execution_policy": {
                    "mode": "local-demo",
                    "console_mode": "auto",
                    "persist_intermediate_checkpoints": False,
                    "stop_on_first_completion": True,
                    "handle_sigint": True,
                },
                "first_completion_step": None,
                "final_step": step,
                "requested_limit": 64,
                "execution_limit": 64,
                "model_kind": model_kind,
                "model": "final_model.zip",
            }
        ),
        encoding="utf-8",
    )


def test_local_train_parser_exposes_only_rom_path() -> None:
    parser = build_local_train_parser()

    assert parser.parse_args(["goal/recipe", "--rom-path", "/tmp/game.nes"]).rom_path == Path(
        "/tmp/game.nes"
    )
    with pytest.raises(SystemExit):
        parser.parse_args(["goal/recipe", "--rom", "/tmp/game.nes"])


def test_builtin_recipe_reference_resolves_goal_and_recipe() -> None:
    source = resolve_recipe_source("SuperMarioBros-Nes-v0/Level1-1/ppo")

    assert source.reference == "SuperMarioBros-Nes-v0/Level1-1/ppo"
    assert source.goal_path.name == "_goal.yaml"
    assert source.goal_path.parent.name == "Level1-1"
    assert source.recipe_path.name == "ppo.yaml"
    assert source.recipe_path.parent.name == "recipes"


def test_local_recipe_path_infers_owning_goal() -> None:
    source = resolve_recipe_source(Path("experiments/goals/VizdoomBasic-v1/recipes/ppo.yaml"))

    assert source.goal_path == Path("experiments/goals/VizdoomBasic-v1/_goal.yaml").resolve()
    assert source.repository_root == Path.cwd()


def test_turbo_demo_uses_the_standard_mario_ppo_contract() -> None:
    source = resolve_recipe_source("SuperMarioBros-Nes-v0/Level1-1/turbo-demo")
    from gradlab.recipe_documents import compose_train_document

    document = compose_train_document(source.goal_path, source.recipe_path)
    config = document["train_config"]
    backend = config["training_backend"]

    assert config["timesteps"] == 98_304
    assert config["n_envs"] == 16
    assert backend["id"] == "sb3.ppo"
    assert backend["config"]["batch_size"] == 512
    assert config["early_stop"]["conditions"]["return_plateau"]["outcome"] == "failure"
    assert config["reward_shape"] == "speedrun-v1"


def test_play_launcher_reuses_editable_local_uvx_source(tmp_path: Path) -> None:
    with (
        mock.patch(
            "gradlab.local_train._distribution_direct_url",
            return_value={
                "url": tmp_path.resolve().as_uri(),
                "dir_info": {"editable": True},
            },
        ),
        mock.patch(
            "gradlab.local_train._source_distribution",
            return_value={"name": "gradlab", "version": "9.8.7"},
        ),
    ):
        launcher = _play_uvx_launcher()

    assert launcher == [
        "uvx",
        "--from",
        str(tmp_path.resolve()),
        "--with-editable",
        str(tmp_path.resolve()),
        "--refresh-package",
        "gradlab",
        "gradlab",
    ]


def test_play_launcher_pins_published_distribution() -> None:
    with (
        mock.patch(
            "gradlab.local_train._distribution_direct_url",
            return_value={},
        ),
        mock.patch(
            "gradlab.local_train._source_distribution",
            return_value={"name": "gradlab", "version": "9.8.7"},
        ),
    ):
        launcher = _play_uvx_launcher()

    assert launcher == ["uvx", "gradlab@9.8.7"]


def test_local_tui_requires_usable_input_and_output_ttys(monkeypatch) -> None:
    tty = mock.Mock()
    tty.isatty.return_value = True
    monkeypatch.setenv("TERM", "xterm-256color")
    with (
        mock.patch("gradlab.local_train.sys.stdin", tty),
        mock.patch("gradlab.local_train.sys.stdout", tty),
    ):
        assert _should_use_training_tui(disabled=False) is True
        assert _should_use_training_tui(disabled=True) is False

    monkeypatch.setenv("TERM", "dumb")
    with (
        mock.patch("gradlab.local_train.sys.stdin", tty),
        mock.patch("gradlab.local_train.sys.stdout", tty),
    ):
        assert _should_use_training_tui(disabled=False) is False


def test_local_train_rejects_non_main_thread_before_recipe_resolution() -> None:
    with (
        mock.patch("gradlab.local_train.threading.current_thread", return_value=object()),
        mock.patch("gradlab.local_train.resolve_recipe_source") as resolve,
        pytest.raises(RuntimeError, match="main thread"),
    ):
        main(["gradlab__bandit/ppo"])

    resolve.assert_not_called()


def test_local_train_materializes_credential_free_playable_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    observed_internal_values: list[str | None] = []

    def fake_learner(
        argv: list[str],
        *,
        runtime_rom_binding=None,
    ) -> int:
        assert runtime_rom_binding is None
        observed_internal_values.append(os.environ.get(INTERNAL_LEARNER_ENV))
        assert argv[argv.index("--execution-mode") + 1] == "local-demo"
        config_path = Path(argv[argv.index("--train-config-json") + 1])
        config = json.loads(config_path.read_text(encoding="utf-8"))
        run_dir = Path(config["runs_dir"]) / config["run_name"]
        (run_dir / "final_model.zip").write_bytes(b"model")
        _write_training_result(run_dir)
        return 0

    monkeypatch.delenv(INTERNAL_LEARNER_ENV, raising=False)
    with mock.patch("gradlab.train.main", side_effect=fake_learner):
        assert (
            main(
                [
                    "gradlab__bandit/ppo",
                    "--runs-dir",
                    str(tmp_path),
                    "--run-name",
                    "smoke",
                    "--set",
                    "train.timesteps=64",
                    "--no-tui",
                ]
            )
            == 0
        )

    assert observed_internal_values == ["1"]
    assert INTERNAL_LEARNER_ENV not in os.environ
    run_dir = tmp_path / "smoke"
    config = json.loads((run_dir / "train-config.json").read_text(encoding="utf-8"))
    recipe = json.loads((run_dir / "recipe.json").read_text(encoding="utf-8"))
    receipt = json.loads((run_dir / LOCAL_RUN_RECEIPT).read_text(encoding="utf-8"))

    assert config["checkpoint_eval_backend"] == "none"
    assert "stop_on_acceptance" not in config
    assert "wandb" not in config
    assert config["wandb_mode"] == "disabled"
    assert config["timesteps"] == 64
    assert config["run_description"] == (
        "PPO smoke test for the ROM-free native-vector gradlab__bandit target."
    )
    assert "image_ref" not in recipe["provenance"]["runtime"]
    assert recipe["provenance"]["runtime"]["packages"]
    assert recipe["provenance"]["source_distribution"]["name"].lower() == "gradlab"
    assert receipt["status"] == "completed"
    assert receipt["model"] == "final_model.zip"
    assert receipt["training_execution"]["mode"] == "local-demo"
    assert receipt["final_step"] == 64
    assert receipt["requested_limit"] == 64
    assert receipt["execution_limit"] == 64
    assert "training_execution" not in recipe["recipe"]["train_config"]
    assert "no_tui" not in config
    assert "no-tui" not in json.dumps(recipe)
    assert "no-tui" not in json.dumps(receipt)


def test_local_interruption_writes_terminal_receipt_and_is_not_auto_selected(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_learner(argv: list[str], **_kwargs) -> int:
        config_path = Path(argv[argv.index("--train-config-json") + 1])
        config = json.loads(config_path.read_text(encoding="utf-8"))
        run_dir = Path(config["runs_dir"]) / config["run_name"]
        (run_dir / "final_model.zip").write_bytes(b"interrupted-model")
        _write_training_result(
            run_dir,
            status="interrupted",
            terminal_reason="local_interruption",
            model_kind="interrupted",
            step=32,
        )
        return 0

    with mock.patch("gradlab.train.main", side_effect=fake_learner):
        assert (
            main(
                [
                    "gradlab__bandit/ppo",
                    "--runs-dir",
                    str(tmp_path),
                    "--run-name",
                    "interrupted",
                    "--set",
                    "train.timesteps=64",
                ]
            )
            == 130
        )

    receipt = json.loads((tmp_path / "interrupted" / LOCAL_RUN_RECEIPT).read_text(encoding="utf-8"))
    assert receipt["status"] == "interrupted"
    assert receipt["terminal_reason"] == "local_interruption"
    assert receipt["model"] == "final_model.zip"
    with pytest.raises(FileNotFoundError, match="no completed local model"):
        latest_local_recipe_model(
            tmp_path,
            goal_id="gradlab__bandit",
            recipe_id="ppo",
        )
    assert "play interrupted model:" in capsys.readouterr().out


def test_local_failure_replaces_running_receipt(
    tmp_path: Path,
) -> None:
    with (
        mock.patch(
            "gradlab.train.main",
            side_effect=RuntimeError("learner exploded"),
        ),
        pytest.raises(RuntimeError, match="learner exploded"),
    ):
        main(
            [
                "gradlab__bandit/ppo",
                "--runs-dir",
                str(tmp_path),
                "--run-name",
                "failed",
                "--set",
                "train.timesteps=64",
            ]
        )

    receipt = json.loads((tmp_path / "failed" / LOCAL_RUN_RECEIPT).read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["terminal_reason"] == "failed"
    assert receipt["error_type"] == "RuntimeError"


def test_local_mario_train_binds_registered_rom_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest = {
        "schema_version": 2,
        "game": "SuperMarioBros-Nes-v0",
        "filename": "mario.nes",
        "size_bytes": 1,
        "sha256": "a" * 64,
        "object_uri": "file:///tmp/mario.nes",
        "provider_rom_identity": "b" * 40,
        "provider_rom_identity_algorithm": "sha1-provider-body-v1",
    }
    observed_cache: list[str | None] = []

    def fake_learner(
        argv: list[str],
        *,
        runtime_rom_binding=None,
    ) -> int:
        assert runtime_rom_binding is None
        observed_cache.append(os.environ.get(LOCAL_ROM_CACHE_ENV))
        assert argv[argv.index("--execution-mode") + 1] == "local-demo"
        config_path = Path(argv[argv.index("--train-config-json") + 1])
        config = json.loads(config_path.read_text(encoding="utf-8"))
        assert config["rom_asset_manifest"] == manifest
        run_dir = Path(config["runs_dir"]) / config["run_name"]
        (run_dir / "final_model.zip").write_bytes(b"model")
        _write_training_result(run_dir)
        return 0

    monkeypatch.setenv(LOCAL_ROM_CACHE_ENV, "restore-me")
    with (
        mock.patch(
            "gradlab.local_train.rom_asset_manifest_for_game",
            return_value=manifest,
        ),
        mock.patch("gradlab.train.main", side_effect=fake_learner),
    ):
        assert (
            main(
                [
                    "SuperMarioBros-Nes-v0/Level1-1/dstack-smoke",
                    "--runs-dir",
                    str(tmp_path),
                    "--run-name",
                    "mario",
                ]
            )
            == 0
        )

    assert observed_cache and observed_cache[0] != "restore-me"
    assert os.environ[LOCAL_ROM_CACHE_ENV] == "restore-me"
    recipe = json.loads((tmp_path / "mario" / "recipe.json").read_text(encoding="utf-8"))
    assert recipe["provenance"]["asset"]["sha256"] == manifest["sha256"]


def test_local_mario_train_uses_direct_rom_without_registry_or_cache_mutation(
    tmp_path: Path,
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rom = tmp_path / "SuperMarioBros.nes"
    rom.write_bytes(b"rom")
    manifest = {
        "schema_version": 2,
        "game": "SuperMarioBros-Nes-v0",
        "filename": rom.name,
        "size_bytes": 3,
        "sha256": "a" * 64,
        "object_uri": rom.resolve().as_uri(),
        "provider_rom_identity": "b" * 40,
        "provider_rom_identity_algorithm": "sha1-provider-body-v1",
    }
    binding = RomRuntimeBinding(manifest=manifest, path=rom.resolve())
    observed_bindings: list[RomRuntimeBinding] = []

    def fake_learner(
        argv: list[str],
        *,
        runtime_rom_binding: RomRuntimeBinding,
    ) -> int:
        observed_bindings.append(runtime_rom_binding)
        assert argv[argv.index("--execution-mode") + 1] == "local-demo"
        config_path = Path(argv[argv.index("--train-config-json") + 1])
        config = json.loads(config_path.read_text(encoding="utf-8"))
        assert config["rom_asset_manifest"] == manifest
        run_dir = Path(config["runs_dir"]) / config["run_name"]
        (run_dir / "final_model.zip").write_bytes(b"model")
        _write_training_result(run_dir)
        return 0

    monkeypatch.setenv(LOCAL_ROM_CACHE_ENV, "unchanged")
    with (
        mock.patch(
            "gradlab.local_train.direct_rom_asset_manifest",
            return_value=manifest,
        ),
        mock.patch(
            "gradlab.local_train.bind_rom_path",
            return_value=binding,
        ),
        mock.patch(
            "gradlab.local_train.rom_asset_manifest_for_game",
            side_effect=AssertionError("direct ROM consulted registry"),
        ),
        mock.patch("gradlab.train.main", side_effect=fake_learner),
    ):
        assert (
            main(
                [
                    "SuperMarioBros-Nes-v0/Level1-1/turbo-demo",
                    "--rom-path",
                    str(rom),
                    "--runs-dir",
                    str(tmp_path / "runs"),
                    "--run-name",
                    "mario-direct",
                ]
            )
            == 0
        )

    assert observed_bindings == [binding]
    assert os.environ[LOCAL_ROM_CACHE_ENV] == "unchanged"
    recipe = json.loads(
        (tmp_path / "runs" / "mario-direct" / "recipe.json").read_text(encoding="utf-8")
    )
    assert recipe["provenance"]["asset"]["sha256"] == manifest["sha256"]
    assert recipe["provenance"]["asset"]["size_bytes"] == manifest["size_bytes"]
    assert (
        recipe["provenance"]["asset"]["provider_rom_identity"]
        == (manifest["provider_rom_identity"])
    )
    assert str(rom.resolve()) not in json.dumps(recipe)
    output = capsys.readouterr().out
    assert f"uvx --from {Path.cwd()} --with-editable {Path.cwd()}" in output
    assert "--refresh-package gradlab gradlab play" in output
    assert f"--rom-path {rom.resolve()}" in output


def test_local_train_rejects_rom_for_rom_free_provider_before_creating_run(
    tmp_path: Path,
) -> None:
    rom = tmp_path / "unused.nes"
    rom.write_bytes(b"rom")

    with pytest.raises(ValueError, match="ROM-free provider"):
        main(
            [
                "gradlab__bandit/ppo",
                "--rom-path",
                str(rom),
                "--runs-dir",
                str(tmp_path),
                "--run-name",
                "must-not-exist",
            ]
        )

    assert not (tmp_path / "must-not-exist").exists()


def test_local_mario_direct_rom_failure_precedes_run_creation(tmp_path: Path) -> None:
    rom = tmp_path / "wrong.nes"
    rom.write_bytes(b"rom")

    with (
        mock.patch(
            "gradlab.local_train.direct_rom_asset_manifest",
            side_effect=ValueError("ROM does not match provider identity"),
        ),
        pytest.raises(ValueError, match="provider identity"),
    ):
        main(
            [
                "SuperMarioBros-Nes-v0/Level1-1/turbo-demo",
                "--rom-path",
                str(rom),
                "--runs-dir",
                str(tmp_path),
                "--run-name",
                "must-not-exist",
            ]
        )

    assert not (tmp_path / "must-not-exist").exists()


def test_local_mario_train_rejects_missing_rom_before_creating_run(
    tmp_path: Path,
) -> None:
    with (
        mock.patch(
            "gradlab.local_train.rom_asset_manifest_for_game",
            side_effect=ValueError("run: gradlab rom sync"),
        ),
        pytest.raises(ValueError, match="gradlab rom sync"),
    ):
        main(
            [
                "SuperMarioBros-Nes-v0/Level1-1/dstack-smoke",
                "--runs-dir",
                str(tmp_path),
                "--run-name",
                "must-not-exist",
            ]
        )

    assert not (tmp_path / "must-not-exist").exists()


def test_playback_direct_rom_matches_model_without_registry_lookup(tmp_path: Path) -> None:
    rom = tmp_path / "mario.nes"
    rom.write_bytes(b"rom")
    manifest = {
        "schema_version": 2,
        "game": "SuperMarioBros-Nes-v0",
        "filename": rom.name,
        "size_bytes": 3,
        "sha256": "a" * 64,
        "object_uri": rom.resolve().as_uri(),
        "provider_rom_identity": "b" * 40,
        "provider_rom_identity_algorithm": "sha1-provider-body-v1",
    }
    portable = {key: value for key, value in manifest.items() if key != "object_uri"}
    binding = RomRuntimeBinding(manifest=portable, path=rom.resolve())

    with (
        mock.patch(
            "gradlab.play_runtime.direct_rom_asset_manifest",
            return_value=manifest,
        ),
        mock.patch(
            "gradlab.play_runtime.bind_rom_path",
            return_value=binding,
        ) as bind,
        mock.patch(
            "gradlab.play_runtime.rom_asset_manifest_for_game",
            side_effect=AssertionError("direct playback consulted registry"),
        ),
    ):
        resolved = resolve_playback_rom_binding(
            env_provider="supermariobrosnes-turbo",
            game="SuperMarioBros-Nes-v0",
            asset=portable,
            rom_path=rom,
        )

    assert resolved == binding
    bind.assert_called_once_with(portable, rom)


def test_playback_direct_rom_rejects_model_mismatch_and_rom_free_provider(
    tmp_path: Path,
) -> None:
    rom = tmp_path / "mario.nes"
    rom.write_bytes(b"rom")
    manifest = {
        "schema_version": 2,
        "game": "SuperMarioBros-Nes-v0",
        "filename": rom.name,
        "size_bytes": 3,
        "sha256": "a" * 64,
        "object_uri": rom.resolve().as_uri(),
        "provider_rom_identity": "b" * 40,
        "provider_rom_identity_algorithm": "sha1-provider-body-v1",
    }

    with (
        mock.patch(
            "gradlab.play_runtime.direct_rom_asset_manifest",
            return_value=manifest,
        ),
        pytest.raises(ValueError, match="recorded by the model"),
    ):
        resolve_playback_rom_binding(
            env_provider="supermariobrosnes-turbo",
            game="SuperMarioBros-Nes-v0",
            asset={**manifest, "sha256": "c" * 64},
            rom_path=rom,
        )

    with pytest.raises(ValueError, match="ROM-free provider"):
        resolve_playback_rom_binding(
            env_provider="gradlab",
            game="gradlab__bandit-v0",
            asset=None,
            rom_path=rom,
        )


def test_shared_player_ignores_rom_option_for_rom_free_checkpoint(tmp_path: Path) -> None:
    rom = tmp_path / "mario.nes"
    rom.write_bytes(b"rom")

    with mock.patch(
        "gradlab.play_runtime.direct_rom_asset_manifest",
        side_effect=AssertionError("ROM-free playback inspected the shared ROM"),
    ):
        resolved = resolve_shared_playback_rom_binding(
            env_provider="vizdoom-turbo",
            game="VizdoomDeadlyCorridor-v1",
            asset=None,
            rom_path=rom,
        )

    assert resolved is None


def test_playback_without_direct_rom_preserves_registered_asset_fallback() -> None:
    binding = mock.sentinel.binding
    with (
        mock.patch(
            "gradlab.play_runtime.rom_asset_manifest_for_game",
            return_value={"registered": True},
        ) as registered,
        mock.patch(
            "gradlab.play_runtime.ensure_local_rom_binding",
            return_value=binding,
        ) as ensure,
    ):
        resolved = resolve_playback_rom_binding(
            env_provider="supermariobrosnes-turbo",
            game="SuperMarioBros-Nes-v0",
            asset=None,
            rom_path=None,
        )

    assert resolved is binding
    registered.assert_called_once_with("SuperMarioBros-Nes-v0")
    ensure.assert_called_once_with(
        {"registered": True},
        game="SuperMarioBros-Nes-v0",
    )


def test_latest_local_recipe_model_uses_newest_completed_receipt(tmp_path: Path) -> None:
    for name, completed_at in (
        ("older", "2026-07-27T10:00:00Z"),
        ("newer", "2026-07-27T11:00:00Z"),
    ):
        run_dir = tmp_path / name
        run_dir.mkdir()
        (run_dir / "final_model.zip").write_bytes(name.encode())
        (run_dir / LOCAL_RUN_RECEIPT).write_text(
            json.dumps(
                {
                    "status": "completed",
                    "goal_id": "VizdoomBasic-v1",
                    "recipe_id": "ppo",
                    "model": "final_model.zip",
                    "completed_at": completed_at,
                }
            ),
            encoding="utf-8",
        )

    assert latest_local_recipe_model(
        tmp_path,
        goal_id="VizdoomBasic-v1",
        recipe_id="ppo",
    ) == (tmp_path / "newer" / "final_model.zip")


def test_play_recipe_selects_latest_local_model(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model_path = run_dir / "final_model.zip"
    model_path.write_bytes(b"model")
    (run_dir / LOCAL_RUN_RECEIPT).write_text(
        json.dumps(
            {
                "status": "completed",
                "goal_id": "gradlab__bandit",
                "recipe_id": "ppo",
                "model": "final_model.zip",
                "completed_at": "2026-07-27T11:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    with mock.patch(
        "gradlab.play_web.run_web_player_application",
        return_value=23,
    ) as run_application:
        assert (
            play_main(
                [
                    "--recipe",
                    "gradlab__bandit/ppo",
                    "--runs-dir",
                    str(tmp_path),
                    "--no-open",
                ]
            )
            == 23
        )

    host, args = run_application.call_args.args[:2]
    assert args.model == str(model_path)
    assert host.snapshot()["app"]["source"]["kind"] == "local"
    assert host.snapshot()["app"]["source"]["value"] == str(model_path)
