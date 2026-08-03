from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from gradlab.eval_backend import EvalHandle
from gradlab.file_utils import file_sha256
from gradlab.modal_eval_backend import ModalEvalBackend
from gradlab.modal_eval_config import load_modal_eval_config, modal_app_name
from gradlab.modal_eval_protocol import SEED_PROTOCOL, build_execution_contract
from gradlab.modal_eval_storage import write_downloaded_file
from gradlab.modal_eval_worker import _prepare_vizdoom_iwad, execute_attempt
from gradlab.r2_store import PUBLIC_OBJECT_USER_AGENT
from gradlab.vizdoom_assets import (
    verify_vizdoom_iwad_file,
    vizdoom_iwad_binding,
    vizdoom_iwad_cache_path,
)


ROOT = Path(__file__).resolve().parents[1]


def _contract(model: Path) -> dict:
    return build_execution_contract(
        checkpoint_sha256=file_sha256(model),
        runtime_image_ref="docker:example.invalid/gradlab@sha256:" + "b" * 64,
        eval_environment={"env_provider": "gradlab", "game": "Bandit-v0", "task": {}},
        episodes=2,
        n_envs=2,
        watchdog_steps=100,
        seed=10_000,
        seed_protocol=SEED_PROTOCOL,
        asset_manifest=None,
        recipe_sha256="c" * 64,
        recipe_format_version=1,
        evaluation_contract_sha256="d" * 64,
    )


def test_checked_in_modal_contract_is_cold_and_cost_bounded() -> None:
    config = load_modal_eval_config(ROOT / "experiments" / "modal_eval.yaml")

    assert config.resources.cpu == 8
    assert config.resources.memory_mib == 4096
    assert config.resources.min_containers == 0
    assert config.resources.buffer_containers == 0
    assert config.resources.max_containers == 10
    assert config.protocol.max_attempts == 2


def test_modal_contract_rejects_noncurrent_enabled_flag(tmp_path: Path) -> None:
    source = ROOT / "experiments" / "modal_eval.yaml"
    path = tmp_path / "modal_eval.yaml"
    path.write_text(f"enabled: true\n{source.read_text(encoding='utf-8')}", encoding="utf-8")

    with pytest.raises(ValueError, match=r"unknown field\(s\): enabled"):
        load_modal_eval_config(path)


def test_modal_app_name_is_immutable_per_source() -> None:
    assert modal_app_name("gradlab-eval-v3", "a" * 40) == "gradlab-eval-v3-aaaaaaaaaaaa"
    with pytest.raises(ValueError, match="full lowercase Git SHA"):
        modal_app_name("gradlab-eval-v3", "main")


def test_modal_download_uses_explicit_gradlab_user_agent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[str] = []

    class Response:
        def __init__(self):
            self.payload = b"checkpoint"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _size: int = -1) -> bytes:
            payload, self.payload = self.payload, b""
            return payload

    def urlopen(request, *, timeout):
        assert timeout == 60
        assert isinstance(request, urllib.request.Request)
        observed.append(str(request.get_header("User-agent")))
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    target = write_downloaded_file(
        "https://models.example.test/model.zip",
        tmp_path / "model.zip",
    )

    assert target.read_bytes() == b"checkpoint"
    assert observed == [PUBLIC_OBJECT_USER_AGENT]


def test_modal_eval_installs_the_contract_bound_vizdoom_iwad(tmp_path: Path) -> None:
    source = tmp_path / "source" / "doom2.wad"
    source.parent.mkdir()
    source.write_bytes(b"IWADdoom")
    binding = vizdoom_iwad_binding(source)
    contract = {
        "environment": {
            "env_provider": "vizdoom-turbo",
            "game": "VizdoomBasic-v1",
            "env_args": {"rom_path": binding},
        }
    }
    cache_root = tmp_path / "cache"

    normalized = _prepare_vizdoom_iwad(
        {
            "vizdoom_iwad_binding": binding,
            "vizdoom_iwad_get_url": source.as_uri(),
        },
        contract,
        cache_root=cache_root,
        root=tmp_path / "attempt",
    )

    assert normalized == binding
    cached = vizdoom_iwad_cache_path(cache_root, binding)
    assert verify_vizdoom_iwad_file(cached, binding) == cached.resolve()

    with pytest.raises(ValueError, match="differs from the evaluation contract"):
        _prepare_vizdoom_iwad(
            {
                "vizdoom_iwad_binding": {**binding, "sha256": "0" * 64},
                "vizdoom_iwad_get_url": source.as_uri(),
            },
            contract,
            cache_root=cache_root,
            root=tmp_path / "mismatch",
        )


def test_backend_uses_spawn_poll_and_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, object]] = []

    class Function:
        @staticmethod
        def from_name(app, function, *, environment_name):
            calls.append(("from_name", (app, function, environment_name)))
            return SimpleNamespace(
                spawn=lambda payload: (
                    calls.append(("spawn", payload)) or SimpleNamespace(object_id="fc-1")
                )
            )

    class FunctionCall:
        @staticmethod
        def from_id(call_id):
            calls.append(("from_id", call_id))
            return SimpleNamespace(
                get=lambda *, timeout: {"ok": timeout == 0},
                cancel=lambda: calls.append(("cancel", call_id)),
            )

    fake_modal = SimpleNamespace(Function=Function, FunctionCall=FunctionCall)
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    backend = ModalEvalBackend(
        app_name="gradlab-eval-v3-aaaaaaaaaaaa",
        environment_name="gradlab-eval",
    )

    handle = backend.submit({"intent": "value"})
    assert handle == EvalHandle(provider="modal", call_id="fc-1")
    assert backend.poll(handle).provider_result == {"ok": True}
    backend.cancel(handle)
    assert ("spawn", {"intent": "value"}) in calls
    assert ("cancel", "fc-1") in calls


def test_expired_attempt_persists_create_only_result_before_download(tmp_path: Path) -> None:
    model = tmp_path / "model.zip"
    model.write_bytes(b"checkpoint")
    result = tmp_path / "result.json"
    payload = {
        "attempt_id": "attempt-1",
        "contract": _contract(model),
        "expires_at": time.time() - 1,
        "child_timeout_seconds": 10,
        "model_get_url": (tmp_path / "missing.zip").as_uri(),
        "result_uri": result.as_uri(),
        "result_put_url": result.as_uri(),
    }

    returned = execute_attempt(payload, cache_root=tmp_path / "cache")
    document = json.loads(result.read_text(encoding="utf-8"))
    assert document["status"] == "expired"
    assert returned["result_uri"] == result.as_uri()

    with pytest.raises(RuntimeError, match="different content"):
        execute_attempt({**payload, "attempt_id": "attempt-2"}, cache_root=tmp_path / "cache")
