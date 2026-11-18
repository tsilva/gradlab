from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "upload_youtube_video.py"
SPEC = importlib.util.spec_from_file_location("upload_youtube_video", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_replace_description_link_block_preserves_existing_copy() -> None:
    existing = """A PPO agent completes Level 1-4.

Model: https://huggingface.co/tsilva/old
gradlab: https://github.com/tsilva/gradlab

#ReinforcementLearning #PPO #SuperMarioBros"""

    updated = MODULE.replace_description_link_block(
        existing,
        "Model: https://huggingface.co/tsilva/new\ngradlab: https://github.com/tsilva/gradlab",
    )

    assert "A PPO agent completes Level 1-4." in updated
    assert "https://huggingface.co/tsilva/new" in updated
    assert "https://huggingface.co/tsilva/old" not in updated
    assert updated.count("Model:") == 1
    assert updated.count("gradlab:") == 1
    assert updated.endswith("#ReinforcementLearning #PPO #SuperMarioBros")


def test_replace_description_link_block_rejects_unrelated_lines() -> None:
    with pytest.raises(ValueError, match="only Model: and gradlab:"):
        MODULE.replace_description_link_block("existing", "Title: not allowed")


def test_replace_description_link_block_accepts_a_direct_model_url() -> None:
    updated = MODULE.replace_description_link_block(
        "Model: https://huggingface.co/tsilva/old",
        "https://huggingface.co/tsilva/new",
    )

    assert updated == "Model: https://huggingface.co/tsilva/new"


def test_parser_defaults_youtube_credentials_to_gradlab_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)

    args = MODULE.build_parser().parse_args([])

    config = (tmp_path / ".config" / "gradlab").resolve()
    assert args.client_secret == config / "youtube_client_secret.json"
    assert args.token == config / "youtube_token.json"


def test_access_token_merges_client_config_into_player_issued_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token_path = tmp_path / "youtube_token.json"
    token_path.write_text(json.dumps({"refresh_token": "refresh"}), encoding="utf-8")
    observed: dict[str, str] = {}

    def fake_refresh(token: dict[str, str]) -> dict[str, str]:
        observed.update(token)
        return {**token, "access_token": "access"}

    monkeypatch.setattr(MODULE, "refresh_token", fake_refresh)

    value = MODULE.access_token(
        {
            "client_id": "client",
            "client_secret": "secret",
            "token_uri": "https://oauth.example/token",
        },
        token_path,
        no_browser=True,
    )

    assert value == "access"
    assert observed["client_id"] == "client"
    assert observed["client_secret"] == "secret"
    assert observed["token_uri"] == "https://oauth.example/token"


def test_auth_only_never_uploads_or_changes_videos(monkeypatch):
    calls = []
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "--auth-only"])
    monkeypatch.setattr(MODULE, "youtube_store", lambda _: object())
    monkeypatch.setattr(MODULE, "load_client_config", lambda _: {"client_id": "client"})
    monkeypatch.setattr(MODULE, "authorize", lambda *args, **kw: calls.append(kw))
    monkeypatch.setattr(MODULE, "access_token", lambda *args, **kw: 1 / 0)
    monkeypatch.setattr(MODULE, "upload_video", lambda *args, **kw: 1 / 0)
    MODULE.main()
    assert calls == [{"no_browser": False}]


@pytest.mark.parametrize("case", ["success", "wrong-channel", "external-location", "wrong-offset", "redirect"])
def test_external_relay_uses_scoped_routes_and_never_loads_oauth(tmp_path, monkeypatch, capsys, case):
    import json
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    from gradlab import youtube_publication

    channel = "UC" + "a" * 22
    handle = "/gradlab-upload/" + "a" * 48
    requests = []

    class Relay(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def reply(self, status, value=None, **headers):
            data = json.dumps(value).encode() if value is not None else b""
            self.send_response(status)
            for name, header in headers.items():
                self.send_header(name, header)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def admit(self):
            assert self.headers.get("Authorization") == "Bearer gradlab_youtube_proxy_placeholder"
            requests.append((self.command, self.path, self.headers.get("Content-Range")))

        def do_GET(self):
            self.admit()
            assert self.path == "/youtube/v3/channels?part=snippet&mine=true"
            self.reply(200, {"items": [{"id": "UC" + "b" * 22 if case == "wrong-channel" else channel}]})

        def do_POST(self):
            self.admit()
            assert self.path == "/upload/youtube/v3/videos?uploadType=resumable&part=snippet%2Cstatus"
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert payload["snippet"]["title"] == "Synthetic replay"
            location = "https://example.invalid/private-upload" if case == "external-location" else origin + handle
            self.reply(302 if case == "redirect" else 200, Location=location)

        def do_PUT(self):
            self.admit()
            assert self.path == handle
            payload = self.rfile.read(int(self.headers["Content-Length"]))
            chunk = self.headers["Content-Range"]
            if chunk == "bytes 0-3/7":
                assert payload == b"abcd"
                self.reply(308, Range="bytes=0-2" if case == "wrong-offset" else "bytes=0-3")
            elif chunk == "bytes 4-6/7":
                assert payload == b"efg"
                self.reply(201, {"id": "fixture-video"})
            else:
                self.reply(400)

    with ThreadingHTTPServer(("127.0.0.1", 0), Relay) as relay:
        origin = f"http://127.0.0.1:{relay.server_port}"
        worker = Thread(target=relay.serve_forever)
        worker.start()
        try:
            monkeypatch.setenv("NO_PROXY", "127.0.0.1")
            monkeypatch.setenv("no_proxy", "127.0.0.1")
            monkeypatch.setenv("GRADLAB_YOUTUBE_RELAY", origin)
            monkeypatch.setenv("GRADLAB_YOUTUBE_CHANNEL_ID", channel)
            monkeypatch.setenv("GRADLAB_YOUTUBE_RELAY_TOKEN", "gradlab_youtube_proxy_placeholder")
            monkeypatch.setattr(youtube_publication, "UPLOAD_CHUNK_BYTES", 4)
            for name in ("youtube_store", "load_client_config", "access_token", "authorize"):
                monkeypatch.setattr(MODULE, name, lambda *_a, **_kw: pytest.fail("OAuth store used"))
            video = tmp_path / "replay.mp4"
            video.write_bytes(b"abcdefg")
            metadata = tmp_path / "metadata.json"
            metadata.write_text(json.dumps({"title": "Synthetic replay", "description": "Descrição de teste",
                                            "tags": [], "category_id": "20"}))
            output = tmp_path / "receipt.json"
            monkeypatch.setattr("sys.argv", [str(SCRIPT), str(video), "--metadata-file", str(metadata),
                                             "--no-thumbnail", "--output", str(output)])
            if case == "success":
                MODULE.main()
                assert json.loads(output.read_text())["id"] == "fixture-video"
                assert [operation for operation, _, _ in requests] == ["GET", "POST", "PUT", "PUT"]
                assert handle not in output.read_text()
                assert handle not in capsys.readouterr().out
            else:
                with pytest.raises(SystemExit) as error:
                    MODULE.main()
                assert len(requests) == {"wrong-channel": 1, "external-location": 2, "wrong-offset": 3, "redirect": 2}[case]
                assert not output.exists()
                assert "private-upload" not in str(error.value)
        finally:
            relay.shutdown()
            worker.join(timeout=5)
            assert not worker.is_alive()
