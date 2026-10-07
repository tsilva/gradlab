from __future__ import annotations

import base64
import os
import subprocess
from pathlib import Path

import pytest

from gradlab.mlflow_tls import configure_private_mlflow_ca


def test_private_mlflow_ca_extends_system_trust_bundle(monkeypatch, tmp_path: Path) -> None:
    system_bundle = tmp_path / "system.pem"
    system_bundle.write_bytes(b"existing system trust\n")
    monkeypatch.setattr("gradlab.mlflow_tls._system_ca_bundle", lambda: system_bundle)
    key = tmp_path / "key.pem"
    cert = tmp_path / "ca.pem"
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-keyout", str(key), "-out", str(cert), "-days", "1",
            "-subj", "/CN=GradLab test CA",
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    monkeypatch.setenv(
        "GRADLAB_MLFLOW_TLS_CA_B64", base64.b64encode(cert.read_bytes()).decode("ascii")
    )
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "")
    monkeypatch.setenv("SSL_CERT_FILE", "")
    bundle = configure_private_mlflow_ca()
    assert bundle is not None
    assert bundle.read_bytes().startswith(system_bundle.read_bytes())
    assert cert.read_bytes() in bundle.read_bytes()
    assert os.stat(bundle).st_mode & 0o077 == 0
    assert os.environ["REQUESTS_CA_BUNDLE"] == str(bundle)
    assert os.environ["SSL_CERT_FILE"] == str(bundle)
    assert "GRADLAB_MLFLOW_TLS_CA_B64" not in os.environ
    bundle.unlink()


def test_private_mlflow_ca_rejects_invalid_input(monkeypatch) -> None:
    monkeypatch.setenv("GRADLAB_MLFLOW_TLS_CA_B64", "not-base64")
    with pytest.raises(ValueError, match="one PEM certificate"):
        configure_private_mlflow_ca()


@pytest.mark.parametrize('fail', [False, True])
def test_preflight_ca_context_restores_binding_and_removes_bundle(monkeypatch, tmp_path, fail):
    from gradlab.mlflow_tls import private_mlflow_ca_context

    bundle = tmp_path / 'temporary.pem'
    bundle.write_bytes(b'test trust bundle')
    monkeypatch.setenv('REQUESTS_CA_BUNDLE', 'original.pem')
    monkeypatch.delenv('SSL_CERT_FILE', raising=False)
    monkeypatch.setenv('GRADLAB_MLFLOW_TLS_CA_B64', 'public-ca-binding')

    def configure():
        os.environ['REQUESTS_CA_BUNDLE'] = str(bundle)
        os.environ['SSL_CERT_FILE'] = str(bundle)
        os.environ.pop('GRADLAB_MLFLOW_TLS_CA_B64')
        return bundle

    monkeypatch.setattr('gradlab.mlflow_tls.configure_private_mlflow_ca', configure)
    try:
        with private_mlflow_ca_context():
            assert os.environ['REQUESTS_CA_BUNDLE'] == str(bundle)
            if fail:
                raise RuntimeError('service unavailable')
    except RuntimeError:
        assert fail
    assert os.environ['REQUESTS_CA_BUNDLE'] == 'original.pem'
    assert 'SSL_CERT_FILE' not in os.environ
    assert os.environ['GRADLAB_MLFLOW_TLS_CA_B64'] == 'public-ca-binding'
    assert not bundle.exists()
