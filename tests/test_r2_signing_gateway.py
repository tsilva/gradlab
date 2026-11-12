import base64
import io
import threading
import urllib.error
import urllib.request
from datetime import UTC, datetime
from unittest.mock import patch

import pytest
from botocore.auth import S3SigV4Auth, S3SigV4QueryAuth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials

from gradlab.r2_signing_gateway import (
    GatewayServer, basic_credential, presigned_url, signed_headers, validate_target,
)


ACCESS = "AKIDEXAMPLE"
SECRET = "syntheticSigningSecret1234567890"
HOST = "0" * 32 + ".r2.cloudflarestorage.com"
NOW = datetime(2026, 11, 12, 12, 0, tzinfo=UTC)


@pytest.mark.parametrize("method,target,body", [
    ("PUT", "/models/checkpoint%20one.zip", b"synthetic policy"),
    ("GET", "/models?prefix=runs%2F&list-type=2&max-keys=10", b""),
    ("HEAD", "/models/run/model.zip", b""),
    ("DELETE", "/models/probe", b""),
])
def test_signing_matches_botocore_for_exact_s3_requests(method, target, body):
    headers = {"Content-Type": "application/octet-stream", "If-None-Match": "*",
               "X-Amz-Meta-Sha256": "a" * 64}
    actual = signed_headers(method, target, headers, body, access=ACCESS, secret=SECRET, host=HOST, now=NOW)
    reference = AWSRequest(method=method, url=f"https://{HOST}{target}", data=body, headers=headers)
    with patch("botocore.auth.get_current_datetime", return_value=NOW):
        S3SigV4Auth(Credentials(ACCESS, SECRET), "s3", "auto").add_auth(reference)
    assert actual["authorization"] == reference.headers["Authorization"]
    assert actual["x-amz-content-sha256"] == reference.headers["X-Amz-Content-SHA256"]


@pytest.mark.parametrize("method,content_type", [("GET", None), ("PUT", "application/json")])
def test_presigning_matches_botocore(method, content_type):
    from urllib.parse import parse_qs, urlsplit

    path = "/models/checkpoint%20one.zip"
    actual = presigned_url(method, path, access=ACCESS, secret=SECRET, host=HOST,
                           expires=1800, content_type=content_type, now=NOW)
    reference = AWSRequest(method=method, url=f"https://{HOST}{path}",
                           headers={"Content-Type": content_type} if content_type else {})
    with patch("botocore.auth.get_current_datetime", return_value=NOW):
        S3SigV4QueryAuth(Credentials(ACCESS, SECRET), "s3", "auto", expires=1800).add_auth(reference)
    assert parse_qs(urlsplit(actual).query) == parse_qs(urlsplit(reference.url).query)
    assert SECRET not in actual


@pytest.mark.parametrize("method,target", [
    ("PUT", "/models"), ("DELETE", "/models/"), ("POST", "/models?delete"),
    ("GET", "/other/key"), ("GET", "https://other.invalid/models/key"),
    ("GET", "/models/../other/key"), ("GET", "/models/%2e%2e/key"),
    ("GET", "/models/foo%2fbar"), ("GET", "/models/key#fragment"),
    ("GET", "/models/foo%zz"), ("GET", "/models/foo\\bar"),
])
def test_gateway_rejects_bucket_management_escape_and_noncanonical_paths(method, target):
    with pytest.raises(ValueError):
        validate_target(method, target, "models")


@pytest.mark.parametrize("value", ["", "Bearer fake", "Basic invalid", "Basic " + base64.b64encode(b"placeholder:placeholder").decode()])
def test_gateway_never_uses_placeholder_as_a_real_upstream_credential(value):
    with pytest.raises(ValueError):
        basic_credential(value)


class FakeResponse:
    def __init__(self, status, body, headers):
        self.status, self.body, self.headers = status, io.BytesIO(body), headers

    def getheaders(self):
        return list(self.headers.items())

    def read(self, size=-1):
        return self.body.read(size)


@pytest.mark.parametrize("status", [200, 403, 412])
def test_gateway_preserves_integrity_headers_and_redacts_upstream_error_bodies(monkeypatch, status):
    body = b"synthetic policy"
    observed = []

    class Upstream:
        def __init__(self, host, **kwargs):
            assert host == HOST

        def request(self, method, target, body, headers):
            observed.append((method, target, body, headers))

        def getresponse(self):
            response_body = body if status == 200 else (SECRET + " reflected provider error").encode()
            return FakeResponse(status, response_body, {
                "Content-Length": str(len(response_body)), "ETag": '"verified-etag"',
                "x-amz-meta-sha256": "a" * 64, "X-Do-Not-Forward": SECRET,
            })

        def close(self):
            pass

    monkeypatch.setattr("gradlab.r2_signing_gateway.HTTPSConnection", Upstream)
    server = GatewayServer(("127.0.0.1", 0), bucket="models", endpoint=f"https://{HOST}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/models/run/model.zip", method="PUT", data=body,
            headers={"Authorization": "Basic " + base64.b64encode(f"{ACCESS}:{SECRET}".encode()).decode(),
                     "If-None-Match": "*", "x-amz-meta-sha256": "a" * 64},
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        if status != 200:
            with pytest.raises(urllib.error.HTTPError) as error:
                opener.open(request, timeout=5)
            assert error.value.code == status
            assert SECRET.encode() not in error.value.read()
        else:
            with opener.open(request, timeout=5) as response:
                assert response.read() == body
                assert response.headers["ETag"] == '"verified-etag"'
                assert response.headers["x-amz-meta-sha256"] == "a" * 64
                assert "X-Do-Not-Forward" not in response.headers
        method, target, actual_body, headers = observed[0]
        assert method == "PUT" and target == "/models/run/model.zip" and actual_body == body
        assert headers["if-none-match"] == "*"
        assert headers["authorization"].startswith("AWS4-HMAC-SHA256 ")
        assert SECRET not in str(headers)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
