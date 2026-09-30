from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import base64
import runpy
import threading

import pytest


@pytest.fixture
def public_view():
    requests = []

    class Upstream(BaseHTTPRequestHandler):
        def do_GET(self):
            # Consume POST bodies before closing HTTP/1.0 responses; leaving
            # unread bytes can reset the socket during the gateway's stream.
            self.rfile.read(int(self.headers.get('Content-Length', '0')))
            requests.append((self.command, self.path, self.headers.get('Authorization')))
            self.send_response(403 if 'private-run' in self.path else 200)
            self.send_header('Content-Type', 'video/mp4' if self.path.startswith('/get-artifact')
                             else 'application/json')
            self.end_headers()
            self.wfile.write(b'video-bytes' if self.path.startswith('/get-artifact')
                             else b'{"histories": [0.25, 0.75]}')

        do_POST = do_GET

        def log_message(self, *_):
            pass

    upstream = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
    worker = threading.Thread(target=upstream.serve_forever, daemon=True)
    worker.start()
    try:
        module = runpy.run_path(str(Path(__file__).parents[1] / 'ops/mlflow/public_gateway.py'))
        app = module['create_app'](f'http://127.0.0.1:{upstream.server_port}',
                                   'reader', 'test-reader-password')
        yield app.test_client(), requests
    finally:
        upstream.shutdown()
        upstream.server_close()
        worker.join()


def test_anonymous_reader_can_inspect_histories_and_video_without_writer_credentials(public_view):
    client, requests = public_view
    assert client.get('/').status_code == 200
    assert client.post('/ajax-api/2.0/mlflow/runs/search', json={}).status_code == 200
    history = client.get('/ajax-api/2.0/mlflow/metrics/get-history?run_id=pilot&metric_key=return',
                         headers={'Authorization': 'Basic supplied-by-visitor'})
    assert history.json == {'histories': [0.25, 0.75]}
    video = client.get('/get-artifact?run_uuid=pilot&path=journal/6/episode.mp4')
    assert video.data == b'video-bytes'
    assert video.headers['Content-Type'] == 'video/mp4'
    expected = 'Basic ' + base64.b64encode(b'reader:test-reader-password').decode()
    assert all(row[2] == expected for row in requests)
    assert client.get('/api/2.0/mlflow/runs/get?run_id=private-run').status_code == 403


@pytest.mark.parametrize('method,path', [
    ('POST', '/api/2.0/mlflow/runs/create'),
    ('POST', '/ajax-api/2.0/mlflow/runs/delete'),
    ('POST', '/ajax-api/2.0/mlflow/upload-artifact'),
    ('PUT', '/api/2.0/mlflow-artifacts/artifacts/journal/6/episode.mp4'),
    ('DELETE', '/api/2.0/mlflow-artifacts/artifacts/journal/6/episode.mp4'),
    ('GET', '/api/2.0/mlflow/users/get'),
    ('POST', '/api/2.0/mlflow/gateway-proxy'),
    ('GET', '/static-files/../api/2.0/mlflow/users/get'),
    ('GET', '/%2f%2fexample.com/'),
])
def test_public_route_rejects_mutation_and_unknown_routes_before_upstream(public_view, method, path):
    client, requests = public_view
    assert client.open(path, method=method).status_code == 403
    assert requests == []


def test_native_ui_graphql_query_is_read_only(public_view):
    client, requests = public_view
    query = 'query Run { mlflowGetRun(input: {runId: "pilot"}) { run { info { runId } } } }'
    assert client.post('/graphql', json={'query': query}).status_code == 200
    assert len(requests) == 1


@pytest.mark.parametrize('query', [
    'mutation { mlflowSearchRuns(input: {}) { runs { info { runId } } } }',
    'query Safe { mlflowGetRun(input: {}) { run { info { runId } } } } mutation Bad { testMutation }',
    '{ __schema { queryType { name } } }',
    '{ unknownField }',
    '{ ...Cycle } fragment Cycle on Query { ...Cycle }',
    'invalid query',
])
def test_graphql_rejects_mutations_and_unrecognized_fields(public_view, query):
    client, requests = public_view
    assert client.post('/graphql', json={'query': query}).status_code == 403
    assert not requests
