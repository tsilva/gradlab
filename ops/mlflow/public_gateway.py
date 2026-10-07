"""Anonymous MLflow inspection backed by an experiment-scoped READ account.

Run with the pinned mlflow-server environment and a production WSGI server:
gunicorn --bind 127.0.0.1:5003 --workers 2 'public_gateway:application()'
"""

import os
from pathlib import PurePosixPath
from urllib.parse import unquote, urlsplit

from flask import Flask, Response, request, stream_with_context
import requests
from graphql import GraphQLError, OperationType, parse
from graphql.language.ast import (
    FieldNode, FragmentDefinitionNode, FragmentSpreadNode, InlineFragmentNode,
    OperationDefinitionNode,
)


READ_ROUTES = frozenset({
    'experiments/get', 'experiments/get-by-name', 'experiments/search',
    'runs/get', 'runs/search', 'metrics/get-history', 'metrics/get-history-bulk',
    'metrics/get-history-bulk-interval', 'artifacts/list',
})
SEARCH_ROUTES = frozenset({'experiments/search', 'runs/search', 'experiments/search-datasets'})
GRAPHQL_READ_FIELDS = frozenset({
    'mlflowGetExperiment', 'mlflowGetRun', 'mlflowGetMetricHistoryBulkInterval',
    'mlflowListArtifacts', 'mlflowSearchRuns', 'mlflowSearchModelVersions',
})


def read_query(document: object) -> bool:
    if not isinstance(document, dict) or not isinstance(document.get('query'), str):
        return False
    try:
        ast = parse(document['query'], max_tokens=20000)
    except (GraphQLError, RecursionError):
        return False
    fragments = {d.name.value: d for d in ast.definitions
                 if isinstance(d, FragmentDefinitionNode)}
    operations = [d for d in ast.definitions if isinstance(d, OperationDefinitionNode)]
    if not operations or any(d.operation != OperationType.QUERY for d in operations):
        return False

    def allowed(selection_set, visited=frozenset()):
        for field in selection_set.selections:
            if isinstance(field, FieldNode):
                if field.name.value not in GRAPHQL_READ_FIELDS:
                    return False
            elif isinstance(field, InlineFragmentNode):
                if not allowed(field.selection_set, visited):
                    return False
            elif isinstance(field, FragmentSpreadNode):
                name = field.name.value
                if name in visited or name not in fragments:
                    return False
                if not allowed(fragments[name].selection_set, visited | {name}):
                    return False
            else:
                return False
        return True

    try:
        return all(allowed(d.selection_set) for d in operations)
    except RecursionError:
        return False


def create_app(upstream: str, username: str, password: str) -> Flask:
    destination = urlsplit(upstream)
    if (destination.scheme != 'http' or destination.hostname != '127.0.0.1'
            or destination.username or destination.password or destination.path
            or destination.query or destination.fragment or not destination.port):
        raise ValueError('the public gateway requires an explicit loopback HTTP upstream')
    if not username or not password:
        raise ValueError('the public gateway requires a dedicated READ account')
    app = Flask(__name__, static_folder=None)
    app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024

    @app.route('/', defaults={'path': ''}, methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
    @app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
    def inspect(path: str):
        path = '/' + path
        decoded = unquote(path)
        if (decoded != path or '\\' in path or '//' in path
                or '..' in PurePosixPath(path).parts or len(request.query_string) > 16384):
            return Response('Public inspection route unavailable', status=403)
        prefixes = ('/api/2.0/mlflow/', '/ajax-api/2.0/mlflow/')
        route = next((path[len(p):] for p in prefixes if path.startswith(p)), None)
        allowed = (
            request.method in {'GET', 'HEAD'}
            and (path in {'/', '/health', '/version', '/get-artifact'}
                 or path.startswith('/static-files/') or path.startswith('/static/')
                 or path in {'/ajax-api/3.0/mlflow/server-info',
                             '/ajax-api/3.0/mlflow/ui-telemetry',
                             '/ajax-api/2.0/mlflow/users/current'}
                 or route in READ_ROUTES)
        ) or (request.method == 'POST' and route in SEARCH_ROUTES)
        if request.method == 'POST' and path == '/graphql':
            allowed = read_query(request.get_json(silent=True))
        if not allowed:
            app.logger.warning('Denied public inspection route: %s %s', request.method, path)
            return Response('Public inspection is read only', status=403)
        headers = {'Content-Type': request.headers.get('Content-Type', 'application/json')}
        if request.headers.get('Range'):
            headers['Range'] = request.headers['Range']
        try:
            response = requests.request(
                request.method, upstream + path,
                params=request.args, data=request.get_data(), headers=headers,
                auth=(username, password), timeout=(5, 30), stream=True,
                allow_redirects=False,
            )
        except requests.RequestException:
            return Response('MLflow inspection is temporarily unavailable', status=502)
        if 300 <= response.status_code < 400:
            response.close()
            return Response('Upstream redirects are unavailable', status=502)

        def content():
            try:
                yield from response.iter_content(chunk_size=64 * 1024)
            finally:
                response.close()

        forwarded = {
            name: response.headers[name]
            for name in ('Content-Type', 'Content-Range', 'Accept-Ranges', 'Content-Disposition')
            if name in response.headers
        }
        forwarded.update({'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'})
        return Response(stream_with_context(content()), status=response.status_code,
                        headers=forwarded)

    return app


def application() -> Flask:
    return create_app(os.environ['GRADLAB_MLFLOW_READER_UPSTREAM'],
                      os.environ['GRADLAB_MLFLOW_READER_USERNAME'],
                      os.environ['GRADLAB_MLFLOW_READER_PASSWORD'])
