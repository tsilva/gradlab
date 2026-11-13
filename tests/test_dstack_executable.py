from gradlab.dstack_backend import DstackBackend
import pytest


def backend(environment, executable='dstack'):
    return DstackBackend(project='synthetic', server_url='http://127.0.0.1:3000',
                         token='public-placeholder', environment=environment, executable=executable)


def test_reviewed_absolute_cli_path_overrides_only_the_default_name():
    env = {'GRADLAB_DSTACK_EXECUTABLE': '/reviewed/dstack'}
    assert backend(env).executable == '/reviewed/dstack'
    assert backend(env, '/explicit/dstack').executable == '/explicit/dstack'
    assert backend({}).executable == 'dstack'


def test_relative_configured_cli_is_rejected():
    with pytest.raises(ValueError, match='absolute'):
        backend({'GRADLAB_DSTACK_EXECUTABLE': 'unreviewed/dstack'})
