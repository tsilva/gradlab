from unittest.mock import patch

import pytest

from gradlab.lifecycle_certification import CertificationFixture
from gradlab.run_authority import LeaseUnavailable
from gradlab.wandb_publisher import WandbProjector


def test_background_loss_defers_observer_and_stop_state_to_main_thread(tmp_path):
    fixture = CertificationFixture(tmp_path)
    prepared = fixture.prepare(run_number=39)
    supervisor = prepared.supervisor
    fixture.clock.advance(15)
    with (
        patch.object(supervisor.authority, "renew_lease", side_effect=LeaseUnavailable("lost")),
        patch.object(supervisor.observer, "emit") as emit,
    ):
        supervisor._renew_lease(fixture.clock.monotonic(), background=True)
        assert supervisor.lease_lost
        assert supervisor.stop_reason == ""
        emit.assert_not_called()
        with pytest.raises(LeaseUnavailable):
            supervisor._lease_heartbeat()
        assert supervisor.stop_reason == "writer_lease_lost"
        assert any(call.args[0] == "writer_lease_lost" for call in emit.call_args_list)


def test_lease_loss_during_wandb_start_does_not_publish_writer_receipt(tmp_path):
    fixture = CertificationFixture(tmp_path)
    prepared = fixture.prepare(run_number=39)
    supervisor = prepared.supervisor

    def start(*args, **kwargs):
        fixture.clock.advance(90)
        return WandbProjector(object())

    with (
        patch("gradlab.run_supervisor.load_materialized_train_config", return_value={}),
        patch("gradlab.run_supervisor.env_config_from_mapping"),
        patch("gradlab.run_supervisor.resolve_env_config"),
        patch.object(prepared.runtime, "start_wandb", side_effect=start),
        patch.object(supervisor.authority, "renew_lease", side_effect=LeaseUnavailable("lost")),
        prepared.runtime.maintain_lease(
            lambda: supervisor._renew_lease(fixture.clock.monotonic(), background=True)
        ),
    ):
        with pytest.raises(LeaseUnavailable):
            supervisor._start_wandb()
    assert (
        supervisor.authority.control.get_json_optional(
            f"runs/{supervisor.manifest.run_id}/wandb.json"
        )
        is None
    )
