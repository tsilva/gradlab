from gradlab.play_timeline import EventOverview


def test_overview_preserves_early_events_with_bounded_storage():
    overview = EventOverview("episode", limit=4)
    for step in range(1, 10001):
        overview.append({"step": step, "events": ["brick"], "boundary": step == 10000})
    payload = overview.payload()
    assert len(payload["points"]) <= 4
    assert payload["points"][0]["step"] == 1
    assert payload["points"][-1]["last_step"] == 10000
    assert sum(point["count"] for point in payload["points"]) == 10000
    assert payload["points"][-1]["boundary"]
    overview.append({"step": 1, "events": ["brick"]})
    assert overview.payload() == payload
    payload["points"][0]["events"].append("mutated")
    assert overview.payload()["points"][0]["events"] == ["brick"]


def test_overview_tracks_empty_steps_without_inventing_events():
    overview = EventOverview("episode")
    overview.append({"step": 1, "events": []})
    assert overview.payload()["points"] == []
    assert overview.payload()["through_step"] == 1
    overview.append({"step": 2, "events": [], "boundary": True})
    assert overview.payload()["points"] == [
        {"step": 2, "last_step": 2, "count": 1, "boundary": True, "events": []}
    ]
