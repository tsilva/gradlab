from __future__ import annotations

import pytest

from gradlab.play_stop_conditions import (
    PlaybackStopController,
    default_stop_expression,
)


def test_stop_condition_counts_events_and_episode_outcomes_across_boundaries() -> None:
    controller = PlaybackStopController(
        event_names=("life_loss",),
        signal_names=("x_pos",),
        source="episode.terminated == 2 and (x_pos >= 3000 or life_loss >= 2)",
    )

    first = controller.observe(
        events=("life_loss",),
        signals={"x_pos": 1200},
        terminated=True,
        truncated=False,
        boundary=True,
    )
    second = controller.observe(
        events=("life_loss",),
        signals={"x_pos": 3100},
        terminated=True,
        truncated=False,
        boundary=True,
    )

    assert first is None
    assert second is not None
    assert second.values == {
        "episode.terminated": 2,
        "x_pos": 3100.0,
        "life_loss": 2,
    }
    assert controller.payload()["values"]["episode.boundary"] == 2


def test_stop_condition_uses_and_before_or_and_parentheses_override_it() -> None:
    ordinary = PlaybackStopController(
        event_names=("a", "b", "c"),
        source="a >= 1 or b >= 1 and c >= 1",
    )
    grouped = PlaybackStopController(
        event_names=("a", "b", "c"),
        source="(a >= 1 or b >= 1) and c >= 1",
    )

    transition = {
        "events": ("a",),
        "signals": {},
        "terminated": False,
        "truncated": False,
        "boundary": False,
    }

    assert ordinary.observe(**transition) is not None
    assert grouped.observe(**transition) is None


def test_unavailable_signal_is_false_until_a_finite_value_arrives() -> None:
    controller = PlaybackStopController(
        signal_names=("x_pos",),
        source="x_pos >= 10",
    )

    assert controller.observe(signals={}) is None
    assert controller.observe(signals={"x_pos": float("nan")}) is None
    assert controller.observe(signals={"x_pos": 10}) is not None


def test_ambiguous_short_name_requires_an_explicit_qualified_identifier() -> None:
    ambiguous = PlaybackStopController(
        event_names=("score",),
        signal_names=("score",),
        source="score >= 1",
    )
    explicit = PlaybackStopController(
        event_names=("score",),
        signal_names=("score",),
        source="event.score >= 1 and signal.score >= 10",
    )

    assert ambiguous.payload()["valid"] is False
    assert "ambiguous identifier 'score'" in ambiguous.payload()["error"]["message"]
    assert explicit.observe(events=("score",), signals={"score": 10}) is not None


def test_project_safe_symbol_names_are_parseable_and_qualified_when_needed() -> None:
    controller = PlaybackStopController(
        event_names=("level-1.complete", "123-ready"),
        signal_names=("123-ready",),
        source="level-1.complete >= 1 and event.123-ready >= 1",
    )

    payload = controller.payload()
    assert payload["valid"] is True
    assert {symbol["name"] for symbol in payload["symbols"]} >= {
        "level-1.complete",
        "event.123-ready",
        "signal.123-ready",
    }
    assert controller.observe(events=("level-1.complete", "123-ready")) is not None


def test_invalid_source_is_retained_with_a_structured_error_and_cannot_match() -> None:
    controller = PlaybackStopController(
        event_names=("life_loss",),
        source="life_loss >=",
    )

    payload = controller.payload()

    assert payload["source"] == "life_loss >="
    assert payload["valid"] is False
    assert payload["error"]["offset"] == len("life_loss >=")
    assert payload["error"]["length"] == 0
    assert controller.observe(events=("life_loss",)) is None


def test_unknown_identifiers_and_excessive_input_are_rejected() -> None:
    unknown = PlaybackStopController(source="missing >= 1")
    excessive = PlaybackStopController(source="x" * 4097)

    assert unknown.payload()["valid"] is False
    assert "unknown identifier 'missing'" in unknown.payload()["error"]["message"]
    assert excessive.payload()["valid"] is False
    assert "at most 4096 characters" in excessive.payload()["error"]["message"]


@pytest.mark.parametrize("operator", ("==", "!=", "<", "<=", ">", ">="))
def test_every_numeric_comparison_operator_is_supported(operator: str) -> None:
    expected = {
        "==": False,
        "!=": True,
        "<": True,
        "<=": True,
        ">": False,
        ">=": False,
    }[operator]
    controller = PlaybackStopController(signal_names=("value",), source=f"value {operator} 2")

    assert (controller.observe(signals={"value": 1}) is not None) is expected


@pytest.mark.parametrize(
    "source",
    (
        "value + 1 >= 2",
        "value() >= 1",
        "value >= 1 >= 0",
        "not value >= 1",
        "value = 1",
        "value == '1'",
    ),
)
def test_non_dsl_constructs_are_rejected(source: str) -> None:
    controller = PlaybackStopController(signal_names=("value",), source=source)

    assert controller.valid is False


def test_token_and_nesting_limits_are_enforced() -> None:
    too_many_tokens = PlaybackStopController(
        event_names=("x",),
        source=" or ".join("x == 0" for _ in range(70)),
    )
    too_deep = PlaybackStopController(
        event_names=("x",),
        source="(" * 33 + "x == 0" + ")" * 33,
    )
    non_finite = PlaybackStopController(signal_names=("x",), source="x >= 1e999")

    assert "at most 256 tokens" in too_many_tokens.payload()["error"]["message"]
    assert "at most 32 levels" in too_deep.payload()["error"]["message"]
    assert non_finite.payload()["error"]["message"] == "expected a finite number"


def test_setting_a_new_source_resets_values_and_last_match() -> None:
    controller = PlaybackStopController(event_names=("goal",), source="goal >= 1")
    assert controller.observe(events=("goal",)) is not None

    controller.set_source("goal >= 2")

    payload = controller.payload()
    assert payload["valid"] is True
    assert payload["values"]["goal"] == 0
    assert payload["matched"] is None


def test_default_expression_preserves_enabled_termination_choices() -> None:
    assert default_stop_expression(
        (
            {"id": "event:level_complete", "event": "level_complete", "enabled": True},
            {"id": "event:life_loss", "event": "life_loss", "enabled": False},
            {"id": "limit:max_episode_steps", "event": None, "enabled": True},
        )
    ) == "level_complete >= 1 or episode.truncated >= 1"
    assert default_stop_expression(
        ({"id": "event:123-ready", "event": "123-ready", "enabled": True},)
    ) == "event.123-ready >= 1"
    assert default_stop_expression(()) == "episode.boundary >= 1"
