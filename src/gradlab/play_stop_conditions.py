from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal


MAX_SOURCE_LENGTH = 4096
MAX_TOKENS = 256
MAX_NESTING = 32
EPISODE_SYMBOLS = (
    "episode.terminated",
    "episode.truncated",
    "episode.boundary",
)

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_NUMBER = re.compile(r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?")


@dataclass(frozen=True)
class StopConditionError:
    message: str
    offset: int
    length: int

    def payload(self) -> dict[str, object]:
        return {
            "message": self.message,
            "offset": self.offset,
            "length": self.length,
        }


@dataclass(frozen=True)
class StopMatch:
    source: str
    values: dict[str, float | int]

    def payload(self) -> dict[str, object]:
        return {"source": self.source, "values": dict(self.values)}


@dataclass(frozen=True)
class _Token:
    kind: str
    text: str
    offset: int

    @property
    def length(self) -> int:
        return len(self.text)


@dataclass(frozen=True)
class _SymbolRef:
    name: str
    kind: Literal["event", "signal", "episode"]
    key: str


@dataclass(frozen=True)
class _Comparison:
    symbol: _SymbolRef
    operator: str
    expected: float


@dataclass(frozen=True)
class _Boolean:
    operator: Literal["and", "or"]
    left: "_Expression"
    right: "_Expression"


_Expression = _Comparison | _Boolean


class _ParseFailure(Exception):
    def __init__(self, error: StopConditionError) -> None:
        super().__init__(error.message)
        self.error = error


def _tokenize(source: str) -> tuple[_Token, ...]:
    if not source.strip():
        raise _ParseFailure(StopConditionError("stop condition is required", 0, 0))
    if len(source) > MAX_SOURCE_LENGTH:
        raise _ParseFailure(
            StopConditionError(
                f"stop condition must be at most {MAX_SOURCE_LENGTH} characters",
                MAX_SOURCE_LENGTH,
                len(source) - MAX_SOURCE_LENGTH,
            )
        )
    tokens: list[_Token] = []
    offset = 0
    while offset < len(source):
        if source[offset].isspace():
            offset += 1
            continue
        operator = next(
            (candidate for candidate in ("==", "!=", "<=", ">=", "<", ">") if source.startswith(candidate, offset)),
            None,
        )
        if operator is not None:
            tokens.append(_Token("operator", operator, offset))
            offset += len(operator)
        elif source[offset] == "(":
            tokens.append(_Token("left_parenthesis", "(", offset))
            offset += 1
        elif source[offset] == ")":
            tokens.append(_Token("right_parenthesis", ")", offset))
            offset += 1
        else:
            identifier = _IDENTIFIER.match(source, offset)
            if identifier is not None:
                text = identifier.group(0)
                kind = text if text in {"and", "or"} else "identifier"
                tokens.append(_Token(kind, text, offset))
                offset = identifier.end()
            else:
                number = _NUMBER.match(source, offset)
                if number is None:
                    raise _ParseFailure(
                        StopConditionError(
                            f"unexpected character {source[offset]!r}",
                            offset,
                            1,
                        )
                    )
                text = number.group(0)
                tokens.append(_Token("number", text, offset))
                offset = number.end()
        if len(tokens) > MAX_TOKENS:
            token = tokens[-1]
            raise _ParseFailure(
                StopConditionError(
                    f"stop condition must contain at most {MAX_TOKENS} tokens",
                    token.offset,
                    token.length,
                )
            )
    tokens.append(_Token("end", "", len(source)))
    return tuple(tokens)


class _Parser:
    def __init__(self, source: str, resolve_symbol) -> None:
        self._tokens = _tokenize(source)
        self._resolve_symbol = resolve_symbol
        self._index = 0
        self._depth = 0

    @property
    def current(self) -> _Token:
        return self._tokens[self._index]

    def parse(self) -> _Expression:
        expression = self._parse_or()
        if self.current.kind != "end":
            self._fail(self.current, f"unexpected token {self.current.text!r}")
        return expression

    def _parse_or(self) -> _Expression:
        expression = self._parse_and()
        while self.current.kind == "or":
            self._index += 1
            expression = _Boolean("or", expression, self._parse_and())
        return expression

    def _parse_and(self) -> _Expression:
        expression = self._parse_primary()
        while self.current.kind == "and":
            self._index += 1
            expression = _Boolean("and", expression, self._parse_primary())
        return expression

    def _parse_primary(self) -> _Expression:
        if self.current.kind != "left_parenthesis":
            return self._parse_comparison()
        opening = self.current
        self._index += 1
        self._depth += 1
        if self._depth > MAX_NESTING:
            self._fail(
                opening,
                f"stop condition nesting must be at most {MAX_NESTING} levels",
            )
        expression = self._parse_or()
        if self.current.kind != "right_parenthesis":
            self._fail(self.current, "expected ')'")
        self._index += 1
        self._depth -= 1
        return expression

    def _parse_comparison(self) -> _Comparison:
        identifier = self.current
        if identifier.kind != "identifier":
            self._fail(identifier, "expected an event, episode counter, or numeric signal")
        self._index += 1
        operator = self.current
        if operator.kind != "operator":
            self._fail(operator, "expected a comparison operator")
        self._index += 1
        number = self.current
        if number.kind != "number":
            self._fail(number, "expected a finite number")
        self._index += 1
        expected = float(number.text)
        if not math.isfinite(expected):
            self._fail(number, "expected a finite number")
        symbol = self._resolve_symbol(identifier)
        return _Comparison(symbol, operator.text, expected)

    @staticmethod
    def _fail(token: _Token, message: str) -> None:
        raise _ParseFailure(StopConditionError(message, token.offset, token.length))


def _numeric(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


class PlaybackStopController:
    """Own one bounded stop program and its Playback-run numeric state."""

    def __init__(
        self,
        *,
        event_names: Iterable[str] = (),
        signal_names: Iterable[str] = (),
        source: str,
    ) -> None:
        self._event_names = {str(name) for name in event_names if str(name)}
        self._signal_names = {str(name) for name in signal_names if str(name)}
        self.source = ""
        self.error: StopConditionError | None = None
        self._expression: _Expression | None = None
        self._event_counts: Counter[str] = Counter()
        self._signals: dict[str, float | None] = {}
        self._episode_counts: Counter[str] = Counter()
        self.last_match: StopMatch | None = None
        self.set_source(source)

    @property
    def valid(self) -> bool:
        return self._expression is not None and self.error is None

    def set_source(self, source: str) -> None:
        self.source = str(source)
        self.reset()
        self._compile()

    def reset(self) -> None:
        self._event_counts = Counter({name: 0 for name in self._event_names})
        self._signals = {name: None for name in self._signal_names}
        self._episode_counts = Counter({name: 0 for name in EPISODE_SYMBOLS})
        self.last_match = None

    def observe(
        self,
        *,
        events: Sequence[str] = (),
        signals: Mapping[str, object] | None = None,
        terminated: bool = False,
        truncated: bool = False,
        boundary: bool = False,
    ) -> StopMatch | None:
        catalog_changed = False
        for raw_name in events:
            name = str(raw_name)
            if name not in self._event_names:
                self._event_names.add(name)
                self._event_counts[name] = 0
                catalog_changed = True
            self._event_counts[name] += 1
        for raw_name, raw_value in (signals or {}).items():
            name = str(raw_name)
            value = _numeric(raw_value)
            if value is None:
                continue
            if name not in self._signal_names:
                self._signal_names.add(name)
                catalog_changed = True
            self._signals[name] = value
        if terminated:
            self._episode_counts["episode.terminated"] += 1
        if truncated:
            self._episode_counts["episode.truncated"] += 1
        if boundary:
            self._episode_counts["episode.boundary"] += 1
        if catalog_changed:
            self._compile()
        if self._expression is None or not self._evaluate(self._expression):
            return None
        values: dict[str, float | int] = {}
        self._collect_values(self._expression, values)
        self.last_match = StopMatch(self.source, values)
        return self.last_match

    def payload(self) -> dict[str, object]:
        symbols = self._symbol_payloads()
        values = {
            symbol["name"]: symbol["value"]
            for symbol in symbols
        }
        return {
            "source": self.source,
            "valid": self.valid,
            "error": None if self.error is None else self.error.payload(),
            "values": values,
            "symbols": symbols,
            "matched": None if self.last_match is None else self.last_match.payload(),
        }

    def _compile(self) -> None:
        try:
            self._expression = _Parser(self.source, self._resolve_symbol).parse()
        except _ParseFailure as exc:
            self._expression = None
            self.error = exc.error
        else:
            self.error = None

    def _resolve_symbol(self, token: _Token) -> _SymbolRef:
        name = token.text
        if name in EPISODE_SYMBOLS:
            return _SymbolRef(name, "episode", name)
        if name.startswith("event.") and name.removeprefix("event.") in self._event_names:
            return _SymbolRef(name, "event", name.removeprefix("event."))
        if name.startswith("signal.") and name.removeprefix("signal.") in self._signal_names:
            return _SymbolRef(name, "signal", name.removeprefix("signal."))
        is_event = name in self._event_names
        is_signal = name in self._signal_names
        if is_event and is_signal:
            raise _ParseFailure(
                StopConditionError(
                    f"ambiguous identifier {name!r}; use event.{name} or signal.{name}",
                    token.offset,
                    token.length,
                )
            )
        if is_event:
            return _SymbolRef(name, "event", name)
        if is_signal:
            return _SymbolRef(name, "signal", name)
        raise _ParseFailure(
            StopConditionError(
                f"unknown identifier {name!r}",
                token.offset,
                token.length,
            )
        )

    def _value(self, symbol: _SymbolRef) -> float | int | None:
        if symbol.kind == "event":
            return self._event_counts[symbol.key]
        if symbol.kind == "signal":
            return self._signals.get(symbol.key)
        return self._episode_counts[symbol.key]

    def _evaluate(self, expression: _Expression) -> bool:
        if isinstance(expression, _Boolean):
            if expression.operator == "and":
                return self._evaluate(expression.left) and self._evaluate(expression.right)
            return self._evaluate(expression.left) or self._evaluate(expression.right)
        actual = self._value(expression.symbol)
        if actual is None:
            return False
        expected = expression.expected
        return {
            "==": actual == expected,
            "!=": actual != expected,
            "<": actual < expected,
            "<=": actual <= expected,
            ">": actual > expected,
            ">=": actual >= expected,
        }[expression.operator]

    def _collect_values(
        self,
        expression: _Expression,
        values: dict[str, float | int],
    ) -> None:
        if isinstance(expression, _Boolean):
            self._collect_values(expression.left, values)
            self._collect_values(expression.right, values)
            return
        value = self._value(expression.symbol)
        if value is not None:
            values[expression.symbol.name] = value

    def _symbol_payloads(self) -> list[dict[str, object]]:
        symbols: list[_SymbolRef] = [
            _SymbolRef(name, "episode", name) for name in EPISODE_SYMBOLS
        ]
        for name in sorted(self._event_names | self._signal_names):
            is_event = name in self._event_names
            is_signal = name in self._signal_names
            if is_event and is_signal:
                symbols.extend(
                    (
                        _SymbolRef(f"event.{name}", "event", name),
                        _SymbolRef(f"signal.{name}", "signal", name),
                    )
                )
            elif is_event:
                symbols.append(_SymbolRef(name, "event", name))
            else:
                symbols.append(_SymbolRef(name, "signal", name))
        payloads: list[dict[str, object]] = []
        for symbol in symbols:
            value = self._value(symbol)
            payloads.append(
                {
                    "name": symbol.name,
                    "kind": symbol.kind,
                    "value": value,
                    "available": value is not None,
                }
            )
        return payloads


def default_stop_expression(conditions: Iterable[Mapping[str, Any]]) -> str:
    comparisons: list[str] = []
    for condition in conditions:
        if condition.get("enabled") is False:
            continue
        event = condition.get("event")
        if isinstance(event, str) and _IDENTIFIER.fullmatch(event):
            comparisons.append(f"{event} >= 1")
        elif condition.get("id") == "limit:max_episode_steps":
            comparisons.append("episode.truncated >= 1")
    return " or ".join(dict.fromkeys(comparisons)) or "episode.boundary >= 1"
