from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Protocol


def format_utc_datetime(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def parse_utc_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def utc_now() -> str:
    return format_utc_datetime(datetime.now(UTC))


def utc_timestamp(unix_seconds: float) -> str:
    return format_utc_datetime(datetime.fromtimestamp(unix_seconds, UTC))


class Clock(Protocol):
    """Time source shared by orchestration state and deterministic certification."""

    def time(self) -> float: ...

    def monotonic(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...

    def utc_datetime(self) -> datetime: ...

    def utc_now(self) -> str: ...


class SystemClock:
    def time(self) -> float:
        return time.time()

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def utc_datetime(self) -> datetime:
        return datetime.now(UTC)

    def utc_now(self) -> str:
        return format_utc_datetime(self.utc_datetime())
