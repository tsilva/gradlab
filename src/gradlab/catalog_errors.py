from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CatalogProblem:
    code: str
    message: str
    status: int
    retryable: bool = False
    source: str = "catalog"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "source": self.source,
        }


class CatalogError(RuntimeError):
    def __init__(self, problem: CatalogProblem):
        super().__init__(problem.message)
        self.problem = problem


class CatalogUnavailable(CatalogError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "catalog_unavailable",
        retryable: bool = False,
        source: str = "control-catalog",
    ) -> None:
        super().__init__(
            CatalogProblem(
                code=code,
                message=str(message),
                status=503,
                retryable=retryable,
                source=source,
            )
        )


class CatalogIntegrityError(CatalogError):
    def __init__(self, message: str, *, source: str = "catalog") -> None:
        super().__init__(
            CatalogProblem(
                code="catalog_integrity",
                message=str(message),
                status=502,
                retryable=False,
                source=source,
            )
        )


class CatalogSnapshotChanged(CatalogError):
    def __init__(self, message: str = "catalog snapshot changed; restart pagination") -> None:
        super().__init__(
            CatalogProblem(
                code="catalog_snapshot_changed",
                message=str(message),
                status=409,
                retryable=True,
                source="catalog",
            )
        )


__all__ = [
    "CatalogError",
    "CatalogIntegrityError",
    "CatalogProblem",
    "CatalogSnapshotChanged",
    "CatalogUnavailable",
]
