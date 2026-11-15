from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ErrorCode(StrEnum):
    """Why a use case failed; entry points map codes, never exception classes."""

    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    INVALID = "invalid"
    UNAVAILABLE = "unavailable"


@dataclass(slots=True, frozen=True)
class Result[T]:
    """The value of a use case, or the code of its failure."""

    value: T | None = None
    error_code: ErrorCode | None = None

    @property
    def ok(self) -> bool:
        """Return whether the use case succeeded."""
        return self.error_code is None
