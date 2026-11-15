from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from django.utils import timezone

if TYPE_CHECKING:
    from datetime import datetime


class Clock(Protocol):
    """The time source of services, so tests can fix it."""

    def now(self) -> datetime:
        """Return the current time."""
        ...


class SystemClock:
    """The real, timezone-aware time."""

    def now(self) -> datetime:
        """Return the current time."""
        return timezone.now()
