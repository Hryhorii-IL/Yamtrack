from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime


class FixedClock:
    """A clock stuck at a given time."""

    def __init__(self, now: datetime) -> None:
        """Keep the time to return."""
        self._now = now

    def now(self) -> datetime:
        """Return the fixed time."""
        return self._now
