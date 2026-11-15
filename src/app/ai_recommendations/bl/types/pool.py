from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    from app.ai_recommendations.bl.types.recommendation import ResolvedTitle


class QueueKind(Enum):
    """Background pool tasks that are queued at most once per media type."""

    GENERATION = auto()
    EXTENSION = auto()


@dataclass(slots=True, frozen=True)
class RecommendationPool:
    """The global recommendations of a media type, generated in batches."""

    generated_at: datetime
    items: tuple[ResolvedTitle, ...] = ()
    # Still being generated, item by item
    partial: bool = False
    error: str | None = None
    extensions: int = 0

    @property
    def version(self) -> str:
        """Return an identifier pages use to notice a regenerated pool."""
        return str(self.generated_at.timestamp())


@dataclass(slots=True, frozen=True)
class PoolPage[CardT]:
    """The pool items a page should append after ``offset`` cards were handled."""

    results: tuple[CardT, ...]
    # Cards on the page after appending the results
    shown: int
    # Pool items handled so far, the next page starts there
    offset: int
    version: str
    growing: bool
    partial: bool
    error: str | None = None


@dataclass(slots=True, frozen=True)
class MediaRecommendations:
    """The recommendations similar to a single media."""

    generated_at: datetime
    items: tuple[ResolvedTitle, ...] = ()
