from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.ai_recommendations.bl.errors import LibraryStoreNotFoundError
from app.ai_recommendations.bl.types.taste import TasteInput
from app.ai_recommendations.tests.fakes.builders import ALL_RATINGS

if TYPE_CHECKING:
    from app.ai_recommendations.bl.types.taste import RatedTitle


@dataclass
class FakeLibrary:
    """What a user tracks and how they configured AI recommendations."""

    enabled_media_types: tuple[str, ...] = ("movie",)
    # Titles tracked per media type, newest first
    titles: dict[str, list[str]] = field(default_factory=dict)
    rated: list[RatedTitle] = field(default_factory=list)
    # User scores by (media_type, source, media_id)
    scores: dict[tuple[str, str, str], float] = field(default_factory=dict)


class InMemoryLibraryStore:
    """A LibraryStore reading fake libraries, for facade tests."""

    def __init__(self, libraries: dict[int, FakeLibrary] | None = None) -> None:
        """Keep the library of each user id."""
        self.libraries = libraries or {}

    def load_enabled_media_types(self, user_id: int) -> tuple[str, ...]:
        """Return the AI media types enabled in the user's preferences."""
        return self._library(user_id).enabled_media_types

    def has_media(self, user_id: int, media_type: str) -> bool:
        """Return whether the user tracks anything of a media type."""
        return bool(self._library(user_id).titles.get(media_type))

    def load_taste(self, user_id: int, media_type: str) -> TasteInput:
        """Return the user's ratings and library of a media type."""
        library = self._library(user_id)
        return TasteInput(
            settings=ALL_RATINGS,
            rated=tuple(
                title
                for title in library.rated
                if title.media_type in library.enabled_media_types
            ),
            library=tuple(library.titles.get(media_type, ())),
        )

    def load_user_score(
        self,
        user_id: int,
        media_type: str,
        source: str,
        media_id: str,
    ) -> float | None:
        """Return the user's score of a media, if rated."""
        return self._library(user_id).scores.get((media_type, source, media_id))

    def _library(self, user_id: int) -> FakeLibrary:
        if user_id not in self.libraries:
            raise LibraryStoreNotFoundError(user_id)
        return self.libraries[user_id]
