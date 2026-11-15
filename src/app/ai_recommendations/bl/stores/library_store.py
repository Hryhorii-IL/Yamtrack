from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from app.ai_recommendations.bl.types.taste import TasteInput


class LibraryStore(Protocol):
    """The user's library and AI preferences, read as plain types.

    Methods raise LibraryStoreNotFoundError when the user doesn't exist.
    """

    def load_enabled_media_types(self, user_id: int) -> tuple[str, ...]:
        """Return the AI media types enabled in the user's preferences."""
        ...

    def has_media(self, user_id: int, media_type: str) -> bool:
        """Return whether the user tracks anything of a media type."""
        ...

    def load_taste(self, user_id: int, media_type: str) -> TasteInput:
        """Return the ratings, library and wishes describing the user's taste."""
        ...

    def load_user_score(
        self,
        user_id: int,
        media_type: str,
        source: str,
        media_id: str,
    ) -> float | None:
        """Return the user's latest score of a media, if rated."""
        ...
