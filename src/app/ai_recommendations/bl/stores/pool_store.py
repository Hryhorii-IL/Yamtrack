from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

    from app.ai_recommendations.bl.types.pool import (
        MediaRecommendations,
        QueueKind,
        RecommendationPool,
    )


class PoolStore(Protocol):
    """Recommendation pools, their locks and queued background tasks.

    ``transaction()`` holds the locks taken inside it and runs ``on_commit``
    callbacks only when it exits without an error.
    """

    def transaction(self) -> AbstractContextManager[None]:
        """Group the writes and locks of a use case."""
        ...

    def on_commit(self, callback: Callable[[], None]) -> None:
        """Run ``callback`` after the current transaction succeeds."""
        ...

    def lock_pool(self, user_id: int, media_type: str) -> None:
        """Lock a media type's pool until the transaction ends.

        Raises PoolStoreLockError when another request or task holds it.
        """
        ...

    def lock_media(
        self, user_id: int, media_type: str, source: str, media_id: str
    ) -> None:
        """Lock the recommendations similar to a media until the transaction ends."""
        ...

    def load_pool(self, user_id: int, media_type: str) -> RecommendationPool | None:
        """Return the cached pool of a media type."""
        ...

    def save_pool(
        self, user_id: int, media_type: str, pool: RecommendationPool
    ) -> None:
        """Save a complete pool."""
        ...

    def save_partial_pool(
        self,
        user_id: int,
        media_type: str,
        pool: RecommendationPool,
    ) -> None:
        """Save a pool that's still being generated, only as long as a lock lives."""
        ...

    def save_failed_pool(
        self,
        user_id: int,
        media_type: str,
        pool: RecommendationPool,
    ) -> None:
        """Save a failed pool briefly, so pages show the error instead of polling."""
        ...

    def delete_pool(self, user_id: int, media_type: str) -> None:
        """Drop the pool of a media type."""
        ...

    def is_current(
        self,
        user_id: int,
        media_type: str,
        pool: RecommendationPool | None,
    ) -> bool:
        """Return whether ``pool`` is still the saved one, i.e. not invalidated."""
        ...

    def load_media(
        self,
        user_id: int,
        media_type: str,
        source: str,
        media_id: str,
    ) -> MediaRecommendations | None:
        """Return the recommendations similar to a media."""
        ...

    def save_media(
        self,
        user_id: int,
        media_type: str,
        source: str,
        media_id: str,
        recommendations: MediaRecommendations,
    ) -> None:
        """Save the recommendations similar to a media."""
        ...

    def mark_queued(self, kind: QueueKind, user_id: int, media_type: str) -> bool:
        """Mark a task as queued, returning False if it already was."""
        ...

    def keep_queued(self, kind: QueueKind, user_id: int, media_type: str) -> None:
        """Keep a task marked as queued while it's retried or queues its next run."""
        ...

    def clear_queued(self, kind: QueueKind, user_id: int, media_type: str) -> None:
        """Mark a task as no longer queued."""
        ...

    def is_background_queued(self, user_id: int, media_type: str) -> bool:
        """Return whether a generation or extension is queued or running."""
        ...

    def invalidate_user(self, user_id: int) -> None:
        """Drop every pool and media recommendations of the user."""
        ...

    def invalidate_media_type(self, user_id: int, media_type: str) -> None:
        """Drop the user's pool and media recommendations of a media type."""
        ...
