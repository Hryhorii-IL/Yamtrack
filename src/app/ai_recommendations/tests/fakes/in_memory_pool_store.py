from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from app.ai_recommendations.bl.errors import PoolStoreLockError

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from app.ai_recommendations.bl.types.pool import (
        MediaRecommendations,
        QueueKind,
        RecommendationPool,
    )

type PoolKey = tuple[int, str]
type MediaKey = tuple[int, str, str, str]


class InMemoryPoolStore:
    """A PoolStore keeping everything in dicts, for facade tests."""

    def __init__(self) -> None:
        """Start empty."""
        self.pools: dict[PoolKey, RecommendationPool] = {}
        self.failed: set[PoolKey] = set()
        self.media: dict[MediaKey, MediaRecommendations] = {}
        self.locks: set[object] = set()
        self.queued: set[tuple[QueueKind, int, str]] = set()
        self._transaction_locks: list[object] | None = None
        self._callbacks: list[Callable[[], None]] = []

    @contextmanager
    def transaction(self) -> Generator[None]:
        """Release the locks at the end and run callbacks only on success."""
        self._transaction_locks, self._callbacks = [], []
        try:
            yield
        except BaseException:
            self._callbacks = []
            raise
        finally:
            for lock in self._transaction_locks or ():
                self.locks.discard(lock)
            self._transaction_locks = None
        callbacks, self._callbacks = self._callbacks, []
        for callback in callbacks:
            callback()

    def on_commit(self, callback: Callable[[], None]) -> None:
        """Run ``callback`` after the current transaction succeeds."""
        self._callbacks.append(callback)

    def lock_pool(self, user_id: int, media_type: str) -> None:
        """Lock a media type's pool until the transaction ends."""
        self._lock(("pool", user_id, media_type))

    def lock_media(
        self,
        user_id: int,
        media_type: str,
        source: str,
        media_id: str,
    ) -> None:
        """Lock the recommendations similar to a media until the transaction ends."""
        self._lock(("media", user_id, media_type, source, media_id))

    def load_pool(self, user_id: int, media_type: str) -> RecommendationPool | None:
        """Return the saved pool of a media type."""
        return self.pools.get((user_id, media_type))

    def save_pool(
        self,
        user_id: int,
        media_type: str,
        pool: RecommendationPool,
    ) -> None:
        """Save a complete pool."""
        self._save(user_id, media_type, pool, failed=False)

    def save_partial_pool(
        self,
        user_id: int,
        media_type: str,
        pool: RecommendationPool,
    ) -> None:
        """Save a pool that's still being generated."""
        self._save(user_id, media_type, pool, failed=False)

    def save_failed_pool(
        self,
        user_id: int,
        media_type: str,
        pool: RecommendationPool,
    ) -> None:
        """Save a failed pool."""
        self._save(user_id, media_type, pool, failed=True)

    def delete_pool(self, user_id: int, media_type: str) -> None:
        """Drop the pool of a media type."""
        self.pools.pop((user_id, media_type), None)
        self.failed.discard((user_id, media_type))

    def is_current(
        self,
        user_id: int,
        media_type: str,
        pool: RecommendationPool | None,
    ) -> bool:
        """Return whether ``pool`` is still the saved one."""
        current = self.load_pool(user_id, media_type)
        current_version = None if current is None else current.version
        return current_version == (None if pool is None else pool.version)

    def load_media(
        self,
        user_id: int,
        media_type: str,
        source: str,
        media_id: str,
    ) -> MediaRecommendations | None:
        """Return the recommendations similar to a media."""
        return self.media.get((user_id, media_type, source, media_id))

    def save_media(
        self,
        user_id: int,
        media_type: str,
        source: str,
        media_id: str,
        recommendations: MediaRecommendations,
    ) -> None:
        """Save the recommendations similar to a media."""
        self.media[user_id, media_type, source, media_id] = recommendations

    def mark_queued(self, kind: QueueKind, user_id: int, media_type: str) -> bool:
        """Mark a task as queued, returning False if it already was."""
        key = (kind, user_id, media_type)
        if key in self.queued:
            return False
        self.queued.add(key)
        return True

    def keep_queued(self, kind: QueueKind, user_id: int, media_type: str) -> None:
        """Keep a task marked as queued."""
        self.queued.add((kind, user_id, media_type))

    def clear_queued(self, kind: QueueKind, user_id: int, media_type: str) -> None:
        """Mark a task as no longer queued."""
        self.queued.discard((kind, user_id, media_type))

    def is_background_queued(self, user_id: int, media_type: str) -> bool:
        """Return whether a generation or extension is queued or running."""
        return any(key[1:] == (user_id, media_type) for key in self.queued)

    def invalidate_user(self, user_id: int) -> None:
        """Drop every pool and media recommendations of the user."""
        for key in [key for key in self.pools if key[0] == user_id]:
            self.delete_pool(*key)
        self.media = {
            key: value for key, value in self.media.items() if key[0] != user_id
        }

    def invalidate_media_type(self, user_id: int, media_type: str) -> None:
        """Drop the user's pool and media recommendations of a media type."""
        self.delete_pool(user_id, media_type)
        self.media = {
            key: value
            for key, value in self.media.items()
            if key[:2] != (user_id, media_type)
        }

    def _save(
        self,
        user_id: int,
        media_type: str,
        pool: RecommendationPool,
        *,
        failed: bool,
    ) -> None:
        self.pools[user_id, media_type] = pool
        if failed:
            self.failed.add((user_id, media_type))
        else:
            self.failed.discard((user_id, media_type))

    def _lock(self, lock: object) -> None:
        if self._transaction_locks is None:
            msg = "Locks are only taken inside a transaction"
            raise RuntimeError(msg)
        if lock in self.locks:
            raise PoolStoreLockError(lock)
        self.locks.add(lock)
        self._transaction_locks.append(lock)
