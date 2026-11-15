from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Final, Protocol, cast

from django.core.cache import caches

from app.ai_recommendations.bl.errors import PoolStoreLockError
from app.ai_recommendations.bl.types.constants import CACHE_TIMEOUT, ERROR_TIMEOUT
from app.ai_recommendations.bl.types.pool import QueueKind

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable

    from app.ai_recommendations.bl.types.pool import (
        MediaRecommendations,
        RecommendationPool,
    )

QUEUE_PREFIXES: Final = {
    QueueKind.GENERATION: "ai_generate",
    QueueKind.EXTENSION: "ai_extend",
}


class PatternCache(Protocol):
    """The django-redis cache API the store relies on."""

    def get(self, key: str) -> Any: ...  # noqa: D102
    def set(self, key: str, value: object, timeout: int) -> None: ...  # noqa: D102
    def add(self, key: str, value: object, timeout: int) -> bool: ...  # noqa: D102
    def delete(self, key: str) -> None: ...  # noqa: D102
    def get_many(self, keys: Iterable[str]) -> dict[str, Any]: ...  # noqa: D102
    def delete_pattern(self, pattern: str) -> int: ...  # noqa: D102


class DjangoCachePoolStore:
    """Keeps pools in the Django cache only.

    They are cheap to regenerate and expire anyway, so a cache "transaction"
    only scopes locks and post-commit callbacks; writes apply immediately.
    """

    def __init__(self, *, lock_timeout: int, cache: PatternCache | None = None) -> None:
        """Keep how long locks, queued flags and partial pools may live."""
        self.lock_timeout = lock_timeout
        # The default cache is django-redis, which adds delete_pattern
        self.cache = cache or cast("PatternCache", caches["default"])
        self._locks: list[tuple[str, str]] | None = None
        self._callbacks: list[Callable[[], None]] = []

    @contextmanager
    def transaction(self) -> Generator[None]:
        """Release the locks at the end and run callbacks only on success."""
        self._locks, self._callbacks = [], []
        try:
            yield
        except BaseException:
            self._callbacks = []
            raise
        finally:
            self._release_locks()
        callbacks, self._callbacks = self._callbacks, []
        for callback in callbacks:
            callback()

    def on_commit(self, callback: Callable[[], None]) -> None:
        """Run ``callback`` after the current transaction succeeds."""
        self._callbacks.append(callback)

    def lock_pool(self, user_id: int, media_type: str) -> None:
        """Lock a media type's pool until the transaction ends."""
        self._lock(pool_key(user_id, media_type))

    def lock_media(
        self, user_id: int, media_type: str, source: str, media_id: str
    ) -> None:
        """Lock the recommendations similar to a media until the transaction ends."""
        self._lock(media_key(user_id, media_type, source, media_id))

    def load_pool(self, user_id: int, media_type: str) -> RecommendationPool | None:
        """Return the cached pool of a media type."""
        return self.cache.get(pool_key(user_id, media_type))

    def save_pool(
        self, user_id: int, media_type: str, pool: RecommendationPool
    ) -> None:
        """Cache a complete pool."""
        self.cache.set(pool_key(user_id, media_type), pool, CACHE_TIMEOUT)

    def save_partial_pool(
        self,
        user_id: int,
        media_type: str,
        pool: RecommendationPool,
    ) -> None:
        """Cache a pool that's still being generated."""
        # Expires with the lock, so a pool left by a killed worker is regenerated
        self.cache.set(pool_key(user_id, media_type), pool, self.lock_timeout)

    def save_failed_pool(
        self,
        user_id: int,
        media_type: str,
        pool: RecommendationPool,
    ) -> None:
        """Cache a failed pool briefly, so pages show the error instead of polling."""
        self.cache.set(pool_key(user_id, media_type), pool, ERROR_TIMEOUT)

    def delete_pool(self, user_id: int, media_type: str) -> None:
        """Drop the cached pool of a media type."""
        self.cache.delete(pool_key(user_id, media_type))

    def is_current(
        self,
        user_id: int,
        media_type: str,
        pool: RecommendationPool | None,
    ) -> bool:
        """Return whether ``pool`` is still the cached one, i.e. not invalidated."""
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
        """Return the cached recommendations similar to a media."""
        return self.cache.get(media_key(user_id, media_type, source, media_id))

    def save_media(
        self,
        user_id: int,
        media_type: str,
        source: str,
        media_id: str,
        recommendations: MediaRecommendations,
    ) -> None:
        """Cache the recommendations similar to a media."""
        self.cache.set(
            media_key(user_id, media_type, source, media_id),
            recommendations,
            CACHE_TIMEOUT,
        )

    def mark_queued(self, kind: QueueKind, user_id: int, media_type: str) -> bool:
        """Mark a task as queued, returning False if it already was."""
        return self.cache.add(
            queued_key(kind, user_id, media_type),
            1,
            self.lock_timeout,
        )

    def keep_queued(self, kind: QueueKind, user_id: int, media_type: str) -> None:
        """Keep a task marked as queued while it's retried or queues its next run."""
        self.cache.set(queued_key(kind, user_id, media_type), 1, self.lock_timeout)

    def clear_queued(self, kind: QueueKind, user_id: int, media_type: str) -> None:
        """Mark a task as no longer queued."""
        self.cache.delete(queued_key(kind, user_id, media_type))

    def is_background_queued(self, user_id: int, media_type: str) -> bool:
        """Return whether a pool generation or extension is queued or running."""
        keys = [queued_key(kind, user_id, media_type) for kind in QueueKind]
        return bool(self.cache.get_many(keys))

    def invalidate_user(self, user_id: int) -> None:
        """Drop every cached recommendation of the user."""
        self.cache.delete_pattern(f"ai_pool_{user_id}_*")
        self.cache.delete_pattern(f"ai_media_{user_id}_*")

    def invalidate_media_type(self, user_id: int, media_type: str) -> None:
        """Drop the user's cached recommendations of a media type."""
        self.delete_pool(user_id, media_type)
        self.cache.delete_pattern(f"ai_media_{user_id}_{media_type}_*")

    def _lock(self, key: str) -> None:
        """Take the lock of ``key`` for the current transaction."""
        if self._locks is None:
            msg = "Locks are only taken inside a transaction"
            raise RuntimeError(msg)
        # Kept outside the pool prefixes so invalidation doesn't release it
        lock_key = f"ai_lock_{key}"
        token = uuid.uuid4().hex
        if not self.cache.add(lock_key, token, self.lock_timeout):
            raise PoolStoreLockError(lock_key)
        self._locks.append((lock_key, token))

    def _release_locks(self) -> None:
        """Release the transaction's locks that weren't taken over meanwhile."""
        for lock_key, token in self._locks or ():
            # Don't release a lock that expired and was taken by another task
            if self.cache.get(lock_key) == token:
                self.cache.delete(lock_key)
        self._locks = None


def pool_key(user_id: int, media_type: str) -> str:
    """Return the cache key of a media type's pool."""
    return f"ai_pool_{user_id}_{media_type}"


def media_key(user_id: int, media_type: str, source: str, media_id: str) -> str:
    """Return the cache key of the recommendations similar to a media."""
    return f"ai_media_{user_id}_{media_type}_{source}_{media_id}"


def queued_key(kind: QueueKind, user_id: int, media_type: str) -> str:
    """Return the cache key marking a queued pool task."""
    return f"{QUEUE_PREFIXES[kind]}_{user_id}_{media_type}"
