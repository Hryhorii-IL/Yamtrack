from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from app.ai_recommendations.bl.types.constants import (
    MAX_EXTENSIONS,
    POOL_SHOWN,
    POOL_SIZE,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from app.ai_recommendations.bl.types.pool import RecommendationPool
    from app.ai_recommendations.bl.types.recommendation import ResolvedTitle


def needs_extension(pool: RecommendationPool) -> bool:
    """Return whether a finished pool should get another batch."""
    return (
        not pool.partial
        and not pool.error
        and len(pool.items) < POOL_SIZE
        and pool.extensions < MAX_EXTENSIONS
    )


def with_item(pool: RecommendationPool, item: ResolvedTitle) -> RecommendationPool:
    """Return the pool with one more resolved item."""
    return dataclasses.replace(pool, items=(*pool.items, item))


def with_batch(
    pool: RecommendationPool,
    items: Iterable[ResolvedTitle],
) -> RecommendationPool:
    """Return the pool extended with the batch items it doesn't hold yet."""
    known = {item.key for item in pool.items}
    new_items = tuple(item for item in items if item.key not in known)
    return dataclasses.replace(
        pool,
        items=pool.items + new_items,
        extensions=pool.extensions + 1,
    )


def remaining_slots(shown: int) -> int:
    """Return how many more cards fit on the page."""
    return max(POOL_SHOWN - shown, 0)


def is_growing(
    pool: RecommendationPool, *, shown: int, background_queued: bool
) -> bool:
    """Return whether the page should keep polling for more cards."""
    if shown >= POOL_SHOWN:
        return False
    return pool.partial or (needs_extension(pool) and background_queued)


def without_media(
    items: Iterable[ResolvedTitle],
    source: str,
    media_id: str,
) -> tuple[ResolvedTitle, ...]:
    """Return the items without the viewed media itself."""
    return tuple(item for item in items if item.key != (source, media_id))
