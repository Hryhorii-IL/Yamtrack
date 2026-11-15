from __future__ import annotations

from typing import TYPE_CHECKING

from app.ai_recommendations.bl.calculations.pool import (
    is_growing,
    needs_extension,
    remaining_slots,
)
from app.ai_recommendations.bl.stores import make_pool_store
from app.ai_recommendations.bl.types.pool import PoolPage

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from app.ai_recommendations.bl.stores.pool_store import PoolStore
    from app.ai_recommendations.bl.types.recommendation import ResolvedTitle


def get_pool_page[CardT](
    user_id: int,
    media_type: str,
    *,
    offset: int,
    shown: int,
    version: str,
    visible: Callable[[Sequence[ResolvedTitle]], Sequence[CardT]],
    store: PoolStore | None = None,
) -> PoolPage[CardT] | None:
    """Return the cards to append after ``offset`` pool items were handled.

    ``visible`` turns titles into the cards the user can see, dropping media
    the user already tracks. Return None when the pool changed since
    ``version``, so the page stops polling.
    """
    store = store or make_pool_store()
    pool = store.load_pool(user_id, media_type)
    if pool is None or pool.version != version:
        return None

    results = tuple(visible(pool.items[offset:])[: remaining_slots(shown)])
    shown += len(results)
    return PoolPage(
        results=results,
        shown=shown,
        offset=len(pool.items),
        version=pool.version,
        growing=is_growing(
            pool,
            shown=shown,
            background_queued=store.is_background_queued(user_id, media_type),
        ),
        partial=pool.partial,
        error=pool.error,
    )


def pool_needs_extension(
    user_id: int,
    media_type: str,
    *,
    store: PoolStore | None = None,
) -> bool:
    """Return whether the saved pool should get another batch."""
    pool = (store or make_pool_store()).load_pool(user_id, media_type)
    return pool is not None and needs_extension(pool)
