from __future__ import annotations

from app.ai_recommendations.bl.clients import llm_max_duration
from app.ai_recommendations.bl.stores.django_cache_pool_store import (
    DjangoCachePoolStore,
)
from app.ai_recommendations.bl.stores.django_library_store import DjangoLibraryStore
from app.ai_recommendations.bl.types.constants import LOCK_EXTRA_TIMEOUT


def make_pool_store() -> DjangoCachePoolStore:
    """Return the pool store, with locks outliving a generation that times out."""
    return DjangoCachePoolStore(lock_timeout=llm_max_duration() + LOCK_EXTRA_TIMEOUT)


def make_library_store() -> DjangoLibraryStore:
    """Return the store reading the user's library."""
    return DjangoLibraryStore()
