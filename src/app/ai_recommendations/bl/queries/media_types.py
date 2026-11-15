from __future__ import annotations

from typing import TYPE_CHECKING

from app.ai_recommendations.bl.stores import make_library_store
from app.ai_recommendations.bl.types.constants import MEDIA_TYPES

if TYPE_CHECKING:
    from app.ai_recommendations.bl.stores.library_store import LibraryStore


def get_media_types(
    user_id: int,
    *,
    store: LibraryStore | None = None,
) -> tuple[str, ...]:
    """Return the enabled media types the user has tracked anything of."""
    store = store or make_library_store()
    return tuple(
        media_type
        for media_type in store.load_enabled_media_types(user_id)
        if store.has_media(user_id, media_type)
    )


def is_media_type_available(
    user_id: int,
    media_type: str,
    *,
    store: LibraryStore | None = None,
) -> bool:
    """Return whether the user can get AI recommendations for a media type."""
    store = store or make_library_store()
    return media_type in store.load_enabled_media_types(user_id) and store.has_media(
        user_id,
        media_type,
    )


def is_supported_media_type(media_type: str) -> bool:
    """Return whether AI recommendations exist for a media type."""
    return media_type in MEDIA_TYPES
