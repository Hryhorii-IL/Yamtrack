from __future__ import annotations

from typing import TYPE_CHECKING

from app.ai_recommendations.bl.types.constants import (
    DISLIKED_LIMIT,
    LIKED_LIMIT,
    TYPE_LABELS,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from app.ai_recommendations.bl.types.taste import RatedTitle, TasteSettings


def select_rated(
    rated: Iterable[RatedTitle],
    settings: TasteSettings,
) -> tuple[tuple[RatedTitle, ...], tuple[RatedTitle, ...]]:
    """Return the liked titles (best first) and disliked ones (worst first)."""
    if not settings.include_liked and not settings.include_disliked:
        return (), ()

    latest_by_item: dict[int, RatedTitle] = {}
    for title in rated:
        # Repeated entries of an item count with their latest rating only
        existing = latest_by_item.get(title.item_id)
        if existing is None or title.created_at > existing.created_at:
            latest_by_item[title.item_id] = title

    ranked = sorted(latest_by_item.values(), key=lambda title: -title.score)
    liked = [
        title
        for title in ranked
        if settings.include_liked and title.score >= settings.liked_min_score
    ]
    disliked = [
        title
        for title in reversed(ranked)
        if title not in liked
        and settings.include_disliked
        and title.score <= settings.disliked_max_score
    ]
    return tuple(liked[:LIKED_LIMIT]), tuple(disliked[:DISLIKED_LIMIT])


def rated_line(title: RatedTitle) -> str:
    """Return the prompt line for a rated title."""
    type_singular = TYPE_LABELS[title.media_type][0]
    return f"- {title.title} ({type_singular}) — {title.score:g}/10"
