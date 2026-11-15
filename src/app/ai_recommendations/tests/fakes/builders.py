"""Builders of the values the AI recommendations tests share."""

from __future__ import annotations

from datetime import UTC, datetime

from django.conf import settings

from app.ai_recommendations.bl.types.recommendation import ResolvedTitle
from app.ai_recommendations.bl.types.taste import TasteInput, TasteSettings

NOW = datetime(2026, 1, 1, tzinfo=UTC)

ALL_RATINGS = TasteSettings(
    include_liked=True,
    liked_min_score=7,
    include_disliked=True,
    disliked_max_score=4,
)


def resolved(
    media_id: str,
    title: str | None = None,
    year: int | None = None,
) -> ResolvedTitle:
    """Return a resolved TMDB movie."""
    return ResolvedTitle(
        media_id=media_id,
        source="tmdb",
        media_type="movie",
        title=title or f"Movie {media_id}",
        image=settings.IMG_NONE,
        release_year=year,
    )


def taste(**kwargs: object) -> TasteInput:
    """Return a taste with every rating included and no wishes."""
    return TasteInput(settings=ALL_RATINGS, **kwargs)  # ty: ignore[invalid-argument-type]
