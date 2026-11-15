from __future__ import annotations

import re
from typing import TYPE_CHECKING

from unidecode import unidecode

if TYPE_CHECKING:
    from collections.abc import Sequence

    from app.ai_recommendations.bl.types.recommendation import (
        RecommendedTitle,
        ResolvedTitle,
    )


def best_match(
    results: Sequence[ResolvedTitle],
    recommendation: RecommendedTitle,
) -> ResolvedTitle | None:
    """Return the search result with the recommended title, if any."""
    title = normalize_title(recommendation.title)
    same_title = [
        result for result in results if normalize_title(result.title) == title
    ]
    for result in same_title:
        if recommendation.year is None or result.release_year in (
            None,
            recommendation.year,
        ):
            return result
    # A different title would be a different media
    return same_title[0] if same_title else None


def normalize_title(title: str) -> str:
    """Return a lowercase ASCII title with only letters and digits."""
    return re.sub(r"[^a-z0-9]", "", unidecode(title).lower())
