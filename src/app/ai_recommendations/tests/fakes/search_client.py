from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from app.ai_recommendations.bl.types.recommendation import (
        RecommendedTitle,
        ResolvedTitle,
    )


class FakeSearchClient:
    """Finds the recommended titles that are in its catalog."""

    def __init__(self, catalog: Iterable[ResolvedTitle] = ()) -> None:
        """Keep the titles that can be found."""
        self.catalog = {title.title: title for title in catalog}

    def resolve(
        self,
        recommendations: Iterable[RecommendedTitle],
        media_type: str,  # noqa: ARG002
    ) -> Iterator[ResolvedTitle]:
        """Yield the catalog entries of the recommended titles."""
        for recommendation in recommendations:
            if recommendation.title in self.catalog:
                yield self.catalog[recommendation.title]
