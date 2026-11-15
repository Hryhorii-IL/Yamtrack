from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from app.ai_recommendations.bl.types.recommendation import (
        RecommendedTitle,
        ResolvedTitle,
    )


class RecommendationSource(Protocol):
    """What services need from an LLM client, so tests can pass fakes."""

    def recommend(
        self,
        system_instruction: str,
        prompt: str,
        *,
        thinking: bool = False,
    ) -> Iterator[RecommendedTitle]:
        """Yield recommended titles as they are generated.

        ``thinking`` trades speed for better picks where the provider supports it.
        """
        ...


class TitleResolver(Protocol):
    """What services need from the media search client."""

    def resolve(
        self,
        recommendations: Iterable[RecommendedTitle],
        media_type: str,
    ) -> Iterator[ResolvedTitle]:
        """Yield the recommendations found on their provider."""
        ...
