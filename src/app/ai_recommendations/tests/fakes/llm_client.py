from __future__ import annotations

from typing import TYPE_CHECKING

from app.ai_recommendations.bl.types.recommendation import RecommendedTitle

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


class FakeLLMClient:
    """Recommends scripted titles, then raises a scripted error."""

    def __init__(
        self,
        titles: Iterable[str] = (),
        error: Exception | None = None,
    ) -> None:
        """Keep the titles to recommend and the error to raise after them."""
        self.titles = list(titles)
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def recommend(
        self,
        system_instruction: str,
        prompt: str,
    ) -> Iterator[RecommendedTitle]:
        """Yield the scripted titles, then raise the scripted error."""
        self.calls.append((system_instruction, prompt))
        for title in self.titles:
            yield RecommendedTitle(title=title)
        if self.error is not None:
            raise self.error
