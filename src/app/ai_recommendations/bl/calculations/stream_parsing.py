from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Final

from app.ai_recommendations.bl.types.recommendation import RecommendedTitle

if TYPE_CHECKING:
    from collections.abc import Iterator

ITEMS_START: Final = re.compile(r'"items"\s*:\s*\[')
SEPARATORS: Final = " \t\r\n,"


class ItemStreamParser:
    """Extract complete recommendations from a streamed ``{"items": [...]}`` JSON.

    LLMs stream the response in arbitrary chunks; yielding each item as soon as
    it's complete lets titles be resolved while the rest is still generated.
    """

    def __init__(self) -> None:
        """Start with an empty buffer."""
        self.buffer = ""
        self.position: int | None = None
        self.decoder = json.JSONDecoder()

    def feed(self, text: str) -> Iterator[RecommendedTitle]:
        """Add a chunk of JSON text and yield the newly completed items."""
        self.buffer += text
        if self.position is None:
            match = ITEMS_START.search(self.buffer)
            if match is None:
                return
            self.position = match.end()

        while True:
            position = self.position
            while position < len(self.buffer) and self.buffer[position] in SEPARATORS:
                position += 1
            self.position = position
            if position >= len(self.buffer) or self.buffer[position] == "]":
                return
            try:
                item, self.position = self.decoder.raw_decode(self.buffer, position)
            except json.JSONDecodeError:
                # The item is incomplete, wait for the next chunk
                return
            recommendation = to_recommended_title(item)
            if recommendation is not None:
                yield recommendation


def to_recommended_title(item: object) -> RecommendedTitle | None:
    """Return the recommendation of a parsed item, or None if it has no title."""
    if not isinstance(item, dict):
        return None
    title = item.get("title")
    if not isinstance(title, str):
        return None
    year = item.get("year")
    # A wrong year only weakens matching, so the title is still worth keeping
    valid_year = isinstance(year, int) and not isinstance(year, bool)
    return RecommendedTitle(title=title, year=year if valid_year else None)
