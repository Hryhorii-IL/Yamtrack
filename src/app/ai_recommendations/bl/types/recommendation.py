from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class RecommendedTitle:
    """A title as the LLM recommended it, before it's found on a provider."""

    title: str
    year: int | None = None


@dataclass(slots=True, frozen=True)
class ResolvedTitle:
    """A recommended title matched to a search result of the default source."""

    media_id: str
    source: str
    media_type: str
    title: str
    image: str
    release_year: int | None = None

    @property
    def key(self) -> tuple[str, str]:
        """Return the identity used to drop duplicate recommendations."""
        return self.source, self.media_id

    def to_card(self) -> dict[str, object]:
        """Return the search result shape shared media cards and helpers expect."""
        return {
            "media_id": self.media_id,
            "source": self.source,
            "media_type": self.media_type,
            "title": self.title,
            "image": self.image,
            "release_year": self.release_year,
        }


@dataclass(slots=True, frozen=True)
class MediaDetails:
    """The viewed media that single media recommendations should resemble."""

    media_id: str
    source: str
    title: str
    genres: tuple[str, ...] = ()
    synopsis: str = ""
    user_score: float | None = None
