from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime


@dataclass(slots=True, frozen=True)
class RatedTitle:
    """A scored entry of the user's library."""

    item_id: int
    title: str
    media_type: str
    score: float
    created_at: datetime


@dataclass(slots=True, frozen=True)
class TasteSettings:
    """The user's choices about which ratings to send to the LLM."""

    include_liked: bool
    liked_min_score: float
    include_disliked: bool
    disliked_max_score: float


@dataclass(slots=True, frozen=True)
class TasteInput:
    """Everything the prompts need to know about the user's taste."""

    settings: TasteSettings
    rated: tuple[RatedTitle, ...] = ()
    library: tuple[str, ...] = ()
    positive: str = ""
    negative: str = ""
