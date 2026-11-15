from __future__ import annotations

from typing import TYPE_CHECKING, Final

from app.ai_recommendations.bl.calculations.rated_media import rated_line, select_rated
from app.ai_recommendations.bl.types.constants import (
    EXTRA_REQUESTED,
    GLOBAL_COUNT,
    MEDIA_COUNT,
    SYNOPSIS_LIMIT,
    TYPE_LABELS,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from app.ai_recommendations.bl.types.recommendation import MediaDetails
    from app.ai_recommendations.bl.types.taste import TasteInput

SYSTEM_INSTRUCTION: Final = """\
You are a recommendation engine for a personal media tracker.
You recommend real, existing titles only. Never invent titles.
Use the exact official title as it appears on {provider_name} so it can be found by search.
Never recommend anything from the "Already in library" list.
Do not recommend sequels or later seasons unless the user has rated the earlier entry highly.
Prefer variety: mix well-known and lesser-known titles, different years and subgenres.
The user's explicit wishes and exclusions override everything else.
Respond only with JSON matching the schema."""  # noqa: E501

GLOBAL_PROMPT: Final = """\
Recommend {count} {type_plural} for this user.

The user's ratings use a 0-10 scale.

{taste}

Base recommendations mostly on what the user liked. Avoid anything similar to what they disliked.
Ratings from other media types are signals of overall taste (themes, tone, genres)."""  # noqa: E501

MEDIA_PROMPT: Final = """\
The user is viewing this {type_singular}:
{media_details}

Recommend {count} {type_plural} that are as close as possible to this one: the same themes, tone, genre and style.

The user's ratings use a 0-10 scale.

{taste}

Similarity to the viewed title is what matters. Don't pick titles just because they match the user's liked titles; use those only to break ties between equally similar titles.
Never recommend a title the user disliked, nor anything close to the titles they disliked."""  # noqa: E501

LIKED_BLOCK: Final = "Titles the user liked (score >= {score:g}):\n{lines}"
DISLIKED_BLOCK: Final = "Titles the user disliked (score <= {score:g}):\n{lines}"
LIBRARY_BLOCK: Final = "Already in library (do not recommend):\n{lines}"
RECOMMENDED_BLOCK: Final = "Already recommended (do not repeat):\n{lines}"
POSITIVE_BLOCK: Final = "The user's own wishes (follow them):\n{text}"
NEGATIVE_BLOCK: Final = (
    "The user explicitly does not want (never recommend anything matching this):"
    "\n{text}"
)
NO_RATINGS: Final = "The user has no ratings yet; recommend widely acclaimed titles."


def system_instruction(provider_name: str) -> str:
    """Return the system instruction naming the source titles are searched on."""
    return SYSTEM_INSTRUCTION.format(provider_name=provider_name)


def taste_text(taste: TasteInput) -> str:
    """Return the prompt blocks describing the user's taste and library."""
    liked, disliked = select_rated(taste.rated, taste.settings)

    blocks = []
    if liked:
        blocks.append(
            LIKED_BLOCK.format(
                score=taste.settings.liked_min_score,
                lines="\n".join(rated_line(title) for title in liked),
            ),
        )
    if disliked:
        blocks.append(
            DISLIKED_BLOCK.format(
                score=taste.settings.disliked_max_score,
                lines="\n".join(rated_line(title) for title in disliked),
            ),
        )
    if not blocks:
        blocks.append(NO_RATINGS)

    if taste.library:
        blocks.append(LIBRARY_BLOCK.format(lines=bullet_lines(taste.library)))

    positive = taste.positive.strip()
    if positive:
        blocks.append(POSITIVE_BLOCK.format(text=positive))
    negative = taste.negative.strip()
    if negative:
        blocks.append(NEGATIVE_BLOCK.format(text=negative))

    return "\n\n".join(blocks)


def global_prompt(
    media_type: str,
    taste: TasteInput,
    recommended: Iterable[str] = (),
) -> str:
    """Return the prompt for a batch of a media type's global recommendations.

    ``recommended`` titles are already in the pool, so the batch must add others.
    """
    taste_blocks = taste_text(taste)
    recommended_lines = bullet_lines(recommended)
    if recommended_lines:
        taste_blocks += "\n\n" + RECOMMENDED_BLOCK.format(lines=recommended_lines)
    return GLOBAL_PROMPT.format(
        count=GLOBAL_COUNT + EXTRA_REQUESTED,
        type_plural=TYPE_LABELS[media_type][1],
        taste=taste_blocks,
    )


def media_prompt(media_type: str, details: MediaDetails, taste: TasteInput) -> str:
    """Return the prompt for recommendations similar to a single media."""
    type_singular, type_plural = TYPE_LABELS[media_type]
    return MEDIA_PROMPT.format(
        type_singular=type_singular,
        media_details=media_details_text(details),
        count=MEDIA_COUNT + EXTRA_REQUESTED,
        type_plural=type_plural,
        taste=taste_text(taste),
    )


def media_details_text(details: MediaDetails) -> str:
    """Return the prompt lines describing the viewed media."""
    lines = [f"Title: {details.title}"]
    if details.genres:
        lines.append(f"Genres: {', '.join(details.genres)}")
    if details.synopsis:
        lines.append(f"Synopsis: {details.synopsis[:SYNOPSIS_LIMIT]}")
    if details.user_score is not None:
        lines.append(f"The user rated it {details.user_score:g}/10.")
    return "\n".join(lines)


def bullet_lines(titles: Iterable[str]) -> str:
    """Return the titles as a prompt bullet list."""
    return "\n".join(f"- {title}" for title in titles)
