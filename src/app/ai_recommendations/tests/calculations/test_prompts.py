from unittest import TestCase

from app.ai_recommendations.bl.calculations.prompts import (
    NO_RATINGS,
    global_prompt,
    media_details_text,
    media_prompt,
    system_instruction,
    taste_text,
)
from app.ai_recommendations.bl.types.constants import (
    EXTRA_REQUESTED,
    GLOBAL_COUNT,
    MEDIA_COUNT,
    SYNOPSIS_LIMIT,
)
from app.ai_recommendations.bl.types.recommendation import MediaDetails
from app.ai_recommendations.bl.types.taste import RatedTitle
from app.ai_recommendations.tests.fakes import NOW, taste


def rated(item_id, title, score):
    """Return a rated movie."""
    return RatedTitle(
        item_id=item_id,
        title=title,
        media_type="movie",
        score=score,
        created_at=NOW,
    )


class TasteTextTests(TestCase):
    """Test the prompt blocks describing the user's taste."""

    def test_no_ratings(self):
        """Ask for acclaimed titles when nothing is rated."""
        self.assertEqual(taste_text(taste()), NO_RATINGS)

    def test_all_blocks(self):
        """Describe ratings, the library and the user's wishes."""
        text = taste_text(
            taste(
                rated=(rated(1, "Great", 9), rated(2, "Bad", 2)),
                library=("Owned",),
                positive=" More sci-fi ",
                negative="No horror",
            ),
        )

        self.assertIn(
            "Titles the user liked (score >= 7):\n- Great (movie) — 9/10", text
        )
        self.assertIn(
            "Titles the user disliked (score <= 4):\n- Bad (movie) — 2/10", text
        )
        self.assertIn("Already in library (do not recommend):\n- Owned", text)
        self.assertIn("The user's own wishes (follow them):\nMore sci-fi", text)
        self.assertIn("never recommend anything matching this):\nNo horror", text)
        self.assertNotIn(NO_RATINGS, text)


class PromptTests(TestCase):
    """Test the full prompts sent to the LLM."""

    def test_system_instruction(self):
        """Name the source titles are searched on."""
        self.assertIn(
            "as it appears on The Movie Database",
            system_instruction("The Movie Database"),
        )

    def test_global_prompt(self):
        """Ask for a batch with spares and the user's taste."""
        prompt = global_prompt("movie", taste())

        count = GLOBAL_COUNT + EXTRA_REQUESTED
        self.assertTrue(prompt.startswith(f"Recommend {count} movies for this user."))
        self.assertIn(NO_RATINGS, prompt)
        self.assertNotIn("Already recommended", prompt)

    def test_global_prompt_excludes_recommended(self):
        """List the pool titles so a new batch adds others."""
        prompt = global_prompt("movie", taste(), recommended=["Dune"])

        self.assertIn("Already recommended (do not repeat):\n- Dune", prompt)

    def test_media_prompt(self):
        """Describe the viewed media first."""
        details = MediaDetails(media_id="1", source="tmdb", title="Viewed")

        prompt = media_prompt("movie", details, taste())

        self.assertTrue(
            prompt.startswith("The user is viewing this movie:\nTitle: Viewed")
        )
        self.assertIn(f"Recommend {MEDIA_COUNT + EXTRA_REQUESTED} movies", prompt)
        self.assertIn("Never recommend a title the user disliked", prompt)

    def test_media_details_text(self):
        """Include genres, a capped synopsis and the user's score."""
        details = MediaDetails(
            media_id="1",
            source="tmdb",
            title="Viewed",
            genres=("Drama", "Sci-Fi"),
            synopsis="x" * (SYNOPSIS_LIMIT + 10),
            user_score=7.0,
        )

        text = media_details_text(details)

        self.assertIn("Genres: Drama, Sci-Fi", text)
        self.assertIn(f"Synopsis: {'x' * SYNOPSIS_LIMIT}\n", text)
        self.assertTrue(text.endswith("The user rated it 7/10."))
