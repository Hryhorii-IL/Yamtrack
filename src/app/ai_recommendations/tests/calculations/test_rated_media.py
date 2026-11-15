from datetime import timedelta
from unittest import TestCase

from app.ai_recommendations.bl.calculations.rated_media import rated_line, select_rated
from app.ai_recommendations.bl.types.constants import LIKED_LIMIT
from app.ai_recommendations.bl.types.taste import RatedTitle, TasteSettings
from app.ai_recommendations.tests.fakes import NOW

SETTINGS = TasteSettings(
    include_liked=True,
    liked_min_score=7,
    include_disliked=True,
    disliked_max_score=4,
)


def rated(item_id, score, *, days_ago=0, title=None, media_type="movie"):
    """Return a rated title."""
    return RatedTitle(
        item_id=item_id,
        title=title or f"Title {item_id}",
        media_type=media_type,
        score=score,
        created_at=NOW - timedelta(days=days_ago),
    )


class SelectRatedTests(TestCase):
    """Test choosing the liked and disliked titles of the prompt."""

    def test_split_and_order(self):
        """Put the best liked titles and the worst disliked titles first."""
        liked, disliked = select_rated(
            [rated(1, 8), rated(2, 10), rated(3, 5), rated(4, 3), rated(5, 1)],
            SETTINGS,
        )

        self.assertEqual([title.item_id for title in liked], [2, 1])
        self.assertEqual([title.item_id for title in disliked], [5, 4])

    def test_latest_rating_of_repeated_entries(self):
        """Use the latest rating when an item was rated several times."""
        liked, disliked = select_rated(
            [rated(1, 9, days_ago=10), rated(1, 2, days_ago=1)],
            SETTINGS,
        )

        self.assertEqual(liked, ())
        self.assertEqual([title.score for title in disliked], [2])

    def test_excluded_kinds(self):
        """Skip liked or disliked titles the user doesn't want to send."""
        settings = TasteSettings(
            include_liked=False,
            liked_min_score=7,
            include_disliked=True,
            disliked_max_score=4,
        )

        liked, disliked = select_rated([rated(1, 9), rated(2, 1)], settings)

        self.assertEqual(liked, ())
        self.assertEqual([title.item_id for title in disliked], [2])

    def test_nothing_included(self):
        """Return no titles when both kinds are disabled."""
        settings = TasteSettings(
            include_liked=False,
            liked_min_score=7,
            include_disliked=False,
            disliked_max_score=4,
        )

        self.assertEqual(select_rated([rated(1, 9)], settings), ((), ()))

    def test_limit(self):
        """Send at most LIKED_LIMIT liked titles."""
        liked, _ = select_rated(
            [rated(item_id, 9) for item_id in range(LIKED_LIMIT + 5)],
            SETTINGS,
        )

        self.assertEqual(len(liked), LIKED_LIMIT)

    def test_rated_line(self):
        """Describe the title, its type and the score."""
        self.assertEqual(
            rated_line(rated(1, 7.5, title="Dune", media_type="tv")),
            "- Dune (TV show) — 7.5/10",
        )
