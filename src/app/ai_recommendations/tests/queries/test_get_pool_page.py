from django.core.cache import cache
from django.test import TestCase

from app.ai_recommendations.bl.queries.pool import get_pool_page
from app.ai_recommendations.bl.types.constants import POOL_SHOWN
from app.ai_recommendations.bl.types.pool import QueueKind, RecommendationPool
from app.ai_recommendations.tests.factories import pool_store
from app.ai_recommendations.tests.fakes import NOW, resolved


def visible(titles):
    """Show every title as its card."""
    return [title.to_card() for title in titles]


class GetPoolPageTests(TestCase):
    """Test get_pool_page."""

    def setUp(self):
        """Use an empty cache."""
        cache.clear()
        self.store = pool_store()

    def save(self, count, **kwargs):
        """Save a movie pool of user 1 with ``count`` items."""
        pool = RecommendationPool(
            generated_at=NOW,
            items=tuple(resolved(str(media_id)) for media_id in range(count)),
            **kwargs,
        )
        self.store.save_pool(1, "movie", pool)
        return pool

    def page(self, pool, *, offset=0, shown=0, version=None):
        """Return the page of the saved pool."""
        return get_pool_page(
            1,
            "movie",
            offset=offset,
            shown=shown,
            version=pool.version if version is None else version,
            visible=visible,
            store=self.store,
        )

    def test_after_offset(self):
        """Return the cards after the offset and where the next page starts."""
        page = self.page(self.save(5), offset=3, shown=3)

        self.assertEqual([card["media_id"] for card in page.results], ["3", "4"])
        self.assertEqual((page.offset, page.shown), (5, 5))
        self.assertFalse(page.growing)

    def test_capped(self):
        """Stop at POOL_SHOWN cards and stop polling."""
        page = self.page(self.save(POOL_SHOWN + 5, partial=True))

        self.assertEqual(len(page.results), POOL_SHOWN)
        self.assertFalse(page.growing)

    def test_growing_while_queued(self):
        """Keep polling while a queued batch may add cards."""
        pool = self.save(2)
        self.store.mark_queued(QueueKind.EXTENSION, 1, "movie")

        self.assertTrue(self.page(pool).growing)

    def test_regenerated(self):
        """Return nothing when the pool changed since the page was rendered."""
        self.assertIsNone(self.page(self.save(2), version="old"))
