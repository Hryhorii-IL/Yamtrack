from django.core.cache import cache
from django.test import TestCase

from app.ai_recommendations.bl.queries.pool import get_pool_page, pool_needs_extension
from app.ai_recommendations.bl.types.constants import POOL_SIZE
from app.ai_recommendations.bl.types.pool import RecommendationPool
from app.ai_recommendations.tests.factories import pool_store
from app.ai_recommendations.tests.fakes import NOW, resolved


def visible(titles):
    """Show every title as its card."""
    return [title.to_card() for title in titles]


class PoolNeedsExtensionTests(TestCase):
    """Test pool_needs_extension."""

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

    def test_pool_needs_extension(self):
        """Extend small pools only."""
        self.assertFalse(pool_needs_extension(1, "movie", store=self.store))
        self.save(2)
        self.assertTrue(pool_needs_extension(1, "movie", store=self.store))
        self.save(POOL_SIZE)
        self.assertFalse(pool_needs_extension(1, "movie", store=self.store))
