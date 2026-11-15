from django.core.cache import cache
from django.test import TestCase

from app.ai_recommendations.bl.errors import LibraryStoreNotFoundError
from app.ai_recommendations.bl.services.pool_generation_service import (
    PoolGenerationStatus,
)
from app.ai_recommendations.bl.types.pool import RecommendationPool
from app.ai_recommendations.tests.factories import create_movie, create_user, pool_store
from app.ai_recommendations.tests.fakes import NOW
from app.ai_recommendations.tests.service_kit import generation_service


class PoolGenerationServiceInitTests(TestCase):
    """Test loading the data a pool generation needs."""

    def setUp(self):
        """Create a user with a rated movie and a saved pool."""
        cache.clear()
        self.user = create_user()
        create_movie(self.user, "1", "Great Movie", score=9)
        self.store = pool_store()
        self.current = RecommendationPool(generated_at=NOW)
        self.store.save_pool(self.user.id, "movie", self.current)

    def test_loads(self):
        """Load the taste, the current pool and the time, without planning."""
        service = generation_service(self.user.id, self.store)

        self.assertEqual(service.status, PoolGenerationStatus.LOADED)
        self.assertEqual(
            [title.title for title in service.taste.rated],
            ["Great Movie"],
        )
        self.assertEqual(service.current, self.current)
        self.assertEqual(service.now, NOW)
        self.assertIsNone(service.pool)

    def test_missing_user(self):
        """Fail with the store's error for a deleted user."""
        with self.assertRaises(LibraryStoreNotFoundError):
            generation_service(self.user.id + 1, self.store)
