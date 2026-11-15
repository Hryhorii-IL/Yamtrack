from django.core.cache import cache
from django.test import TestCase

from app.ai_recommendations.bl.errors import (
    PoolGenerationServiceWrongStatusForOperationError,
)
from app.ai_recommendations.bl.services.pool_generation_service import (
    PoolGenerationStatus,
)
from app.ai_recommendations.bl.types.pool import RecommendationPool
from app.ai_recommendations.tests.factories import create_movie, create_user, pool_store
from app.ai_recommendations.tests.fakes import NOW, resolved
from app.ai_recommendations.tests.service_kit import generation_service

EARLIER = NOW.replace(year=2025)


class PoolGenerationServicePlanTests(TestCase):
    """Test deciding how to generate the pool and building its prompt."""

    def setUp(self):
        """Create a user with a rated movie."""
        cache.clear()
        self.user = create_user()
        create_movie(self.user, "1", "Great Movie", score=9)
        self.store = pool_store()

    def plan(self, **kwargs):
        """Return a planned service."""
        service = generation_service(self.user.id, self.store, **kwargs)
        service.plan()
        return service

    def save_current(self, **kwargs):
        """Save the pool the generation finds."""
        pool = RecommendationPool(
            generated_at=EARLIER, items=(resolved("7", "Old"),), **kwargs
        )
        self.store.save_pool(self.user.id, "movie", pool)
        return pool

    def test_new_pool(self):
        """Start a partial pool at the clock's time and describe the taste."""
        service = self.plan()

        self.assertEqual(service.status, PoolGenerationStatus.PLANNED)
        self.assertEqual(
            service.pool, RecommendationPool(generated_at=NOW, partial=True)
        )
        self.assertIn("The Movie Database", service.system_instruction)
        self.assertIn("- Great Movie (movie) — 9/10", service.prompt)

    def test_continues_partial_pool(self):
        """Continue the partial pool of a retried job without repeating titles."""
        current = self.save_current(partial=True)

        service = self.plan()

        self.assertEqual(service.pool, current)
        self.assertIn("Already recommended (do not repeat):\n- Old", service.prompt)

    def test_skips_generated_pool(self):
        """Do nothing when the pool was generated already."""
        self.save_current()

        self.assertIsNone(self.plan().pool)

    def test_refresh_replaces_pool(self):
        """Start a new complete pool for background refreshes."""
        self.save_current()

        service = self.plan(refresh=True, progressive=False)

        self.assertEqual(service.pool, RecommendationPool(generated_at=NOW))

    def test_plan_twice(self):
        """Refuse to plan twice and keep the status."""
        service = self.plan()

        with self.assertRaises(PoolGenerationServiceWrongStatusForOperationError):
            service.plan()
        self.assertEqual(service.status, PoolGenerationStatus.PLANNED)
