from django.core.cache import cache
from django.test import TestCase

from app.ai_recommendations.bl.errors import (
    PoolExtensionServiceWrongStatusForOperationError,
)
from app.ai_recommendations.bl.services.pool_extension_service import (
    PoolExtensionStatus,
)
from app.ai_recommendations.bl.types.constants import MAX_EXTENSIONS
from app.ai_recommendations.bl.types.pool import RecommendationPool
from app.ai_recommendations.tests.factories import create_user, pool_store
from app.ai_recommendations.tests.fakes import NOW, resolved
from app.ai_recommendations.tests.service_kit import extension_service


class PoolExtensionServicePlanTests(TestCase):
    """Test building the prompt of the next batch."""

    def setUp(self):
        """Create a user and a store."""
        cache.clear()
        self.user = create_user()
        self.store = pool_store()

    def planned(self, pool):
        """Return a planned service extending ``pool``."""
        self.store.save_pool(self.user.id, "movie", pool)
        service = extension_service(self.user.id, self.store)
        service.plan()
        return service

    def test_excludes_pool_titles(self):
        """Ask for titles that aren't in the pool yet."""
        service = self.planned(
            RecommendationPool(generated_at=NOW, items=(resolved("1", "Dune"),)),
        )

        self.assertEqual(service.status, PoolExtensionStatus.PLANNED)
        self.assertIn("Already recommended (do not repeat):\n- Dune", service.prompt)

    def test_finished_pool(self):
        """Plan nothing for a pool that got its last batch."""
        service = self.planned(
            RecommendationPool(generated_at=NOW, extensions=MAX_EXTENSIONS),
        )

        self.assertIsNone(service.prompt)

    def test_plan_twice(self):
        """Refuse to plan twice."""
        service = self.planned(RecommendationPool(generated_at=NOW))

        with self.assertRaises(PoolExtensionServiceWrongStatusForOperationError):
            service.plan()
