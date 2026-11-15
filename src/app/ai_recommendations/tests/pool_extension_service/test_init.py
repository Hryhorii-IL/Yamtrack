from django.core.cache import cache
from django.test import TestCase

from app.ai_recommendations.bl.services.pool_extension_service import (
    PoolExtensionStatus,
)
from app.ai_recommendations.bl.types.pool import RecommendationPool
from app.ai_recommendations.tests.factories import create_user, pool_store
from app.ai_recommendations.tests.fakes import NOW
from app.ai_recommendations.tests.service_kit import extension_service


class PoolExtensionServiceInitTests(TestCase):
    """Test loading the pool to extend."""

    def test_loads_pool(self):
        """Load the saved pool and the user's taste."""
        cache.clear()
        user = create_user()
        store = pool_store()
        pool = RecommendationPool(generated_at=NOW)
        store.save_pool(user.id, "movie", pool)

        service = extension_service(user.id, store)

        self.assertEqual(service.status, PoolExtensionStatus.LOADED)
        self.assertEqual(service.pool, pool)
        self.assertEqual(service.taste.rated, ())
