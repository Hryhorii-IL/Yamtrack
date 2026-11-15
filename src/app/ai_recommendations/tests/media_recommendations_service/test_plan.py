from django.core.cache import cache
from django.test import TestCase

from app.ai_recommendations.bl.errors import (
    MediaRecommendationsServiceWrongStatusForOperationError,
)
from app.ai_recommendations.bl.services.media_recommendations_service import (
    MediaRecommendationsStatus,
)
from app.ai_recommendations.tests.factories import create_user, pool_store
from app.ai_recommendations.tests.service_kit import media_service


class MediaRecommendationsServicePlanTests(TestCase):
    """Test building the prompt for a single media."""

    def setUp(self):
        """Create a service for a viewed movie."""
        cache.clear()
        self.service = media_service(create_user().id, pool_store())

    def test_plan(self):
        """Describe the viewed media in the prompt."""
        self.service.plan()

        self.assertEqual(self.service.status, MediaRecommendationsStatus.PLANNED)
        self.assertTrue(
            self.service.prompt.startswith(
                "The user is viewing this movie:\nTitle: Viewed",
            ),
        )

    def test_plan_twice(self):
        """Refuse to plan twice."""
        self.service.plan()

        with self.assertRaises(MediaRecommendationsServiceWrongStatusForOperationError):
            self.service.plan()
