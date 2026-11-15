from django.core.cache import cache
from django.test import TestCase

from app.ai_recommendations.bl.errors import (
    LLMClientTransientError,
    MediaRecommendationsServiceTransientError,
    MediaRecommendationsServiceWrongStatusForOperationError,
)
from app.ai_recommendations.bl.services.media_recommendations_service import (
    MediaRecommendationsStatus,
)
from app.ai_recommendations.tests.factories import create_user, pool_store
from app.ai_recommendations.tests.fakes import (
    NOW,
    FakeLLMClient,
    FakeSearchClient,
    resolved,
)
from app.ai_recommendations.tests.service_kit import media_service

VIEWED = resolved("1", "Viewed")
OTHER = resolved("2", "Other")


class MediaRecommendationsServiceCommitTests(TestCase):
    """Test generating and saving recommendations for a single media."""

    def setUp(self):
        """Create a user and a store."""
        cache.clear()
        self.user = create_user()
        self.store = pool_store()

    def planned(self, llm_client):
        """Return a planned service whose LLM also recommends the viewed media."""
        service = media_service(
            self.user.id,
            self.store,
            llm_client=llm_client,
            search_client=FakeSearchClient([VIEWED, OTHER]),
        )
        service.plan()
        return service

    def test_commit(self):
        """Save the recommendations without the viewed media itself."""
        service = self.planned(FakeLLMClient(["Viewed", "Other"]))

        recommendations = service.commit()

        self.assertEqual(recommendations.items, (OTHER,))
        self.assertEqual(recommendations.generated_at, NOW)
        self.assertEqual(
            self.store.load_media(self.user.id, "movie", "tmdb", "1"),
            recommendations,
        )
        self.assertEqual(service.status, MediaRecommendationsStatus.COMMITTED)

    def test_transient_error(self):
        """Translate transient LLM errors into the service's family."""
        error = LLMClientTransientError("Fake", 503, "busy")
        service = self.planned(FakeLLMClient(error=error))

        with self.assertRaises(MediaRecommendationsServiceTransientError):
            service.commit()

        self.assertEqual(service.status, MediaRecommendationsStatus.FAILED)
        self.assertIsNone(self.store.load_media(self.user.id, "movie", "tmdb", "1"))

    def test_commit_before_plan(self):
        """Refuse to commit an unplanned service."""
        service = media_service(self.user.id, self.store)

        with self.assertRaises(MediaRecommendationsServiceWrongStatusForOperationError):
            service.commit()
