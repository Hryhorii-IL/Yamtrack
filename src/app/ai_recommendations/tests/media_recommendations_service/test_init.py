from django.core.cache import cache
from django.test import TestCase

from app.ai_recommendations.bl.errors import MediaRecommendationsServiceNotFoundError
from app.ai_recommendations.bl.services.media_recommendations_service import (
    MediaRecommendationsStatus,
)
from app.ai_recommendations.tests.factories import create_movie, create_user, pool_store
from app.ai_recommendations.tests.service_kit import media_service
from app.models import Sources
from app.providers import services

METADATA = {"title": "Viewed", "genres": ["Drama"], "synopsis": "A story."}


class MediaRecommendationsServiceInitTests(TestCase):
    """Test loading the viewed media."""

    def setUp(self):
        """Create a user who rated the viewed movie."""
        cache.clear()
        self.user = create_user()
        create_movie(self.user, "1", "Viewed", score=7)

    def test_loads_details(self):
        """Describe the media with its metadata and the user's score."""
        service = media_service(
            self.user.id,
            pool_store(),
            fetch_metadata=lambda *_args: METADATA,
        )

        self.assertEqual(service.status, MediaRecommendationsStatus.LOADED)
        self.assertEqual(service.details.title, "Viewed")
        self.assertEqual(service.details.genres, ("Drama",))
        self.assertEqual(service.details.user_score, 7)

    def test_metadata_error(self):
        """Raise the service's own error when the provider fails."""

        def fetch_metadata(*_args):
            raise services.ProviderAPIError(Sources.TMDB.value, Exception())

        with self.assertRaises(MediaRecommendationsServiceNotFoundError):
            media_service(self.user.id, pool_store(), fetch_metadata=fetch_metadata)
