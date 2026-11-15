from django.test import TestCase

from app.ai_recommendations.bl.queries.media_types import (
    is_media_type_available,
)
from app.ai_recommendations.bl.stores.django_library_store import DjangoLibraryStore
from app.ai_recommendations.tests.factories import create_movie, create_user


class IsMediaTypeAvailableTests(TestCase):
    """Test is_media_type_available."""

    def setUp(self):
        """Create a user tracking a movie, with games disabled."""
        self.user = create_user(game_enabled=False)
        create_movie(self.user, "1", "Tracked")
        self.store = DjangoLibraryStore()

    def test_is_media_type_available(self):
        """Require the media type to be enabled and tracked."""
        self.assertTrue(
            is_media_type_available(self.user.id, "movie", store=self.store)
        )
        self.assertFalse(
            is_media_type_available(self.user.id, "anime", store=self.store)
        )
        self.assertFalse(
            is_media_type_available(self.user.id, "game", store=self.store)
        )
