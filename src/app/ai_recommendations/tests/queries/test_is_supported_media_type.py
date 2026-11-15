from django.test import TestCase

from app.ai_recommendations.bl.queries.media_types import (
    is_supported_media_type,
)
from app.ai_recommendations.bl.stores.django_library_store import DjangoLibraryStore
from app.ai_recommendations.tests.factories import create_movie, create_user


class IsSupportedMediaTypeTests(TestCase):
    """Test is_supported_media_type."""

    def setUp(self):
        """Create a user tracking a movie, with games disabled."""
        self.user = create_user(game_enabled=False)
        create_movie(self.user, "1", "Tracked")
        self.store = DjangoLibraryStore()

    def test_is_supported_media_type(self):
        """Support media types that can be searched, not episodes."""
        self.assertTrue(is_supported_media_type("movie"))
        self.assertFalse(is_supported_media_type("episode"))
