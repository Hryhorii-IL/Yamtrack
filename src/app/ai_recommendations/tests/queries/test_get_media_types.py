from django.test import TestCase

from app.ai_recommendations.bl.queries.media_types import (
    get_media_types,
)
from app.ai_recommendations.bl.stores.django_library_store import DjangoLibraryStore
from app.ai_recommendations.tests.factories import create_movie, create_user


class GetMediaTypesTests(TestCase):
    """Test get_media_types."""

    def setUp(self):
        """Create a user tracking a movie, with games disabled."""
        self.user = create_user(game_enabled=False)
        create_movie(self.user, "1", "Tracked")
        self.store = DjangoLibraryStore()

    def test_get_media_types(self):
        """Offer the enabled media types the user tracks anything of."""
        self.assertEqual(get_media_types(self.user.id, store=self.store), ("movie",))
