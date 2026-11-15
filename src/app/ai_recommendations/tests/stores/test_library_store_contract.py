from typing import TYPE_CHECKING

from django.test import SimpleTestCase, TestCase

from app.ai_recommendations.bl.errors import LibraryStoreNotFoundError
from app.ai_recommendations.bl.stores.django_library_store import DjangoLibraryStore
from app.ai_recommendations.bl.types.taste import RatedTitle
from app.ai_recommendations.tests.factories import create_movie, create_user
from app.ai_recommendations.tests.fakes import (
    NOW,
    FakeLibrary,
    InMemoryLibraryStore,
)

MISSING_USER_ID = 999_999


# The contract is a mixin; type checkers still need to see the assertions
ContractBase = SimpleTestCase if TYPE_CHECKING else object


class LibraryStoreContract(ContractBase):
    """Behaviour every LibraryStore implementation must have.

    Each implementation sets up a user who tracks "Great" (rated 9) and
    "Unrated" movies, newest first.
    """

    def make_store(self):
        """Return the store and the id of the set up user."""
        raise NotImplementedError

    def setUp(self):
        """Set up the user's library."""
        self.store, self.user_id = self.make_store()

    def test_enabled_media_types(self):
        """Offer movies, which are enabled by default."""
        self.assertIn("movie", self.store.load_enabled_media_types(self.user_id))

    def test_has_media(self):
        """Tell media types the user tracks from the others."""
        self.assertTrue(self.store.has_media(self.user_id, "movie"))
        self.assertFalse(self.store.has_media(self.user_id, "anime"))

    def test_load_taste(self):
        """Return the rated titles and the library, newest first."""
        taste = self.store.load_taste(self.user_id, "movie")

        self.assertEqual([title.title for title in taste.rated], ["Great"])
        self.assertEqual(taste.rated[0].score, 9)
        self.assertEqual(taste.library, ("Unrated", "Great"))

    def test_load_user_score(self):
        """Return the score of a rated media only."""
        self.assertEqual(
            self.store.load_user_score(self.user_id, "movie", "tmdb", "1"),
            9,
        )
        self.assertIsNone(
            self.store.load_user_score(self.user_id, "movie", "tmdb", "2")
        )

    def test_missing_user(self):
        """Raise the store's own error for unknown users."""
        with self.assertRaises(LibraryStoreNotFoundError):
            self.store.load_taste(MISSING_USER_ID, "movie")


class InMemoryLibraryStoreTests(LibraryStoreContract, SimpleTestCase):
    """Run the contract against the fake."""

    def make_store(self):
        """Return a fake library of user 1."""
        great = RatedTitle(
            item_id=1,
            title="Great",
            media_type="movie",
            score=9,
            created_at=NOW,
        )
        library = FakeLibrary(
            titles={"movie": ["Unrated", "Great"]},
            rated=[great],
            scores={("movie", "tmdb", "1"): 9},
        )
        return InMemoryLibraryStore({1: library}), 1


class DjangoLibraryStoreTests(LibraryStoreContract, TestCase):
    """Run the contract against the database."""

    def make_store(self):
        """Create the user and their movies."""
        user = create_user()
        create_movie(user, "1", "Great", score=9)
        create_movie(user, "2", "Unrated")
        return DjangoLibraryStore(), user.id
