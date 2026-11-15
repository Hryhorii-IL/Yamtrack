from typing import TYPE_CHECKING

from django.core.cache import cache
from django.test import SimpleTestCase, TestCase

from app.ai_recommendations.bl.errors import PoolStoreLockError
from app.ai_recommendations.bl.types.pool import (
    MediaRecommendations,
    QueueKind,
    RecommendationPool,
)
from app.ai_recommendations.tests.factories import pool_store
from app.ai_recommendations.tests.fakes import NOW, InMemoryPoolStore, resolved

POOL = RecommendationPool(generated_at=NOW, items=(resolved("1"),))


# The contract is a mixin; type checkers still need to see the assertions
ContractBase = SimpleTestCase if TYPE_CHECKING else object


class PoolStoreContract(ContractBase):
    """Behaviour every PoolStore implementation must have."""

    def make_store(self):
        """Return an empty store."""
        raise NotImplementedError

    def setUp(self):
        """Create an empty store."""
        self.store = self.make_store()

    def test_save_and_load_pool(self):
        """Load what was saved, until it's deleted."""
        self.store.save_pool(1, "movie", POOL)

        self.assertEqual(self.store.load_pool(1, "movie"), POOL)
        self.assertIsNone(self.store.load_pool(1, "anime"))

        self.store.delete_pool(1, "movie")

        self.assertIsNone(self.store.load_pool(1, "movie"))

    def test_partial_and_failed_pools_load(self):
        """Load partial and failed pools like complete ones."""
        self.store.save_partial_pool(1, "movie", POOL)
        self.assertEqual(self.store.load_pool(1, "movie"), POOL)

        self.store.save_failed_pool(1, "anime", POOL)
        self.assertEqual(self.store.load_pool(1, "anime"), POOL)

    def test_is_current(self):
        """Compare pools by version, None meaning no pool."""
        self.assertTrue(self.store.is_current(1, "movie", None))
        self.store.save_pool(1, "movie", POOL)

        self.assertTrue(self.store.is_current(1, "movie", POOL))
        self.assertFalse(self.store.is_current(1, "movie", None))

    def test_media(self):
        """Save recommendations per media."""
        recommendations = MediaRecommendations(generated_at=NOW)
        self.store.save_media(1, "movie", "tmdb", "5", recommendations)

        self.assertEqual(
            self.store.load_media(1, "movie", "tmdb", "5"),
            recommendations,
        )
        self.assertIsNone(self.store.load_media(1, "movie", "tmdb", "6"))

    def test_lock_until_transaction_ends(self):
        """Refuse a held lock and release it when the transaction ends."""
        with self.store.transaction():
            self.store.lock_pool(1, "movie")
            with self.assertRaises(PoolStoreLockError):
                self.store.lock_pool(1, "movie")
            # Other pools have their own lock
            self.store.lock_pool(1, "anime")

        with self.store.transaction():
            self.store.lock_pool(1, "movie")

    def test_lock_outside_transaction(self):
        """Refuse locks that nothing would release."""
        with self.assertRaises(RuntimeError):
            self.store.lock_pool(1, "movie")

    def test_on_commit(self):
        """Run callbacks after a successful transaction only."""
        calls = []
        with self.store.transaction():
            self.store.on_commit(lambda: calls.append("ok"))
            self.assertEqual(calls, [])
        self.assertEqual(calls, ["ok"])

        with self.assertRaises(ValueError), self.store.transaction():
            self.store.on_commit(lambda: calls.append("rolled back"))
            raise ValueError
        self.assertEqual(calls, ["ok"])

    def test_queued(self):
        """Mark a task once until it's cleared."""
        self.assertFalse(self.store.is_background_queued(1, "movie"))
        self.assertTrue(self.store.mark_queued(QueueKind.EXTENSION, 1, "movie"))
        self.assertFalse(self.store.mark_queued(QueueKind.EXTENSION, 1, "movie"))
        self.assertTrue(self.store.is_background_queued(1, "movie"))

        self.store.clear_queued(QueueKind.EXTENSION, 1, "movie")

        self.assertFalse(self.store.is_background_queued(1, "movie"))
        self.store.keep_queued(QueueKind.GENERATION, 1, "movie")
        self.assertTrue(self.store.is_background_queued(1, "movie"))

    def test_invalidate_media_type(self):
        """Drop one media type's pool and media recommendations."""
        self.store.save_pool(1, "movie", POOL)
        self.store.save_pool(1, "anime", POOL)
        self.store.save_media(1, "movie", "tmdb", "5", MediaRecommendations(NOW))

        self.store.invalidate_media_type(1, "movie")

        self.assertIsNone(self.store.load_pool(1, "movie"))
        self.assertIsNone(self.store.load_media(1, "movie", "tmdb", "5"))
        self.assertIsNotNone(self.store.load_pool(1, "anime"))

    def test_invalidate_user(self):
        """Drop every recommendation of one user only."""
        self.store.save_pool(1, "movie", POOL)
        self.store.save_pool(12, "movie", POOL)
        self.store.save_media(1, "movie", "tmdb", "5", MediaRecommendations(NOW))

        self.store.invalidate_user(1)

        self.assertIsNone(self.store.load_pool(1, "movie"))
        self.assertIsNone(self.store.load_media(1, "movie", "tmdb", "5"))
        self.assertIsNotNone(self.store.load_pool(12, "movie"))


class InMemoryPoolStoreTests(PoolStoreContract, SimpleTestCase):
    """Run the contract against the fake."""

    def make_store(self):
        """Return an in-memory store."""
        return InMemoryPoolStore()


class DjangoCachePoolStoreTests(PoolStoreContract, TestCase):
    """Run the contract against the Django cache."""

    def make_store(self):
        """Return a store over an empty cache."""
        cache.clear()
        return pool_store()
