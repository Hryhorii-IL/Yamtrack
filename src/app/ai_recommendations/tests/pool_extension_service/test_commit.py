from django.core.cache import cache
from django.test import TestCase

from app.ai_recommendations.bl.errors import (
    LLMClientError,
    LLMClientTransientError,
    PoolExtensionServiceError,
    PoolExtensionServiceTransientError,
    PoolExtensionServiceWrongStatusForOperationError,
)
from app.ai_recommendations.bl.services.pool_extension_service import (
    PoolExtensionStatus,
)
from app.ai_recommendations.bl.types.pool import RecommendationPool
from app.ai_recommendations.tests.factories import create_user, pool_store
from app.ai_recommendations.tests.fakes import (
    NOW,
    FakeLLMClient,
    FakeSearchClient,
    resolved,
)
from app.ai_recommendations.tests.service_kit import extension_service

DUNE = resolved("1", "Dune")
ARRIVAL = resolved("2", "Arrival")


class PoolExtensionServiceCommitTests(TestCase):
    """Test adding a batch to the saved pool."""

    def setUp(self):
        """Save a pool with one title."""
        cache.clear()
        self.user = create_user()
        self.store = pool_store()
        self.pool = RecommendationPool(generated_at=NOW, items=(DUNE,))
        self.store.save_pool(self.user.id, "movie", self.pool)

    def planned(self, llm_client):
        """Return a planned service."""
        service = extension_service(
            self.user.id,
            self.store,
            llm_client=llm_client,
            search_client=FakeSearchClient([DUNE, ARRIVAL]),
        )
        service.plan()
        return service

    def test_adds_new_titles(self):
        """Append unseen titles and count the extension."""
        service = self.planned(FakeLLMClient(["Dune", "Arrival"]))

        self.assertTrue(service.commit())

        pool = self.store.load_pool(self.user.id, "movie")
        self.assertEqual(pool.items, (DUNE, ARRIVAL))
        self.assertEqual(pool.extensions, 1)
        self.assertEqual(service.status, PoolExtensionStatus.COMMITTED)

    def test_transient_error(self):
        """Ask for a retry and leave the pool unchanged."""
        error = LLMClientTransientError("Fake", 503, "busy")
        service = self.planned(FakeLLMClient(["Arrival"], error))

        with self.assertRaises(PoolExtensionServiceTransientError):
            service.commit()

        self.assertEqual(self.store.load_pool(self.user.id, "movie"), self.pool)
        self.assertEqual(service.status, PoolExtensionStatus.FAILED)

    def test_permanent_error(self):
        """Fail without asking for a retry."""
        service = self.planned(FakeLLMClient(error=LLMClientError("Fake", 400, "bad")))

        with self.assertRaises(PoolExtensionServiceError) as context:
            service.commit()

        self.assertNotIsInstance(context.exception, PoolExtensionServiceTransientError)

    def test_invalidated_while_generating(self):
        """Don't restore a pool invalidated meanwhile."""
        store = self.store
        user_id = self.user.id

        class InvalidatingLLMClient(FakeLLMClient):
            def recommend(self, system_instruction, prompt):
                store.invalidate_media_type(user_id, "movie")
                yield from super().recommend(system_instruction, prompt)

        self.assertFalse(self.planned(InvalidatingLLMClient(["Arrival"])).commit())
        self.assertIsNone(self.store.load_pool(self.user.id, "movie"))

    def test_without_pool(self):
        """Do nothing when the pool is gone."""
        self.store.delete_pool(self.user.id, "movie")
        llm_client = FakeLLMClient(["Arrival"])

        self.assertFalse(self.planned(llm_client).commit())
        self.assertEqual(llm_client.calls, [])

    def test_commit_before_plan(self):
        """Refuse to commit an unplanned service."""
        service = extension_service(self.user.id, self.store)

        with self.assertRaises(PoolExtensionServiceWrongStatusForOperationError):
            service.commit()
