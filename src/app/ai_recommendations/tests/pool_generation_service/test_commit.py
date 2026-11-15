from django.core.cache import cache
from django.test import TestCase

from app.ai_recommendations.bl.errors import (
    LLMClientError,
    LLMClientTransientError,
    PoolGenerationServiceError,
    PoolGenerationServiceNoResultsError,
    PoolGenerationServiceTransientError,
    PoolGenerationServiceWrongStatusForOperationError,
)
from app.ai_recommendations.bl.services.pool_generation_service import (
    PoolGenerationStatus,
)
from app.ai_recommendations.bl.types.pool import RecommendationPool
from app.ai_recommendations.tests.factories import create_user, pool_store
from app.ai_recommendations.tests.fakes import (
    NOW,
    FakeLLMClient,
    FakeSearchClient,
    resolved,
)
from app.ai_recommendations.tests.service_kit import generation_service

FIRST = resolved("1", "First")
SECOND = resolved("2", "Second")
CATALOG = [FIRST, SECOND]
TRANSIENT = LLMClientTransientError("Fake", 503, "busy")


class RecordingSearchClient(FakeSearchClient):
    """Records how many items the saved pool has when each title resolves."""

    def __init__(self, store, user_id):
        """Keep the store to inspect while resolving."""
        super().__init__(CATALOG)
        self.store = store
        self.user_id = user_id
        self.saved = []

    def resolve(self, recommendations, media_type):
        """Record the saved pool before yielding each title."""
        for title in super().resolve(recommendations, media_type):
            pool = self.store.load_pool(self.user_id, media_type)
            self.saved.append(None if pool is None else len(pool.items))
            yield title


class PoolGenerationServiceCommitTests(TestCase):
    """Test generating and saving the pool."""

    def setUp(self):
        """Create a user and a store."""
        cache.clear()
        self.user = create_user()
        self.store = pool_store()

    def commit(self, titles=(), error=None, **kwargs):
        """Plan and commit a service whose LLM recommends ``titles``."""
        service = generation_service(
            self.user.id,
            self.store,
            llm_client=FakeLLMClient(titles, error),
            search_client=kwargs.pop("search_client", FakeSearchClient(CATALOG)),
            **kwargs,
        )
        service.plan()
        self.service = service
        return service.commit()

    def saved(self):
        """Return the saved movie pool."""
        return self.store.load_pool(self.user.id, "movie")

    def test_progressive_pool_saved_item_by_item(self):
        """Save a partial pool while titles resolve, then the complete pool."""
        search_client = RecordingSearchClient(self.store, self.user.id)

        pool = self.commit(["First", "Second"], search_client=search_client)

        self.assertEqual(search_client.saved, [0, 1])
        self.assertEqual(pool.items, (FIRST, SECOND))
        self.assertFalse(pool.partial)
        self.assertEqual(self.saved(), pool)
        self.assertEqual(self.service.status, PoolGenerationStatus.COMMITTED)

    def test_transient_error_keeps_partial_pool(self):
        """Leave the partial pool for the retried job and ask for a retry."""
        with self.assertRaises(PoolGenerationServiceTransientError):
            self.commit(["First"], TRANSIENT, last_attempt=False)

        self.assertEqual(self.saved().items, (FIRST,))
        self.assertTrue(self.saved().partial)
        self.assertEqual(self.service.status, PoolGenerationStatus.FAILED)

    def test_retry_continues_partial_pool(self):
        """Add only the new titles to the pool a failed attempt left."""
        with self.assertRaises(PoolGenerationServiceTransientError):
            self.commit(["First"], TRANSIENT, last_attempt=False)
        partial = self.saved()

        pool = self.commit(["First", "Second"])

        self.assertEqual(pool.items, (FIRST, SECOND))
        self.assertEqual(pool.version, partial.version)

    def test_transient_error_on_last_attempt(self):
        """Show the error once the job won't retry anymore."""
        with self.assertRaises(PoolGenerationServiceError) as context:
            self.commit(["First"], TRANSIENT, last_attempt=True)

        self.assertNotIsInstance(
            context.exception,
            PoolGenerationServiceTransientError,
        )
        self.assertEqual(self.saved().error, str(TRANSIENT))
        self.assertFalse(self.saved().partial)

    def test_permanent_llm_error(self):
        """Show permanent LLM errors without asking for a retry."""
        error = LLMClientError("Fake", 400, "bad request")

        with self.assertRaises(PoolGenerationServiceError):
            self.commit(error=error, last_attempt=False)

        self.assertEqual(self.saved().error, str(error))

    def test_no_results(self):
        """Fail with a readable error when no title is found."""
        with self.assertRaises(PoolGenerationServiceNoResultsError):
            self.commit(["Unknown"])

        self.assertIn("None of the recommended titles", self.saved().error)

    def test_invalidated_while_generating(self):
        """Stop and don't restore a pool invalidated meanwhile."""
        store = self.store
        user_id = self.user.id

        class InvalidatingLLMClient(FakeLLMClient):
            def recommend(self, system_instruction, prompt, *, thinking=False):
                store.invalidate_user(user_id)
                yield from super().recommend(
                    system_instruction,
                    prompt,
                    thinking=thinking,
                )

        service = generation_service(
            user_id,
            store,
            llm_client=InvalidatingLLMClient(["First"]),
            search_client=FakeSearchClient(CATALOG),
        )
        service.plan()

        self.assertIsNone(service.commit())
        self.assertIsNone(self.saved())

    def test_already_generated(self):
        """Return the current pool without calling the LLM."""
        current = RecommendationPool(generated_at=NOW, items=(FIRST,))
        self.store.save_pool(self.user.id, "movie", current)

        self.assertEqual(self.commit(["Second"]), current)

    def test_refresh_keeps_pool_on_failure(self):
        """Keep the current pool when a background refresh fails."""
        current = RecommendationPool(generated_at=NOW, items=(FIRST,))
        self.store.save_pool(self.user.id, "movie", current)

        with self.assertRaises(PoolGenerationServiceError):
            self.commit(
                error=LLMClientError("Fake", 400, "bad"),
                refresh=True,
                progressive=False,
            )

        self.assertEqual(self.saved(), current)

    def test_refresh_replaces_pool(self):
        """Swap in the refreshed pool once it's complete."""
        self.store.save_pool(
            self.user.id,
            "movie",
            RecommendationPool(generated_at=NOW.replace(year=2025), items=(FIRST,)),
        )

        pool = self.commit(["Second"], refresh=True, progressive=False)

        self.assertEqual(pool.items, (SECOND,))
        self.assertEqual(self.saved(), pool)

    def test_thinking_passed_to_llm(self):
        """Ask the LLM to think when the service was told to."""
        llm_client = FakeLLMClient(["First"])
        service = generation_service(
            self.user.id,
            self.store,
            thinking=True,
            llm_client=llm_client,
            search_client=FakeSearchClient(CATALOG),
        )
        service.plan()

        service.commit()

        self.assertEqual(llm_client.thinking, [True])

    def test_commit_before_plan(self):
        """Refuse to commit an unplanned service."""
        service = generation_service(self.user.id, self.store)

        with self.assertRaises(PoolGenerationServiceWrongStatusForOperationError):
            service.commit()
        self.assertEqual(service.status, PoolGenerationStatus.LOADED)
