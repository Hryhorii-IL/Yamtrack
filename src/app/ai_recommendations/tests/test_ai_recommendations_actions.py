from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase

from app.ai_recommendations.bl.ai_recommendations_actions import (
    AIRecommendationsActions,
)
from app.ai_recommendations.bl.errors import (
    LLMClientError,
    LLMClientTransientError,
    PoolExtensionServiceTransientError,
    PoolGenerationServiceTransientError,
)
from app.ai_recommendations.bl.types.constants import POOL_SIZE
from app.ai_recommendations.bl.types.pool import (
    MediaRecommendations,
    QueueKind,
    RecommendationPool,
)
from app.ai_recommendations.bl.types.result import ErrorCode
from app.ai_recommendations.tests.fakes import (
    NOW,
    FakeLibrary,
    FakeLLMClient,
    FakeSearchClient,
    FixedClock,
    InMemoryLibraryStore,
    InMemoryPoolStore,
    resolved,
)

ACTIONS = "app.ai_recommendations.bl.ai_recommendations_actions"
USER_ID = 1
FIRST = resolved("1", "First")
SECOND = resolved("2", "Second")
TRANSIENT = LLMClientTransientError("Fake", 503, "busy")


class ActionsTestCase(SimpleTestCase):
    """Run the facade against in-memory stores and scripted clients."""

    def setUp(self):
        """Create a user tracking a movie."""
        self.pool_store = InMemoryPoolStore()
        self.library_store = InMemoryLibraryStore(
            {USER_ID: FakeLibrary(titles={"movie": ["Tracked"]})},
        )
        self.llm_client = FakeLLMClient(["First", "Second"])

    def collaborators(self, **kwargs):
        """Return the fakes the facade runs with."""
        return {
            "llm_client": self.llm_client,
            "search_client": FakeSearchClient([FIRST, SECOND]),
            "pool_store": self.pool_store,
            "library_store": self.library_store,
            **kwargs,
        }

    def save_pool(self, count=0, **kwargs):
        """Save a movie pool with ``count`` items."""
        pool = RecommendationPool(
            generated_at=NOW,
            items=tuple(resolved(str(index)) for index in range(10, 10 + count)),
            **kwargs,
        )
        self.pool_store.save_pool(USER_ID, "movie", pool)
        return pool


class PoolActionsTests(ActionsTestCase):
    """Test serving the pool to pages."""

    def pool(self, *, force=False):
        """Return the pool result, recording queued tasks."""
        self.queue_generation = MagicMock()
        self.queue_extension = MagicMock()
        return AIRecommendationsActions.pool(
            USER_ID,
            "movie",
            force=force,
            queue_generation=self.queue_generation,
            queue_extension=self.queue_extension,
            pool_store=self.pool_store,
        )

    def test_missing_pool_queues_generation(self):
        """Queue a generation after commit and report nothing to show yet."""
        result = self.pool()

        self.assertTrue(result.ok)
        self.assertIsNone(result.value)
        self.queue_generation.assert_called_once_with(USER_ID, "movie")

    def test_small_pool_queues_extension(self):
        """Serve a small pool and queue its next batch."""
        pool = self.save_pool(2)

        result = self.pool()

        self.assertEqual(result.value, pool)
        self.queue_generation.assert_not_called()
        self.queue_extension.assert_called_once_with(USER_ID, "movie")

    def test_full_pool_not_extended(self):
        """Don't queue a batch for a full pool."""
        self.save_pool(POOL_SIZE)

        self.pool()

        self.queue_extension.assert_not_called()

    def test_force_drops_pool(self):
        """Drop the pool and queue a new generation."""
        self.save_pool(2)

        result = self.pool(force=True)

        self.assertIsNone(result.value)
        self.queue_generation.assert_called_once()


class GeneratePoolActionsTests(ActionsTestCase):
    """Test generating pools from jobs."""

    def generate(self, **kwargs):
        """Generate the movie pool with the fakes."""
        return AIRecommendationsActions.generate_pool(
            USER_ID,
            "movie",
            clock=FixedClock(NOW),
            **self.collaborators(),
            **kwargs,
        )

    def test_generate(self):
        """Save and return the generated pool."""
        result = self.generate()

        self.assertEqual(result.value.items, (FIRST, SECOND))
        self.assertEqual(self.pool_store.load_pool(USER_ID, "movie"), result.value)
        self.assertEqual(self.pool_store.locks, set())

    def test_transient_error_raised(self):
        """Re-raise transient errors so the job retries."""
        self.llm_client.error = TRANSIENT

        with self.assertRaises(PoolGenerationServiceTransientError):
            self.generate(last_attempt=False)

        self.assertEqual(self.pool_store.locks, set())

    def test_permanent_error_returned(self):
        """Log permanent errors and return their code."""
        self.llm_client.titles = []
        self.llm_client.error = LLMClientError("Fake", 400, "bad")

        with self.assertLogs(ACTIONS, level="ERROR"):
            result = self.generate()

        self.assertEqual(result.error_code, ErrorCode.INVALID)

    def test_lock_held(self):
        """Report a conflict without calling the LLM while another job runs."""
        self.pool_store.locks.add(("pool", USER_ID, "movie"))

        result = self.generate()

        self.assertEqual(result.error_code, ErrorCode.CONFLICT)
        self.assertEqual(self.llm_client.calls, [])

    def test_missing_user(self):
        """Report users deleted after the job was queued."""
        with self.assertLogs(ACTIONS, level="ERROR"):
            result = AIRecommendationsActions.generate_pool(
                USER_ID + 1,
                "movie",
                clock=FixedClock(NOW),
                **self.collaborators(),
            )

        self.assertEqual(result.error_code, ErrorCode.NOT_FOUND)

    def test_unexpected_error_raised(self):
        """Let errors of other families reach the caller."""
        self.llm_client.titles = []
        self.llm_client.error = KeyError("bug")

        with self.assertRaises(KeyError):
            self.generate()


class ExtendPoolActionsTests(ActionsTestCase):
    """Test extending pools from jobs."""

    def extend(self):
        """Extend the movie pool with the fakes."""
        return AIRecommendationsActions.extend_pool(
            USER_ID,
            "movie",
            **self.collaborators(),
        )

    def test_extend(self):
        """Add a batch to the pool."""
        self.save_pool(1)

        result = self.extend()

        self.assertTrue(result.value)
        self.assertEqual(len(self.pool_store.pools[USER_ID, "movie"].items), 3)

    def test_transient_error_raised(self):
        """Re-raise transient errors so the job retries."""
        self.save_pool(1)
        self.llm_client.error = TRANSIENT

        with self.assertRaises(PoolExtensionServiceTransientError):
            self.extend()

    def test_permanent_error_returned(self):
        """Log permanent errors and return their code."""
        self.save_pool(1)
        self.llm_client.error = LLMClientError("Fake", 400, "bad")

        with self.assertLogs(ACTIONS, level="ERROR"):
            self.assertEqual(self.extend().error_code, ErrorCode.INVALID)


class MediaRecommendationsActionsTests(ActionsTestCase):
    """Test recommendations for a single media."""

    def recommendations(self, *, force=False):
        """Return the recommendations similar to movie 100."""
        return AIRecommendationsActions.media_recommendations(
            USER_ID,
            "movie",
            "tmdb",
            "100",
            force=force,
            clock=FixedClock(NOW),
            **self.collaborators(),
        )

    def setUp(self):
        """Stub the provider metadata of the viewed media."""
        super().setUp()
        self.enterContext(
            patch(
                "app.providers.services.get_media_metadata",
                return_value={"title": "Tracked"},
            ),
        )

    def test_generated_then_cached(self):
        """Generate once and serve the cache afterwards."""
        first = self.recommendations()
        second = self.recommendations()

        self.assertEqual(first.value.items, (FIRST, SECOND))
        self.assertEqual(second.value, first.value)
        self.assertEqual(len(self.llm_client.calls), 1)

    def test_force_regenerates(self):
        """Regenerate cached recommendations on demand."""
        self.recommendations()
        self.recommendations(force=True)

        self.assertEqual(len(self.llm_client.calls), 2)

    def test_transient_error_returned(self):
        """Return UNAVAILABLE in the request, where nothing would retry."""
        self.llm_client.titles = []
        self.llm_client.error = TRANSIENT

        with self.assertLogs(ACTIONS, level="WARNING"):
            result = self.recommendations()

        self.assertEqual(result.error_code, ErrorCode.UNAVAILABLE)

    def test_lock_held(self):
        """Serve old recommendations, or a conflict, while another request runs."""
        self.pool_store.locks.add(("media", USER_ID, "movie", "tmdb", "100"))

        self.assertEqual(self.recommendations().error_code, ErrorCode.CONFLICT)

        old = MediaRecommendations(generated_at=NOW)
        self.pool_store.save_media(USER_ID, "movie", "tmdb", "100", old)
        self.assertEqual(self.recommendations().value, old)


class QueueAndInvalidationActionsTests(ActionsTestCase):
    """Test the bookkeeping writes."""

    def test_queue_flags(self):
        """Mark a task once until it's cleared."""
        mark = AIRecommendationsActions.mark_queued
        self.assertTrue(
            mark(QueueKind.GENERATION, 1, "movie", pool_store=self.pool_store)
        )
        self.assertFalse(
            mark(QueueKind.GENERATION, 1, "movie", pool_store=self.pool_store)
        )

        AIRecommendationsActions.clear_queued(
            QueueKind.GENERATION,
            1,
            "movie",
            pool_store=self.pool_store,
        )

        self.assertTrue(
            mark(QueueKind.GENERATION, 1, "movie", pool_store=self.pool_store)
        )

    def test_invalidate(self):
        """Drop a media type's pool, or all of the user's."""
        self.save_pool(1)
        self.pool_store.save_pool(USER_ID, "anime", RecommendationPool(NOW))

        AIRecommendationsActions.invalidate_media_type(
            USER_ID,
            "movie",
            pool_store=self.pool_store,
        )
        self.assertIsNone(self.pool_store.load_pool(USER_ID, "movie"))
        self.assertIsNotNone(self.pool_store.load_pool(USER_ID, "anime"))

        AIRecommendationsActions.invalidate_user(USER_ID, pool_store=self.pool_store)
        self.assertIsNone(self.pool_store.load_pool(USER_ID, "anime"))
