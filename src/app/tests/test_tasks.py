from datetime import timedelta
from unittest.mock import call, patch

from django.apps import apps
from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import TestCase, override_settings
from django.utils import timezone

from app import config
from app.ai_recommendations.bl.errors import (
    PoolExtensionServiceTransientError,
    PoolGenerationServiceTransientError,
)
from app.ai_recommendations.bl.stores.django_cache_pool_store import (
    DjangoCachePoolStore,
)
from app.ai_recommendations.bl.types.pool import QueueKind, RecommendationPool
from app.ai_recommendations.bl.types.result import ErrorCode, Result
from app.models import Item, MediaTypes, Status, UserMessage, UserMessageLevel
from app.tasks import (
    AI_MAX_RETRIES,
    AI_RECOMMENDATIONS_DELAY,
    cleanup_user_messages,
    extend_ai_recommendations,
    generate_ai_recommendations,
    queue_ai_extension,
    queue_ai_generation,
    refresh_ai_recommendations,
)

ACTIONS = "app.tasks.AIRecommendationsActions"
POOL = RecommendationPool(generated_at=timezone.now())


class CleanupUserMessagesTaskTests(TestCase):
    """Test cleanup of old shown user messages."""

    def setUp(self):
        """Create a user for task tests."""
        self.user = get_user_model().objects.create_user(
            username="test",
        )

    @override_settings(USER_MESSAGE_RETENTION_DAYS=30)
    def test_cleanup_user_messages_deletes_only_old_shown_messages(self):
        """Delete only shown messages older than the retention window."""
        now = timezone.now()
        old_shown = UserMessage.objects.create(
            user=self.user,
            level=UserMessageLevel.INFO,
            message="old shown",
            shown_at=now - timedelta(days=31),
        )
        recent_shown = UserMessage.objects.create(
            user=self.user,
            level=UserMessageLevel.INFO,
            message="recent shown",
            shown_at=now - timedelta(days=5),
        )
        unseen = UserMessage.objects.create(
            user=self.user,
            level=UserMessageLevel.INFO,
            message="unseen",
        )

        deleted_count = cleanup_user_messages()

        self.assertEqual(deleted_count, 1)
        self.assertFalse(UserMessage.objects.filter(id=old_shown.id).exists())
        self.assertTrue(UserMessage.objects.filter(id=recent_shown.id).exists())
        self.assertTrue(UserMessage.objects.filter(id=unseen.id).exists())


def track(user, media_type):
    """Add a planned entry of a media type for the user."""
    item = Item.objects.create(
        media_id="100",
        source=config.get_default_source_name(media_type).value,
        media_type=media_type,
        title=f"Tracked {media_type}",
        image=settings.IMG_NONE,
    )
    apps.get_model(app_label="app", model_name=media_type).objects.create(
        item=item,
        user=user,
        status=Status.PLANNING.value,
    )


class RefreshAIRecommendationsTaskTests(TestCase):
    """Test the nightly AI recommendations refresh."""

    def setUp(self):
        """Create an opted-in and an opted-out user."""
        cache.clear()
        self.enabled_user = get_user_model().objects.create_user(
            username="enabled",
            ai_recommendations_enabled=True,
        )
        for field in ("tv", "anime", "manga", "game", "book", "comic", "boardgame"):
            setattr(self.enabled_user, f"{field}_enabled", False)
        self.enabled_user.anime_enabled = True
        # Enabled but nothing tracked, so nothing to refresh
        self.enabled_user.book_enabled = True
        self.enabled_user.save()
        track(self.enabled_user, MediaTypes.MOVIE.value)
        track(self.enabled_user, MediaTypes.ANIME.value)
        get_user_model().objects.create_user(username="disabled")

    @patch("app.tasks.generate_ai_recommendations.apply_async")
    def test_refresh_queues_spread_generations(self, mock_apply):
        """Queue a refresh per tracked media type of opted-in users, spread out."""
        self.assertEqual(refresh_ai_recommendations(), 2)

        self.assertEqual(
            mock_apply.call_args_list,
            [
                call(
                    (self.enabled_user.id, MediaTypes.MOVIE.value),
                    {"refresh": True},
                    countdown=0,
                ),
                call(
                    (self.enabled_user.id, MediaTypes.ANIME.value),
                    {"refresh": True},
                    countdown=AI_RECOMMENDATIONS_DELAY,
                ),
            ],
        )

    @override_settings(GEMINI_API_KEY="")
    @patch("app.tasks.generate_ai_recommendations.apply_async")
    def test_refresh_without_api_key(self, mock_apply):
        """Skip the refresh when no AI provider API key is configured."""
        self.assertEqual(refresh_ai_recommendations(), 0)
        mock_apply.assert_not_called()


class QueueAITasksTests(TestCase):
    """Test queueing pool tasks once."""

    def setUp(self):
        """Clear queued flags."""
        cache.clear()

    @patch("app.tasks.generate_ai_recommendations.apply_async")
    def test_generation_queued_once(self, mock_apply):
        """Queue a generation once until it runs."""
        queue_ai_generation(1, MediaTypes.MOVIE.value)
        queue_ai_generation(1, MediaTypes.MOVIE.value)

        mock_apply.assert_called_once_with(
            (1, MediaTypes.MOVIE.value),
            {"refresh": False},
            countdown=0,
        )

    @patch("app.tasks.extend_ai_recommendations.delay")
    def test_extension_queued_once(self, mock_delay):
        """Queue an extension once until it runs."""
        queue_ai_extension(1, MediaTypes.MOVIE.value)
        queue_ai_extension(1, MediaTypes.MOVIE.value)

        mock_delay.assert_called_once_with(1, MediaTypes.MOVIE.value)


class AITaskTestCase(TestCase):
    """Base test case for the pool tasks."""

    kind = QueueKind.GENERATION

    def setUp(self):
        """Mark the task as queued, like queue_ai_* does."""
        cache.clear()
        self.store = DjangoCachePoolStore(lock_timeout=60)
        self.store.mark_queued(self.kind, 1, MediaTypes.MOVIE.value)

    def is_queued(self):
        """Return whether the task is still marked as queued."""
        return self.store.is_background_queued(1, MediaTypes.MOVIE.value)


class GenerateAIRecommendationsTaskTests(AITaskTestCase):
    """Test the background generation of AI recommendation pools."""

    @patch("app.tasks.queue_ai_extension")
    @patch("app.tasks.pool_needs_extension", return_value=True)
    @patch(f"{ACTIONS}.generate_pool")
    def test_generate_queues_extension(self, mock_generate, _, mock_queue):
        """Generate the pool progressively, queue its extension, clear the flag."""
        mock_generate.return_value = Result(value=POOL)

        self.assertTrue(generate_ai_recommendations(1, MediaTypes.MOVIE.value))

        mock_generate.assert_called_once_with(
            1,
            MediaTypes.MOVIE.value,
            progressive=True,
            refresh=False,
            last_attempt=False,
        )
        mock_queue.assert_called_once_with(1, MediaTypes.MOVIE.value)
        self.assertFalse(self.is_queued())

    @patch(f"{ACTIONS}.generate_pool", return_value=Result(value=POOL))
    def test_refresh(self, mock_generate):
        """Keep the current pool until a refreshed one is complete."""
        generate_ai_recommendations(1, MediaTypes.MOVIE.value, refresh=True)

        kwargs = mock_generate.call_args.kwargs
        self.assertEqual((kwargs["progressive"], kwargs["refresh"]), (False, True))

    @patch("app.tasks.queue_ai_extension")
    @patch(f"{ACTIONS}.generate_pool")
    def test_generate_failure(self, mock_generate, mock_queue):
        """Skip the extension and clear the flag when nothing was generated."""
        mock_generate.return_value = Result(error_code=ErrorCode.INVALID)

        self.assertFalse(generate_ai_recommendations(1, MediaTypes.MOVIE.value))

        mock_queue.assert_not_called()
        self.assertFalse(self.is_queued())

    @patch(f"{ACTIONS}.generate_pool")
    def test_transient_error_retried(self, mock_generate):
        """Retry transient failures, telling the last attempt it's the last."""
        mock_generate.side_effect = PoolGenerationServiceTransientError("busy")

        result = generate_ai_recommendations.apply((1, MediaTypes.MOVIE.value))

        self.assertIsInstance(result.result, PoolGenerationServiceTransientError)
        self.assertEqual(
            [call.kwargs["last_attempt"] for call in mock_generate.call_args_list],
            [False, False, False, True],
        )

    @patch(f"{ACTIONS}.generate_pool")
    def test_transient_error_keeps_flag(self, mock_generate):
        """Keep pages polling while the generation will be retried."""
        mock_generate.side_effect = [
            PoolGenerationServiceTransientError("busy"),
            Result(value=POOL),
        ]

        generate_ai_recommendations.apply((1, MediaTypes.MOVIE.value))

        self.assertEqual(mock_generate.call_count, 2)
        self.assertFalse(self.is_queued())


class ExtendAIRecommendationsTaskTests(AITaskTestCase):
    """Test the background extension of AI recommendation pools."""

    kind = QueueKind.EXTENSION

    @patch("app.tasks.pool_needs_extension", return_value=False)
    @patch(f"{ACTIONS}.extend_pool", return_value=Result(value=True))
    def test_extend_full_pool(self, mock_extend, _):
        """Extend the pool and allow queueing the next extension."""
        self.assertTrue(extend_ai_recommendations(1, MediaTypes.MOVIE.value))

        mock_extend.assert_called_once_with(1, MediaTypes.MOVIE.value)
        self.assertFalse(self.is_queued())

    @patch("app.tasks.pool_needs_extension", side_effect=[True, True, False])
    @patch(f"{ACTIONS}.extend_pool", return_value=Result(value=True))
    def test_extend_chains_until_full(self, mock_extend, _):
        """Queue the next batch while the pool needs more."""
        extend_ai_recommendations(1, MediaTypes.MOVIE.value)

        # Eager Celery runs the chained batches inline
        self.assertEqual(mock_extend.call_count, 3)
        self.assertFalse(self.is_queued())

    @patch(f"{ACTIONS}.extend_pool", return_value=Result(error_code=ErrorCode.INVALID))
    def test_extend_failure(self, _):
        """Clear the queued flag when no batch was added."""
        self.assertFalse(extend_ai_recommendations(1, MediaTypes.MOVIE.value))
        self.assertFalse(self.is_queued())

    @patch(f"{ACTIONS}.extend_pool")
    def test_transient_error_retried(self, mock_extend):
        """Retry transient failures, then give up and clear the flag."""
        mock_extend.side_effect = PoolExtensionServiceTransientError("busy")

        extend_ai_recommendations.apply((1, MediaTypes.MOVIE.value))

        self.assertEqual(mock_extend.call_count, AI_MAX_RETRIES + 1)
        self.assertFalse(self.is_queued())
