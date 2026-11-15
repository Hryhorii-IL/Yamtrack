from datetime import timedelta
from unittest.mock import call, patch

from django.apps import apps
from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import TestCase, override_settings
from django.utils import timezone

from app import ai_recommendations, config
from app.models import Item, MediaTypes, Status, UserMessage, UserMessageLevel
from app.providers import services
from app.tasks import (
    cleanup_user_messages,
    extend_ai_recommendations,
    generate_ai_recommendations,
    refresh_ai_recommendations,
)


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
        self.enabled_user = get_user_model().objects.create_user(
            username="enabled",
            ai_recommendations_enabled=True,
        )
        for field in ("tv", "anime", "manga", "game", "book", "comic", "boardgame"):
            setattr(self.enabled_user, f"{field}_enabled", False)
        self.enabled_user.save()
        track(self.enabled_user, MediaTypes.MOVIE.value)
        track(self.enabled_user, MediaTypes.ANIME.value)
        get_user_model().objects.create_user(username="disabled")

    @patch("app.tasks.time.sleep")
    @patch("app.ai_recommendations.generate_global_recommendations")
    def test_refresh_only_enabled_users(self, mock_recommendations, mock_sleep):
        """Force regeneration for each enabled, tracked media type of opted-in users."""
        mock_recommendations.return_value = None
        self.enabled_user.anime_enabled = True
        # Enabled but nothing tracked, so no LLM request
        self.enabled_user.book_enabled = True
        self.enabled_user.save()

        refreshed_count = refresh_ai_recommendations()

        self.assertEqual(refreshed_count, 2)
        self.assertEqual(
            mock_recommendations.call_args_list,
            [
                call(self.enabled_user, MediaTypes.MOVIE.value, progressive=False),
                call(self.enabled_user, MediaTypes.ANIME.value, progressive=False),
            ],
        )
        mock_sleep.assert_called_once()

    @patch("app.tasks.time.sleep")
    @patch("app.ai_recommendations.generate_global_recommendations")
    def test_refresh_continues_after_error(self, mock_recommendations, _):
        """Keep refreshing other media types when one fails."""
        self.enabled_user.anime_enabled = True
        self.enabled_user.save()
        mock_recommendations.side_effect = [
            services.ProviderAPIError("gemini", Exception("boom")),
            {"items": []},
        ]

        self.assertEqual(refresh_ai_recommendations(), 1)

    @patch("app.tasks.queue_ai_extension")
    @patch("app.tasks.time.sleep")
    @patch("app.ai_recommendations.generate_global_recommendations")
    def test_refresh_queues_extension(self, mock_generate, _, mock_queue):
        """Extend refreshed pools that are below their size."""
        mock_generate.return_value = {"items": [], "extensions": 0}

        refresh_ai_recommendations()

        mock_queue.assert_called_once_with(
            self.enabled_user.id,
            MediaTypes.MOVIE.value,
        )

    @override_settings(GEMINI_API_KEY="")
    @patch("app.ai_recommendations.generate_global_recommendations")
    def test_refresh_without_api_key(self, mock_recommendations):
        """Skip the refresh when no AI provider API key is configured."""
        self.assertEqual(refresh_ai_recommendations(), 0)
        mock_recommendations.assert_not_called()

    @patch("app.tasks.time.sleep")
    @patch("app.ai_recommendations.generate_global_recommendations")
    def test_refresh_continues_after_unexpected_error(self, mock_recommendations, _):
        """Keep refreshing when an unexpected exception is raised."""
        self.enabled_user.anime_enabled = True
        self.enabled_user.save()
        mock_recommendations.side_effect = [KeyError("results"), {"items": []}]

        self.assertEqual(refresh_ai_recommendations(), 1)
        self.assertEqual(mock_recommendations.call_count, 2)

    @patch("app.tasks.time.sleep")
    @patch("app.ai_recommendations.generate_global_recommendations")
    def test_refresh_skips_in_progress(self, mock_recommendations, _):
        """Don't count media types another request is generating."""
        mock_recommendations.side_effect = (
            ai_recommendations.RecommendationsInProgressError
        )

        self.assertEqual(refresh_ai_recommendations(), 0)


class ExtendAIRecommendationsTaskTests(TestCase):
    """Test the background extension of AI recommendation pools."""

    def setUp(self):
        """Create a user with a queued extension."""
        cache.clear()
        self.user = get_user_model().objects.create_user(username="test")
        self.queued_key = ai_recommendations.extension_queued_key(
            self.user.id,
            MediaTypes.MOVIE.value,
        )
        cache.set(self.queued_key, 1)

    @patch("app.ai_recommendations.extend_global_recommendations")
    def test_extend(self, mock_extend):
        """Extend a full pool and allow queueing the next extension."""
        mock_extend.return_value = True

        result = extend_ai_recommendations(self.user.id, MediaTypes.MOVIE.value)

        self.assertTrue(result)
        mock_extend.assert_called_once_with(self.user, MediaTypes.MOVIE.value)
        self.assertIsNone(cache.get(self.queued_key))

    @patch("app.ai_recommendations.extend_global_recommendations")
    def test_extend_chains_until_full(self, mock_extend):
        """Queue the next batch while the pool is below its size."""
        key = ai_recommendations.global_cache_key(self.user.id, MediaTypes.MOVIE.value)
        cache.set(key, {"items": [], "extensions": 0})

        def extend(*_args):
            data = cache.get(key)
            cache.set(key, {**data, "extensions": data["extensions"] + 1})
            return True

        mock_extend.side_effect = extend

        extend_ai_recommendations(self.user.id, MediaTypes.MOVIE.value)

        # Eager Celery runs the chained batches inline
        self.assertEqual(mock_extend.call_count, ai_recommendations.MAX_EXTENSIONS)
        self.assertIsNone(cache.get(self.queued_key))

    @patch("app.ai_recommendations.extend_global_recommendations")
    def test_extend_failure(self, mock_extend):
        """Log failures and still clear the queued flag."""
        mock_extend.side_effect = services.ProviderAPIError(
            "gemini",
            Exception("boom"),
        )

        result = extend_ai_recommendations(self.user.id, MediaTypes.MOVIE.value)

        self.assertFalse(result)
        self.assertIsNone(cache.get(self.queued_key))

    def test_extend_missing_user(self):
        """Ignore users deleted after the extension was queued."""
        self.assertFalse(extend_ai_recommendations(0, MediaTypes.MOVIE.value))


class GenerateAIRecommendationsTaskTests(TestCase):
    """Test the background generation of AI recommendation pools."""

    def setUp(self):
        """Create a user with a queued generation."""
        cache.clear()
        self.user = get_user_model().objects.create_user(username="test")
        self.queued_key = ai_recommendations.generation_queued_key(
            self.user.id,
            MediaTypes.MOVIE.value,
        )
        cache.set(self.queued_key, 1)

    @patch("app.tasks.queue_ai_extension")
    @patch("app.ai_recommendations.generate_global_recommendations")
    def test_generate_queues_extension(self, mock_generate, mock_queue):
        """Generate the pool, queue its extension and clear the flag."""
        mock_generate.return_value = {"items": [], "extensions": 0}

        self.assertTrue(
            generate_ai_recommendations(self.user.id, MediaTypes.MOVIE.value),
        )

        mock_generate.assert_called_once_with(self.user, MediaTypes.MOVIE.value)
        mock_queue.assert_called_once_with(self.user.id, MediaTypes.MOVIE.value)
        self.assertIsNone(cache.get(self.queued_key))

    @patch("app.tasks.queue_ai_extension")
    @patch("app.ai_recommendations.generate_global_recommendations")
    def test_generate_failure(self, mock_generate, mock_queue):
        """Log failures, skip the extension and clear the flag."""
        mock_generate.side_effect = services.ProviderAPIError(
            "gemini",
            Exception("boom"),
        )

        self.assertFalse(
            generate_ai_recommendations(self.user.id, MediaTypes.MOVIE.value),
        )

        mock_queue.assert_not_called()
        self.assertIsNone(cache.get(self.queued_key))

    @patch("app.ai_recommendations.generate_global_recommendations")
    def test_generate_in_progress(self, mock_generate):
        """Skip when another generation holds the lock."""
        mock_generate.side_effect = ai_recommendations.RecommendationsInProgressError

        self.assertFalse(
            generate_ai_recommendations(self.user.id, MediaTypes.MOVIE.value),
        )
        self.assertIsNone(cache.get(self.queued_key))
