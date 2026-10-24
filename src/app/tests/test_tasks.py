from datetime import timedelta
from unittest.mock import call, patch

from django.apps import apps
from django.conf import settings
from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.utils import timezone

from app import ai_recommendations, config
from app.models import Item, MediaTypes, Status, UserMessage, UserMessageLevel
from app.providers import services
from app.tasks import cleanup_user_messages, refresh_ai_recommendations


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
    @patch("app.ai_recommendations.global_recommendations")
    def test_refresh_only_enabled_users(self, mock_recommendations, mock_sleep):
        """Force regeneration for each enabled, tracked media type of opted-in users."""
        self.enabled_user.anime_enabled = True
        # Enabled but nothing tracked, so no Gemini request
        self.enabled_user.book_enabled = True
        self.enabled_user.save()

        refreshed_count = refresh_ai_recommendations()

        self.assertEqual(refreshed_count, 2)
        mock_recommendations.assert_has_calls(
            [
                call(self.enabled_user, MediaTypes.MOVIE.value, force=True),
                call(self.enabled_user, MediaTypes.ANIME.value, force=True),
            ],
        )
        self.assertEqual(mock_recommendations.call_count, 2)
        mock_sleep.assert_called_once()

    @patch("app.tasks.time.sleep")
    @patch("app.ai_recommendations.global_recommendations")
    def test_refresh_continues_after_error(self, mock_recommendations, _):
        """Keep refreshing other media types when one fails."""
        self.enabled_user.anime_enabled = True
        self.enabled_user.save()
        mock_recommendations.side_effect = [
            services.ProviderAPIError("gemini", Exception("boom")),
            {"items": []},
        ]

        self.assertEqual(refresh_ai_recommendations(), 1)

    @override_settings(GEMINI_API_KEY="")
    @patch("app.ai_recommendations.global_recommendations")
    def test_refresh_without_api_key(self, mock_recommendations):
        """Skip the refresh when no Gemini API key is configured."""
        self.assertEqual(refresh_ai_recommendations(), 0)
        mock_recommendations.assert_not_called()

    @patch("app.tasks.time.sleep")
    @patch("app.ai_recommendations.global_recommendations")
    def test_refresh_continues_after_unexpected_error(self, mock_recommendations, _):
        """Keep refreshing when an unexpected exception is raised."""
        self.enabled_user.anime_enabled = True
        self.enabled_user.save()
        mock_recommendations.side_effect = [KeyError("results"), {"items": []}]

        self.assertEqual(refresh_ai_recommendations(), 1)
        self.assertEqual(mock_recommendations.call_count, 2)

    @patch("app.tasks.time.sleep")
    @patch("app.ai_recommendations.global_recommendations")
    def test_refresh_skips_in_progress(self, mock_recommendations, _):
        """Don't count media types another request is generating."""
        mock_recommendations.side_effect = (
            ai_recommendations.RecommendationsInProgressError
        )

        self.assertEqual(refresh_ai_recommendations(), 0)
