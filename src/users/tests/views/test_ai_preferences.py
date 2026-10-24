from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.contrib.messages import get_messages
from django.core.cache import cache
from django.db import IntegrityError
from django.test import TestCase, override_settings
from django.urls import reverse

from app import ai_recommendations
from app.models import MediaTypes
from users.models import DateFormatChoices


class AIRecommendationPreferencesTests(TestCase):
    """Tests for the AI recommendation preferences."""

    def setUp(self):
        """Create user for the tests."""
        self.watch_regions_patcher = patch(
            "users.views.tmdb.watch_provider_regions",
            return_value=[("UNSET", "Disabled")],
        )
        self.watch_regions_patcher.start()
        self.addCleanup(self.watch_regions_patcher.stop)
        cache.clear()

        self.credentials = {"username": "test", "password": "12345"}
        self.user = get_user_model().objects.create_user(**self.credentials)
        self.client.login(**self.credentials)
        self.media_types = [
            media_type
            for media_type in MediaTypes.values
            if media_type != MediaTypes.EPISODE.value
            and getattr(self.user, f"{media_type}_enabled")
        ]
        self.cache_key = ai_recommendations.global_cache_key(
            self.user.id,
            MediaTypes.MOVIE.value,
        )

    def post(self, data):
        """Post the preferences form with the current media types."""
        return self.client.post(
            reverse("preferences"),
            {"media_types_checkboxes": self.media_types, **data},
        )

    def test_get_renders_ai_settings(self):
        """Render the AI toggles and sliders."""
        response = self.client.get(reverse("preferences"))

        self.assertTrue(response.context["gemini_configured"])
        self.assertContains(response, 'name="ai_recommendations_enabled"')
        self.assertContains(response, 'name="ai_liked_min_score"')

    @override_settings(GEMINI_API_KEY="")
    def test_get_without_api_key(self):
        """Explain how to enable AI recommendations without an API key."""
        response = self.client.get(reverse("preferences"))

        self.assertContains(response, "GEMINI_API_KEY")

    def test_save_ai_settings(self):
        """Save the AI toggles and slider values."""
        response = self.post(
            {
                "ai_recommendations_enabled": "on",
                "ai_media_recommendations_enabled": "on",
                "ai_include_liked": "on",
                "ai_liked_min_score": "7",
                "ai_disliked_max_score": "3",
            },
        )
        self.assertRedirects(response, reverse("preferences"))

        self.user.refresh_from_db()
        self.assertTrue(self.user.ai_recommendations_enabled)
        self.assertTrue(self.user.ai_media_recommendations_enabled)
        self.assertTrue(self.user.ai_include_liked)
        self.assertFalse(self.user.ai_include_disliked)
        self.assertEqual(self.user.ai_liked_min_score, 7)
        self.assertEqual(self.user.ai_disliked_max_score, 3)

    def test_scores_are_clamped(self):
        """Clamp out of range scores and ignore invalid ones."""
        self.post({"ai_liked_min_score": "42", "ai_disliked_max_score": "abc"})

        self.user.refresh_from_db()
        self.assertEqual(self.user.ai_liked_min_score, 10)
        self.assertEqual(self.user.ai_disliked_max_score, 4)

    def test_prompt_change_invalidates_cache(self):
        """Delete cached recommendations when prompt settings change."""
        cache.set(self.cache_key, {"items": []})

        self.post(
            {
                "ai_include_liked": "on",
                "ai_include_disliked": "on",
                "ai_liked_min_score": "9",
                "ai_disliked_max_score": "4",
            },
        )

        self.assertIsNone(cache.get(self.cache_key))

    def test_unrelated_change_keeps_cache(self):
        """Keep cached recommendations when prompt settings are unchanged."""
        cache.set(self.cache_key, {"items": []})

        self.post(
            {
                "ai_recommendations_enabled": "on",
                "ai_include_liked": "on",
                "ai_include_disliked": "on",
                "ai_liked_min_score": "8",
                "ai_disliked_max_score": "4",
                "progress_bar": "on",
            },
        )

        self.assertEqual(cache.get(self.cache_key), {"items": []})

    def test_liked_score_must_be_above_disliked(self):
        """Reject liked scores that aren't higher than disliked scores."""
        response = self.post(
            {
                "ai_recommendations_enabled": "on",
                "ai_liked_min_score": "5",
                "ai_disliked_max_score": "5",
            },
        )
        self.assertRedirects(response, reverse("preferences"))

        self.user.refresh_from_db()
        self.assertFalse(self.user.ai_recommendations_enabled)
        self.assertEqual(self.user.ai_liked_min_score, 8)
        self.assertEqual(self.user.ai_disliked_max_score, 4)
        messages = list(get_messages(response.wsgi_request))
        self.assertIn("must be higher", str(messages[0]))

    def test_liked_score_constraint(self):
        """Enforce the score order in the database."""
        self.user.ai_liked_min_score = 3
        self.user.ai_disliked_max_score = 6

        with self.assertRaises(IntegrityError):
            self.user.save()

    def test_save_ai_prompts(self):
        """Save non-empty wishes per media type."""
        self.post(
            {
                "ai_prompt_positive_movie": "  European films ",
                "ai_prompt_negative_movie": "No Russian content",
                "ai_prompt_positive_anime": "   ",
                "ai_prompt_negative_season": "ignored",
            },
        )

        self.user.refresh_from_db()
        self.assertEqual(
            self.user.ai_prompts,
            {"movie": {"positive": "European films", "negative": "No Russian content"}},
        )

    def test_too_long_ai_prompt(self):
        """Reject wishes longer than the limit without saving anything."""
        response = self.post(
            {
                "ai_recommendations_enabled": "on",
                "ai_prompt_positive_movie": "x" * 501,
            },
        )

        self.user.refresh_from_db()
        self.assertEqual(self.user.ai_prompts, {})
        self.assertFalse(self.user.ai_recommendations_enabled)
        messages = list(get_messages(response.wsgi_request))
        self.assertIn("at most 500 characters", str(messages[0]))

    def test_prompt_change_invalidates_media_type_only(self):
        """Delete only the cache of the media type whose wishes changed."""
        anime_key = ai_recommendations.global_cache_key(
            self.user.id,
            MediaTypes.ANIME.value,
        )
        cache.set(self.cache_key, {"items": []})
        cache.set(anime_key, {"items": []})

        self.post(
            {
                "ai_include_liked": "on",
                "ai_include_disliked": "on",
                "ai_prompt_negative_movie": "No Russian content",
            },
        )

        self.assertIsNone(cache.get(self.cache_key))
        self.assertEqual(cache.get(anime_key), {"items": []})

    def test_get_renders_ai_prompts(self):
        """Prefill the wishes of each media type."""
        self.user.ai_prompts = {"book": {"negative": "No self-help"}}
        self.user.save()

        response = self.client.get(reverse("preferences"))

        self.assertContains(response, 'name="ai_prompt_negative_book"')
        self.assertContains(response, "No self-help")
        self.assertNotContains(response, 'name="ai_prompt_positive_season"')

    def test_invalid_ai_preferences_keep_other_preferences(self):
        """Save the other preferences when the AI preferences are invalid."""
        self.post(
            {
                "date_format": DateFormatChoices.US,
                "progress_bar": "on",
                "ai_recommendations_enabled": "on",
                "ai_prompt_positive_movie": "x" * 501,
            },
        )

        self.user.refresh_from_db()
        self.assertEqual(self.user.date_format, DateFormatChoices.US)
        self.assertFalse(self.user.ai_recommendations_enabled)
        self.assertEqual(self.user.ai_prompts, {})

    def test_ai_prompt_line_breaks_count_once(self):
        """Count CRLF line breaks as one character like the browser does."""
        text = ("x" * 44 + "\r\n") * 10 + "x" * 45

        self.post({"ai_prompt_positive_movie": text})

        self.user.refresh_from_db()
        self.assertEqual(
            self.user.ai_prompts["movie"]["positive"],
            text.replace("\r\n", "\n"),
        )
