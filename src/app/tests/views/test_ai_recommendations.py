from unittest.mock import patch

from django.apps import apps
from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from app import config
from app.ai_recommendations.bl.stores.django_cache_pool_store import (
    DjangoCachePoolStore,
)
from app.ai_recommendations.bl.types.constants import (
    MAX_EXTENSIONS,
    MEDIA_COUNT,
    POOL_SHOWN,
    POOL_SIZE,
)
from app.ai_recommendations.bl.types.pool import (
    MediaRecommendations,
    QueueKind,
    RecommendationPool,
)
from app.ai_recommendations.bl.types.recommendation import ResolvedTitle
from app.ai_recommendations.bl.types.result import ErrorCode, Result
from app.models import Item, MediaTypes, Sources, Status
from app.providers import services

HTMX = {"HX-Request": "true"}
ACTIONS = "app.views.AIRecommendationsActions"


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


def pool_of(*media_ids, **kwargs):
    """Return a pool of TMDB movies."""
    return RecommendationPool(
        generated_at=timezone.now(),
        items=movies(media_ids),
        **kwargs,
    )


def media_result(*media_ids):
    """Return ready recommendations similar to a media."""
    return Result(
        value=MediaRecommendations(
            generated_at=timezone.now(),
            items=movies(media_ids),
        ),
    )


def movies(media_ids):
    """Return resolved TMDB movies."""
    return tuple(
        ResolvedTitle(
            media_id=media_id,
            source=Sources.TMDB.value,
            media_type=MediaTypes.MOVIE.value,
            title=f"Movie {media_id}",
            image=settings.IMG_NONE,
        )
        for media_id in media_ids
    )


class AIRecommendationsViewTests(TestCase):
    """Test the AI recommendations page and sections."""

    def setUp(self):
        """Create a user with AI recommendations enabled and log in."""
        cache.clear()
        self.credentials = {"username": "test", "password": "12345"}
        self.user = get_user_model().objects.create_user(**self.credentials)
        self.user.ai_recommendations_enabled = True
        self.user.ai_media_recommendations_enabled = True
        self.user.save()
        for media_type in ("tv", "movie", "anime", "game", "boardgame"):
            track(self.user, media_type)
        self.client.login(**self.credentials)
        self.section_url = reverse(
            "ai_recommendations_section",
            args=[MediaTypes.MOVIE.value],
        )
        self.media_url = reverse(
            "media_ai_recommendations",
            args=[Sources.TMDB.value, MediaTypes.MOVIE.value, "1"],
        )

    def test_page_disabled(self):
        """Return 404 when the page toggle is off."""
        self.user.ai_recommendations_enabled = False
        self.user.save()

        response = self.client.get(reverse("ai_recommendations"))

        self.assertEqual(response.status_code, 404)

    @override_settings(GEMINI_API_KEY="")
    def test_page_without_api_key(self):
        """Return 404 when the AI provider has no API key."""
        response = self.client.get(reverse("ai_recommendations"))

        self.assertEqual(response.status_code, 404)

    def test_page_tabs(self):
        """Render a tab per enabled media type and select the requested one."""
        self.user.game_enabled = False
        self.user.save()

        response = self.client.get(
            reverse("ai_recommendations"),
            {"type": MediaTypes.ANIME.value},
        )

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "app/ai_recommendations.html")
        self.assertEqual(response.context["media_type"], MediaTypes.ANIME.value)
        self.assertIn(MediaTypes.BOARDGAME.value, response.context["media_types"])
        self.assertNotIn(MediaTypes.GAME.value, response.context["media_types"])
        self.assertNotIn(MediaTypes.SEASON.value, response.context["media_types"])
        # Enabled but nothing tracked
        self.assertNotIn(MediaTypes.MANGA.value, response.context["media_types"])
        self.assertContains(
            response,
            reverse("ai_recommendations_section", args=[MediaTypes.ANIME.value]),
        )

    def test_page_invalid_type_falls_back(self):
        """Select the first media type when the requested one is invalid."""
        response = self.client.get(reverse("ai_recommendations"), {"type": "nope"})

        self.assertEqual(response.context["media_type"], MediaTypes.TV.value)

    def test_sidebar_link(self):
        """Show the sidebar link only when the page is enabled."""
        response = self.client.get(reverse("ai_recommendations"))
        self.assertContains(response, "AI Recommendations")

        self.user.ai_recommendations_enabled = False
        self.user.save()
        response = self.client.get(reverse("statistics"))
        self.assertNotContains(response, reverse("ai_recommendations"))

    def cache_pool(self, pool):
        """Cache the user's movie pool and return it."""
        DjangoCachePoolStore(lock_timeout=60).save_pool(
            self.user.id,
            MediaTypes.MOVIE.value,
            pool,
        )
        return pool

    def mark_extension_queued(self):
        """Mark a background pool extension as queued."""
        DjangoCachePoolStore(lock_timeout=60).mark_queued(
            QueueKind.EXTENSION,
            self.user.id,
            MediaTypes.MOVIE.value,
        )

    @patch("app.tasks.generate_ai_recommendations.apply_async")
    @patch("app.tasks.extend_ai_recommendations.delay")
    def test_section_get(self, _, mock_generate):
        """Render the cached pool without generating it again."""
        self.cache_pool(pool_of("1", "2"))

        response = self.client.get(self.section_url, headers=HTMX)

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(
            response,
            "app/components/ai_recommendations_section.html",
        )
        self.assertEqual(len(response.context["results"]), 2)
        self.assertContains(response, "Movie 1")
        mock_generate.assert_not_called()

    @patch("app.tasks.extend_ai_recommendations.delay")
    def test_section_shows_whole_pool(self, _):
        """Show every pool item and poll for more while the pool grows."""
        self.cache_pool(pool_of(*[str(media_id) for media_id in range(1, 21)]))

        response = self.client.get(self.section_url, headers=HTMX)

        self.assertEqual(len(response.context["results"]), 20)
        self.assertTrue(response.context["growing"])
        self.assertContains(
            response,
            f"{self.section_url}?offset=20&shown=20&generated=",
        )

    def test_section_finished_pool_stops_polling(self):
        """Don't poll when no batch will be added anymore."""
        self.cache_pool(pool_of("1", "2", extensions=MAX_EXTENSIONS))

        response = self.client.get(self.section_url, headers=HTMX)

        self.assertFalse(response.context["growing"])
        self.assertNotContains(response, "offset=")

    def test_more_returns_new_items(self):
        """Return only the items added after the offset."""
        pool = self.cache_pool(pool_of("1", "2", "3"))
        self.mark_extension_queued()

        response = self.client.get(
            f"{self.section_url}?offset=1&generated={pool.version}",
            headers=HTMX,
        )

        self.assertTemplateUsed(response, "app/components/ai_recommendations_more.html")
        self.assertEqual(
            [result["item"]["media_id"] for result in response.context["results"]],
            ["2", "3"],
        )
        self.assertContains(response, "offset=3&shown=2&generated=")

    def test_section_partial_pool_polls_fast(self):
        """Poll every second while the first batch is generating."""
        self.cache_pool(pool_of("1", partial=True))

        response = self.client.get(self.section_url, headers=HTMX)

        self.assertTrue(response.context["growing"])
        self.assertContains(response, 'hx-trigger="load delay:1s"')

    def test_section_cached_error(self):
        """Show a failed generation without polling."""
        self.cache_pool(pool_of(error="Gemini down"))

        response = self.client.get(self.section_url, headers=HTMX)

        self.assertContains(response, "Gemini down")
        self.assertNotContains(response, "offset=")

    def test_more_shows_error_inline(self):
        """Show a generation that failed after some cards were shown."""
        pool = self.cache_pool(pool_of("1", error="Gemini down"))

        response = self.client.get(
            f"{self.section_url}?offset=1&generated={pool.version}",
            headers=HTMX,
        )

        self.assertContains(response, "Gemini down")
        self.assertNotContains(response, "offset=")

    @patch("app.tasks.extend_ai_recommendations.delay")
    def test_section_shows_at_most_pool_shown(self, _):
        """Cap the cards and stop polling once enough are shown."""
        self.cache_pool(pool_of(*[str(media_id) for media_id in range(1, 36)]))

        response = self.client.get(self.section_url, headers=HTMX)

        self.assertEqual(len(response.context["results"]), POOL_SHOWN)
        self.assertFalse(response.context["growing"])
        self.assertNotContains(response, "offset=")

    def test_more_fills_up_to_pool_shown(self):
        """Append only the cards still missing from the shown limit."""
        pool = self.cache_pool(pool_of(*[str(media_id) for media_id in range(1, 36)]))
        self.mark_extension_queued()

        response = self.client.get(
            f"{self.section_url}?offset=30&shown={POOL_SHOWN - 2}"
            f"&generated={pool.version}",
            headers=HTMX,
        )

        self.assertEqual(
            [result["item"]["media_id"] for result in response.context["results"]],
            ["31", "32"],
        )
        self.assertNotContains(response, "offset=")

    def test_more_stops_after_regeneration(self):
        """Return nothing when the pool changed since the page was rendered."""
        self.cache_pool(pool_of("1", "2"))

        response = self.client.get(
            f"{self.section_url}?offset=1&generated=old",
            headers=HTMX,
        )

        self.assertEqual(response.content, b"")

    def test_more_invalid_offset(self):
        """Return nothing for an invalid offset."""
        response = self.client.get(f"{self.section_url}?offset=x", headers=HTMX)

        self.assertEqual(response.content, b"")

    @patch("app.tasks.extend_ai_recommendations.delay")
    def test_section_queues_extension_once(self, mock_delay):
        """Queue a single background extension while the pool is small."""
        self.cache_pool(pool_of("1", "2"))

        self.client.get(self.section_url, headers=HTMX)
        self.client.get(self.section_url, headers=HTMX)

        mock_delay.assert_called_once_with(self.user.id, MediaTypes.MOVIE.value)

    @patch("app.tasks.extend_ai_recommendations.delay")
    def test_section_full_pool_not_extended(self, mock_delay):
        """Don't extend a pool that reached its size."""
        self.cache_pool(pool_of(*[str(media_id) for media_id in range(POOL_SIZE)]))

        self.client.get(self.section_url, headers=HTMX)

        mock_delay.assert_not_called()

    @patch("app.tasks.generate_ai_recommendations.apply_async")
    def test_section_post_regenerates(self, mock_generate):
        """Drop the pool and queue a new generation on POST."""
        self.cache_pool(pool_of("1"))

        response = self.client.post(self.section_url, headers=HTMX)

        mock_generate.assert_called_once_with(
            (self.user.id, MediaTypes.MOVIE.value),
            {"refresh": False},
            countdown=0,
        )
        self.assertTrue(response.context["in_progress"])

    @patch("app.tasks.generate_ai_recommendations.apply_async")
    def test_section_in_progress(self, mock_generate):
        """Queue the generation of a missing pool and poll for it."""
        response = self.client.get(self.section_url, headers=HTMX)

        mock_generate.assert_called_once_with(
            (self.user.id, MediaTypes.MOVIE.value),
            {"refresh": False},
            countdown=0,
        )
        self.assertContains(response, 'hx-trigger="load delay:2s"')

    def test_section_disabled_media_type(self):
        """Return 404 for media types the user disabled."""
        self.user.movie_enabled = False
        self.user.save()

        response = self.client.get(self.section_url, headers=HTMX)

        self.assertEqual(response.status_code, 404)

    @patch(f"{ACTIONS}.media_recommendations")
    def test_media_section(self, mock_recommendations):
        """Render the recommendations for a single media."""
        mock_recommendations.return_value = media_result("2")

        response = self.client.get(self.media_url, headers=HTMX)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.context["results"]), 1)
        mock_recommendations.assert_called_once_with(
            self.user.id,
            MediaTypes.MOVIE.value,
            Sources.TMDB.value,
            "1",
            force=False,
        )

    @patch(f"{ACTIONS}.media_recommendations")
    def test_media_section_error(self, mock_recommendations):
        """Show the error of a failed generation."""
        mock_recommendations.return_value = Result(
            error_code=ErrorCode.UNAVAILABLE,
        )

        response = self.client.get(self.media_url, headers=HTMX)

        self.assertContains(response, "The Gemini API is unavailable")

    @patch(f"{ACTIONS}.media_recommendations")
    def test_media_section_in_progress(self, mock_recommendations):
        """Poll while another request generates the recommendations."""
        mock_recommendations.return_value = Result(error_code=ErrorCode.CONFLICT)

        response = self.client.get(self.media_url, headers=HTMX)

        self.assertTrue(response.context["in_progress"])

    @patch(f"{ACTIONS}.media_recommendations")
    def test_media_section_capped(self, mock_recommendations):
        """Show at most MEDIA_COUNT single media recommendations."""
        mock_recommendations.return_value = media_result(
            *[str(media_id) for media_id in range(2, 40)],
        )

        response = self.client.get(self.media_url, headers=HTMX)

        self.assertEqual(len(response.context["results"]), MEDIA_COUNT)
        self.assertNotContains(response, "offset=")

    def test_media_section_disabled(self):
        """Return 404 when the media tab toggle is off."""
        self.user.ai_media_recommendations_enabled = False
        self.user.save()

        response = self.client.get(self.media_url, headers=HTMX)

        self.assertEqual(response.status_code, 404)

    @patch("app.providers.services.get_media_metadata")
    def test_media_details_tab(self, mock_metadata):
        """Show the AI tab on media details without generating anything."""
        mock_metadata.return_value = {
            "media_id": "1",
            "title": "Test Movie",
            "media_type": MediaTypes.MOVIE.value,
            "source": Sources.TMDB.value,
            "image": settings.IMG_NONE,
        }
        details_url = reverse(
            "media_details",
            args=[Sources.TMDB.value, MediaTypes.MOVIE.value, "1", "test-movie"],
        )

        with patch(f"{ACTIONS}.media_recommendations") as mock_recs:
            response = self.client.get(details_url)
            mock_recs.assert_not_called()

        self.assertTrue(response.context["show_ai_tab"])
        self.assertContains(response, "tab = 'ai'")
        self.assertContains(response, self.media_url)

        self.user.ai_media_recommendations_enabled = False
        self.user.save()
        response = self.client.get(details_url)
        self.assertFalse(response.context["show_ai_tab"])
        self.assertNotContains(response, "tab = 'ai'")

    def test_section_without_htmx_redirects_to_page(self):
        """Redirect to the full page, e.g. after saving from a track modal."""
        response = self.client.get(self.section_url)

        self.assertRedirects(
            response,
            f"{reverse('ai_recommendations')}?type={MediaTypes.MOVIE.value}",
        )

    @patch("app.providers.services.get_media_metadata")
    def test_media_section_without_htmx_provider_error(self, mock_metadata):
        """Return 404 instead of 500 when the media can't be fetched."""
        mock_metadata.side_effect = services.ProviderAPIError(
            Sources.TMDB.value,
            Exception("boom"),
        )

        response = self.client.get(self.media_url)

        self.assertEqual(response.status_code, 404)

    @patch("app.providers.services.get_media_metadata")
    def test_media_section_without_htmx_redirects_to_details(self, mock_metadata):
        """Redirect to the media details page when not loaded by HTMX."""
        mock_metadata.return_value = {"title": "Test Movie"}

        response = self.client.get(self.media_url)

        self.assertRedirects(
            response,
            reverse(
                "media_details",
                args=[Sources.TMDB.value, MediaTypes.MOVIE.value, "1", "test-movie"],
            ),
            fetch_redirect_response=False,
        )

    def test_section_soft_navigation_redirects_to_page(self):
        """Serve the full page after saving from a track modal."""
        response = self.client.get(
            self.section_url,
            headers={**HTMX, "X-Soft-Navigation": "true"},
        )

        self.assertRedirects(
            response,
            f"{reverse('ai_recommendations')}?type={MediaTypes.MOVIE.value}",
        )
