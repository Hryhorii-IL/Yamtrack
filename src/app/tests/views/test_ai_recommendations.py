from unittest.mock import patch

from django.apps import apps
from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from app import ai_recommendations, config
from app.models import Item, MediaTypes, Sources, Status
from app.providers import services

HTMX = {"HX-Request": "true"}


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


def cached_data(*media_ids):
    """Return cached recommendations for TMDB movies."""
    return {
        "generated_at": timezone.now(),
        "items": [
            {
                "media_id": media_id,
                "source": Sources.TMDB.value,
                "media_type": MediaTypes.MOVIE.value,
                "title": f"Movie {media_id}",
                "image": settings.IMG_NONE,
                "reason": f"Reason {media_id}",
            }
            for media_id in media_ids
        ],
    }


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
        """Return 404 when no Gemini API key is configured."""
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

    @patch("app.ai_recommendations.global_recommendations")
    def test_section_get(self, mock_recommendations):
        """Render cached recommendations without forcing regeneration."""
        mock_recommendations.return_value = cached_data("1", "2")

        response = self.client.get(self.section_url, headers=HTMX)

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(
            response,
            "app/components/ai_recommendations_section.html",
        )
        self.assertEqual(len(response.context["results"]), 2)
        self.assertContains(response, "Reason 1")
        mock_recommendations.assert_called_once_with(
            self.user,
            MediaTypes.MOVIE.value,
            force=False,
        )

    @patch("app.ai_recommendations.global_recommendations")
    def test_section_post_regenerates(self, mock_recommendations):
        """Force regeneration on POST."""
        mock_recommendations.return_value = cached_data("1")

        self.client.post(self.section_url, headers=HTMX)

        mock_recommendations.assert_called_once_with(
            self.user,
            MediaTypes.MOVIE.value,
            force=True,
        )

    @patch("app.ai_recommendations.global_recommendations")
    def test_section_error(self, mock_recommendations):
        """Show provider errors in the section."""
        mock_recommendations.side_effect = services.ProviderAPIError(
            "gemini",
            Exception("boom"),
        )

        response = self.client.get(self.section_url, headers=HTMX)

        self.assertContains(response, "error contacting the Gemini API")

    @patch("app.ai_recommendations.global_recommendations")
    def test_section_in_progress(self, mock_recommendations):
        """Poll again while another request is generating."""
        mock_recommendations.side_effect = (
            ai_recommendations.RecommendationsInProgressError
        )

        response = self.client.get(self.section_url, headers=HTMX)

        self.assertContains(response, 'hx-trigger="load delay:5s"')

    def test_section_disabled_media_type(self):
        """Return 404 for media types the user disabled."""
        self.user.movie_enabled = False
        self.user.save()

        response = self.client.get(self.section_url, headers=HTMX)

        self.assertEqual(response.status_code, 404)

    @patch("app.ai_recommendations.media_recommendations")
    def test_media_section(self, mock_recommendations):
        """Render the recommendations for a single media."""
        mock_recommendations.return_value = cached_data("2")

        response = self.client.get(self.media_url, headers=HTMX)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.context["results"]), 1)
        mock_recommendations.assert_called_once_with(
            self.user,
            MediaTypes.MOVIE.value,
            Sources.TMDB.value,
            "1",
            force=False,
        )

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

        with patch("app.ai_recommendations.media_recommendations") as mock_recs:
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
