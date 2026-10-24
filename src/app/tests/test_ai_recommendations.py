from unittest.mock import MagicMock, patch

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import RequestFactory, TestCase, override_settings

from app import ai_recommendations
from app.models import (
    Anime,
    Item,
    MediaTypes,
    Movie,
    Sources,
    Status,
)
from app.providers import services


def search_result(media_id, title, release_year=None):
    """Return a TMDB-like movie search result."""
    return {
        "media_id": media_id,
        "source": Sources.TMDB.value,
        "media_type": MediaTypes.MOVIE.value,
        "title": title,
        "image": settings.IMG_NONE,
        "release_year": release_year,
    }


class AIRecommendationsTestCase(TestCase):
    """Base test case with a user and helpers to create rated media."""

    def setUp(self):
        """Create a user and clear the cache."""
        cache.clear()
        self.user = get_user_model().objects.create_user(username="test")

    def create_movie(self, media_id, title, score=None, status=Status.PLANNING):
        """Create a movie entry for the user."""
        item = Item.objects.create(
            media_id=media_id,
            source=Sources.TMDB.value,
            media_type=MediaTypes.MOVIE.value,
            title=title,
            image=settings.IMG_NONE,
        )
        return Movie.objects.create(
            item=item,
            user=self.user,
            score=score,
            status=status.value,
        )

    def create_anime(self, media_id, title, score=None):
        """Create an anime entry for the user."""
        item = Item.objects.create(
            media_id=media_id,
            source=Sources.MAL.value,
            media_type=MediaTypes.ANIME.value,
            title=title,
            image=settings.IMG_NONE,
        )
        return Anime.objects.create(
            item=item,
            user=self.user,
            score=score,
            status=Status.PLANNING.value,
        )


class TasteTests(AIRecommendationsTestCase):
    """Test the taste profile sent to Gemini."""

    def test_rated_media_split_by_thresholds(self):
        """Split rated media into liked and disliked using the user thresholds."""
        self.create_movie("1", "Great Movie", score=9)
        self.create_movie("2", "Okay Movie", score=6)
        self.create_movie("3", "Bad Movie", score=2.5)
        self.create_anime("4", "Great Anime", score=8)

        liked, disliked = ai_recommendations.get_rated_media(self.user)

        self.assertEqual(
            liked,
            ["- Great Movie (movie) — 9/10", "- Great Anime (anime) — 8/10"],
        )
        self.assertEqual(disliked, ["- Bad Movie (movie) — 2.5/10"])

    def test_rated_media_respects_include_flags(self):
        """Skip liked or disliked media when excluded from the prompt."""
        self.create_movie("1", "Great Movie", score=9)
        self.create_movie("2", "Bad Movie", score=1)

        self.user.ai_include_liked = False
        liked, disliked = ai_recommendations.get_rated_media(self.user)
        self.assertEqual(liked, [])
        self.assertEqual(disliked, ["- Bad Movie (movie) — 1/10"])

        self.user.ai_include_liked = True
        self.user.ai_include_disliked = False
        liked, disliked = ai_recommendations.get_rated_media(self.user)
        self.assertEqual(liked, ["- Great Movie (movie) — 9/10"])
        self.assertEqual(disliked, [])

    def test_rated_media_custom_thresholds(self):
        """Use the user's slider values as thresholds."""
        self.create_movie("1", "Good Movie", score=7)
        self.create_movie("2", "Mediocre Movie", score=5)
        self.user.ai_liked_min_score = 7
        self.user.ai_disliked_max_score = 5

        liked, disliked = ai_recommendations.get_rated_media(self.user)

        self.assertEqual(liked, ["- Good Movie (movie) — 7/10"])
        self.assertEqual(disliked, ["- Mediocre Movie (movie) — 5/10"])

    def test_rated_media_skips_disabled_media_types(self):
        """Ignore ratings of media types the user disabled."""
        self.create_anime("1", "Great Anime", score=9)
        self.user.anime_enabled = False

        liked, _ = ai_recommendations.get_rated_media(self.user)

        self.assertEqual(liked, [])

    def test_build_taste_without_ratings(self):
        """Ask for acclaimed titles when there are no ratings."""
        taste = ai_recommendations.build_taste(self.user, MediaTypes.MOVIE.value)

        self.assertEqual(taste, ai_recommendations.NO_RATINGS)

    def test_build_taste_includes_library(self):
        """List tracked titles of the media type so they aren't recommended."""
        self.create_movie("1", "Great Movie", score=9)
        self.create_movie("2", "Planned Movie")
        self.create_anime("3", "Some Anime")

        taste = ai_recommendations.build_taste(self.user, MediaTypes.MOVIE.value)

        self.assertEqual(
            taste,
            "Titles the user liked (score >= 8):\n"
            "- Great Movie (movie) — 9/10\n\n"
            "Already in library (do not recommend):\n"
            "- Planned Movie\n"
            "- Great Movie",
        )


class WishesTests(AIRecommendationsTestCase):
    """Test the user-written wishes added to the prompt."""

    def test_build_taste_includes_wishes_of_media_type(self):
        """Add the positive and negative wishes of the same media type only."""
        self.user.ai_prompts = {
            MediaTypes.MOVIE.value: {
                "positive": "  I love European films  ",
                "negative": "No Russian content",
            },
            MediaTypes.ANIME.value: {"positive": "Only isekai"},
        }

        taste = ai_recommendations.build_taste(self.user, MediaTypes.MOVIE.value)

        self.assertTrue(
            taste.endswith(
                "The user's own wishes (follow them):\n"
                "I love European films\n\n"
                "The user explicitly does not want "
                "(never recommend anything matching this):\n"
                "No Russian content",
            ),
        )
        self.assertNotIn("isekai", taste)

    def test_build_taste_skips_empty_wishes(self):
        """Leave out blank wishes."""
        self.user.ai_prompts = {MediaTypes.MOVIE.value: {"positive": "   "}}

        taste = ai_recommendations.build_taste(self.user, MediaTypes.MOVIE.value)

        self.assertEqual(taste, ai_recommendations.NO_RATINGS)

    def test_invalidate_media_type(self):
        """Delete only the cached recommendations of one media type."""
        movie_key = ai_recommendations.global_cache_key(self.user.id, "movie")
        movie_media_key = ai_recommendations.media_cache_key(
            self.user.id,
            "movie",
            "tmdb",
            "1",
        )
        anime_key = ai_recommendations.global_cache_key(self.user.id, "anime")
        for key in (movie_key, movie_media_key, anime_key):
            cache.set(key, 1)

        ai_recommendations.invalidate_media_type(self.user, "movie")

        self.assertIsNone(cache.get(movie_key))
        self.assertIsNone(cache.get(movie_media_key))
        self.assertEqual(cache.get(anime_key), 1)


class ResolveTests(AIRecommendationsTestCase):
    """Test matching Gemini titles to provider search results."""

    def test_best_match_prefers_title_and_year(self):
        """Pick the result with the same title and year."""
        results = [
            search_result(1, "Other"),
            search_result(2, "Dune", 1984),
            search_result(3, "Dune", 2021),
        ]
        recommendation = {"title": "Dune", "year": 2021, "reason": ""}

        match = ai_recommendations.best_match(results, recommendation)

        self.assertEqual(match["media_id"], 3)

    def test_best_match_ignores_punctuation_and_accents(self):
        """Normalize titles before comparing them."""
        results = [search_result(1, "Other"), search_result(2, "Amélie")]
        recommendation = {"title": "amelie!", "year": None, "reason": ""}

        match = ai_recommendations.best_match(results, recommendation)

        self.assertEqual(match["media_id"], 2)

    def test_best_match_requires_exact_title(self):
        """Skip recommendations without a result of the same title."""
        results = [search_result(1, "First"), search_result(2, "Second")]
        recommendation = {"title": "Unknown", "year": None, "reason": ""}

        self.assertIsNone(ai_recommendations.best_match(results, recommendation))
        self.assertIsNone(ai_recommendations.best_match([], recommendation))

    @patch("app.providers.services.search")
    def test_resolve_skips_errors_and_duplicates(self, mock_search):
        """Skip titles that fail to resolve or resolve to the same media."""
        error = services.ProviderAPIError(Sources.TMDB.value, Exception("boom"))
        mock_search.side_effect = [
            {"results": [search_result(1, "Dune", 2021)]},
            error,
            {"results": [search_result(1, "Dune", 2021)]},
            {"results": []},
        ]
        recommendations = [
            {"title": "Dune", "year": 2021, "reason": "Epic"},
            {"title": "Broken", "year": None, "reason": "x"},
            {"title": "Dune: Part One", "year": 2021, "reason": "Again"},
            {"title": "Nothing", "year": None, "reason": "x"},
        ]

        items = ai_recommendations.resolve(recommendations, MediaTypes.MOVIE.value)

        self.assertEqual(items, [{**search_result(1, "Dune", 2021), "reason": "Epic"}])
        mock_search.assert_any_call(
            MediaTypes.MOVIE.value,
            "Dune",
            1,
            Sources.TMDB.value,
        )


class CacheTests(AIRecommendationsTestCase):
    """Test caching of generated recommendations."""

    def test_get_or_generate_caches(self):
        """Generate once and reuse the cached value."""
        generate = MagicMock(return_value=["a"])

        first = ai_recommendations.get_or_generate("key", generate)
        second = ai_recommendations.get_or_generate("key", generate)

        self.assertEqual(first, second)
        self.assertEqual(first["items"], ["a"])
        generate.assert_called_once()

    def test_get_or_generate_force(self):
        """Regenerate when forced."""
        generate = MagicMock(side_effect=[["a"], ["b"]])

        ai_recommendations.get_or_generate("key", generate)
        data = ai_recommendations.get_or_generate("key", generate, force=True)

        self.assertEqual(data["items"], ["b"])
        self.assertEqual(cache.get("key")["items"], ["b"])

    def test_get_or_generate_in_progress(self):
        """Raise when another request holds the lock and nothing is cached."""
        cache.add("ai_lock_key", "other")

        with self.assertRaises(ai_recommendations.RecommendationsInProgressError):
            ai_recommendations.get_or_generate("key", MagicMock())

    def test_get_or_generate_in_progress_returns_cached(self):
        """Return the cached value while another request regenerates it."""
        cache.set("key", {"items": ["old"]})
        cache.add("ai_lock_key", "other")

        data = ai_recommendations.get_or_generate("key", MagicMock())

        self.assertEqual(data["items"], ["old"])

    def test_get_or_generate_force_in_progress(self):
        """Don't pass off cached data as regenerated when the lock is held."""
        cache.set("key", {"items": ["old"]})
        cache.add("ai_lock_key", "other")

        with self.assertRaises(ai_recommendations.RecommendationsInProgressError):
            ai_recommendations.get_or_generate("key", MagicMock(), force=True)

    def test_get_or_generate_keeps_lock_of_other_request(self):
        """Don't release a lock that expired and was taken by another request."""

        def generate():
            cache.set("ai_lock_key", "other")
            return ["a"]

        ai_recommendations.get_or_generate("key", generate)

        self.assertEqual(cache.get("ai_lock_key"), "other")

    @override_settings(GEMINI_MODELS=["a", "b"], GEMINI_TIMEOUT=30)
    def test_lock_timeout_covers_every_model(self):
        """Hold the lock while every model may time out."""
        self.assertEqual(
            ai_recommendations.lock_timeout(),
            60 + ai_recommendations.LOCK_EXTRA_TIMEOUT,
        )

    def test_invalidation_keeps_lock(self):
        """Keep the generation lock when the user's cache is invalidated."""
        key = ai_recommendations.global_cache_key(self.user.id, "movie")
        cache.add(f"ai_lock_{key}", "token")

        ai_recommendations.invalidate_user(self.user)
        ai_recommendations.invalidate_media_type(self.user, "movie")

        self.assertEqual(cache.get(f"ai_lock_{key}"), "token")

    def test_get_or_generate_releases_lock_on_error(self):
        """Release the lock when generation fails."""
        generate = MagicMock(side_effect=ValueError)

        with self.assertRaises(ValueError):
            ai_recommendations.get_or_generate("key", generate)

        self.assertIsNone(cache.get("ai_lock_key"))

    def test_invalidate_user(self):
        """Delete only the user's cached recommendations."""
        cache.set(ai_recommendations.global_cache_key(self.user.id, "movie"), 1)
        cache.set(
            ai_recommendations.media_cache_key(self.user.id, "movie", "tmdb", "1"),
            1,
        )
        other_key = ai_recommendations.global_cache_key(self.user.id + 10, "movie")
        cache.set(other_key, 1)

        ai_recommendations.invalidate_user(self.user)

        self.assertIsNone(
            cache.get(ai_recommendations.global_cache_key(self.user.id, "movie")),
        )
        self.assertEqual(cache.get(other_key), 1)


class GenerationTests(AIRecommendationsTestCase):
    """Test generating recommendations end to end with mocked providers."""

    @patch("app.providers.services.search")
    @patch("app.providers.gemini.recommend")
    def test_global_recommendations(self, mock_recommend, mock_search):
        """Build the global prompt and resolve the response."""
        self.create_movie("1", "Great Movie", score=9)
        mock_recommend.return_value = [
            {"title": "Dune", "year": 2021, "reason": "Epic"},
        ]
        mock_search.return_value = {"results": [search_result(10, "Dune", 2021)]}

        data = ai_recommendations.global_recommendations(
            self.user,
            MediaTypes.MOVIE.value,
        )

        self.assertEqual(data["items"][0]["media_id"], 10)
        system_instruction, prompt = mock_recommend.call_args.args
        self.assertIn("as it appears on The Movie Database", system_instruction)
        count = ai_recommendations.GLOBAL_COUNT + ai_recommendations.EXTRA_REQUESTED
        self.assertTrue(prompt.startswith(f"Recommend {count} movies for this user."))
        self.assertIn("- Great Movie (movie) — 9/10", prompt)

        # Second call is served from the cache
        ai_recommendations.global_recommendations(self.user, MediaTypes.MOVIE.value)
        mock_recommend.assert_called_once()

    @patch("app.providers.services.search")
    @patch("app.providers.services.get_media_metadata")
    @patch("app.providers.gemini.recommend")
    def test_media_recommendations(self, mock_recommend, mock_metadata, mock_search):
        """Describe the viewed media and exclude it from the results."""
        self.create_movie("1", "Viewed Movie", score=7)
        mock_metadata.return_value = {
            "title": "Viewed Movie",
            "genres": ["Drama", "Sci-Fi"],
            "synopsis": "A story.",
        }
        mock_recommend.return_value = [
            {"title": "Viewed Movie", "year": None, "reason": "Same"},
            {"title": "Other Movie", "year": None, "reason": "Similar"},
        ]
        mock_search.side_effect = [
            {"results": [search_result("1", "Viewed Movie")]},
            {"results": [search_result("2", "Other Movie")]},
        ]

        data = ai_recommendations.media_recommendations(
            self.user,
            MediaTypes.MOVIE.value,
            Sources.TMDB.value,
            "1",
        )

        self.assertEqual([item["media_id"] for item in data["items"]], ["2"])
        _, prompt = mock_recommend.call_args.args
        count = ai_recommendations.MEDIA_COUNT + ai_recommendations.EXTRA_REQUESTED
        self.assertTrue(
            prompt.startswith(
                "The user is viewing this movie:\n"
                "Title: Viewed Movie\n"
                "Genres: Drama, Sci-Fi\n"
                "Synopsis: A story.\n"
                "The user rated it 7/10.\n\n"
                f"Recommend {count} movies similar to this one",
            ),
        )

    def test_enrich_drops_tracked_media_and_limits(self):
        """Hide media already in the library and cap the results."""
        self.create_movie("1", "Tracked Movie")
        items = [
            search_result("1", "Tracked Movie"),
            search_result("2", "New Movie"),
            search_result("3", "Another Movie"),
        ]
        request = RequestFactory().get("/")
        request.user = self.user

        results = ai_recommendations.enrich(request, items, limit=1)

        self.assertEqual([result["item"]["media_id"] for result in results], ["2"])
