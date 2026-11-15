import threading
import time
from unittest.mock import MagicMock, patch

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import RequestFactory, TestCase, override_settings
from django.utils import timezone

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


def search_by_title(responses):
    """Return a search mock side effect answering by query, in any call order."""

    def search(_media_type, query, _page, _source=None):
        response = responses[query]
        if isinstance(response, Exception):
            raise response
        return response

    return search


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
    """Test the taste profile sent to the LLM."""

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
    """Test matching LLM titles to provider search results."""

    def test_best_match_prefers_title_and_year(self):
        """Pick the result with the same title and year."""
        results = [
            search_result(1, "Other"),
            search_result(2, "Dune", 1984),
            search_result(3, "Dune", 2021),
        ]
        recommendation = {"title": "Dune", "year": 2021}

        match = ai_recommendations.best_match(results, recommendation)

        self.assertEqual(match["media_id"], 3)

    def test_best_match_ignores_punctuation_and_accents(self):
        """Normalize titles before comparing them."""
        results = [search_result(1, "Other"), search_result(2, "Amélie")]
        recommendation = {"title": "amelie!", "year": None}

        match = ai_recommendations.best_match(results, recommendation)

        self.assertEqual(match["media_id"], 2)

    def test_best_match_requires_exact_title(self):
        """Skip recommendations without a result of the same title."""
        results = [search_result(1, "First"), search_result(2, "Second")]
        recommendation = {"title": "Unknown", "year": None}

        self.assertIsNone(ai_recommendations.best_match(results, recommendation))
        self.assertIsNone(ai_recommendations.best_match([], recommendation))

    @patch("app.providers.services.search")
    def test_resolve_skips_errors_and_duplicates(self, mock_search):
        """Skip titles that fail to resolve or resolve to the same media."""
        error = services.ProviderAPIError(Sources.TMDB.value, Exception("boom"))
        mock_search.side_effect = search_by_title(
            {
                "Dune": {"results": [search_result(1, "Dune", 2021)]},
                "Broken": error,
                "Dune: Part One": {"results": [search_result(1, "Dune", 2021)]},
                "Nothing": {"results": []},
            },
        )
        recommendations = [
            {"title": "Dune", "year": 2021},
            {"title": "Broken", "year": None},
            {"title": "Dune: Part One", "year": 2021},
            {"title": "Nothing", "year": None},
        ]

        items = ai_recommendations.resolve(recommendations, MediaTypes.MOVIE.value)

        self.assertEqual(items, [search_result(1, "Dune", 2021)])
        mock_search.assert_any_call(
            MediaTypes.MOVIE.value,
            "Dune",
            1,
            Sources.TMDB.value,
        )

    @patch("app.providers.services.search")
    def test_resolve_searches_while_streaming(self, mock_search):
        """Start searching before the stream ends and keep the stream's order."""
        first_searched = threading.Event()

        def search(_media_type, query, _page, _source=None):
            if query == "First":
                first_searched.set()
                # Finish after the second search to check the result order
                time.sleep(0.05)
            return {"results": [search_result(query, query)]}

        def stream():
            yield {"title": "First", "year": None}
            self.assertTrue(first_searched.wait(timeout=5))
            yield {"title": "Second", "year": None}

        mock_search.side_effect = search

        items = ai_recommendations.resolve(stream(), MediaTypes.MOVIE.value)

        self.assertEqual([item["title"] for item in items], ["First", "Second"])


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

    @override_settings(AI_PROVIDER="claude", CLAUDE_MODELS=["a"], CLAUDE_TIMEOUT=20)
    def test_lock_timeout_uses_selected_provider(self):
        """Use the timeout settings of the selected provider."""
        self.assertEqual(
            ai_recommendations.lock_timeout(),
            2 * 20 + ai_recommendations.LOCK_EXTRA_TIMEOUT,
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
            {"title": "Dune", "year": 2021},
        ]
        mock_search.return_value = {"results": [search_result(10, "Dune", 2021)]}

        data = ai_recommendations.generate_global_recommendations(
            self.user,
            MediaTypes.MOVIE.value,
        )

        self.assertEqual(data["items"][0]["media_id"], 10)
        self.assertFalse(data["partial"])
        self.assertEqual(
            cache.get(
                ai_recommendations.global_cache_key(
                    self.user.id,
                    MediaTypes.MOVIE.value,
                ),
            ),
            data,
        )
        system_instruction, prompt = mock_recommend.call_args.args
        self.assertIn("as it appears on The Movie Database", system_instruction)
        count = ai_recommendations.GLOBAL_COUNT + ai_recommendations.EXTRA_REQUESTED
        self.assertTrue(prompt.startswith(f"Recommend {count} movies for this user."))
        self.assertIn("- Great Movie (movie) — 9/10", prompt)

    @override_settings(AI_PROVIDER="claude")
    @patch("app.providers.services.search")
    @patch("app.providers.gemini.recommend")
    @patch("app.providers.claude.recommend")
    def test_uses_selected_provider(self, mock_claude, mock_gemini, mock_search):
        """Generate with Claude when it is the selected provider."""
        mock_claude.return_value = [{"title": "Dune", "year": 2021}]
        mock_search.return_value = {"results": [search_result(10, "Dune", 2021)]}

        ai_recommendations.generate_global_recommendations(
            self.user,
            MediaTypes.MOVIE.value,
        )

        mock_claude.assert_called_once()
        mock_gemini.assert_not_called()

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
            {"title": "Viewed Movie", "year": None},
            {"title": "Other Movie", "year": None},
        ]
        mock_search.side_effect = search_by_title(
            {
                "Viewed Movie": {"results": [search_result("1", "Viewed Movie")]},
                "Other Movie": {"results": [search_result("2", "Other Movie")]},
            },
        )

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


class PoolExtensionTests(AIRecommendationsTestCase):
    """Test extending the global recommendation pool in the background."""

    def setUp(self):
        """Cache a small pool of movie recommendations."""
        super().setUp()
        self.key = ai_recommendations.global_cache_key(
            self.user.id,
            MediaTypes.MOVIE.value,
        )
        cache.set(
            self.key,
            {
                "generated_at": timezone.now(),
                "items": [search_result(1, "Dune", 2021)],
            },
        )

    @patch("app.providers.services.search")
    @patch("app.providers.gemini.recommend")
    def test_extend_adds_new_items(self, mock_recommend, mock_search):
        """Append unseen titles, skip pool duplicates and exclude pool titles."""
        mock_recommend.return_value = [
            {"title": "Dune", "year": 2021},
            {"title": "Arrival", "year": 2016},
        ]
        mock_search.side_effect = search_by_title(
            {
                "Dune": {"results": [search_result(1, "Dune", 2021)]},
                "Arrival": {"results": [search_result(2, "Arrival", 2016)]},
            },
        )

        extended = ai_recommendations.extend_global_recommendations(
            self.user,
            MediaTypes.MOVIE.value,
        )

        self.assertTrue(extended)
        data = cache.get(self.key)
        self.assertEqual(
            [item["title"] for item in data["items"]],
            ["Dune", "Arrival"],
        )
        self.assertEqual(data["extensions"], 1)
        prompt = mock_recommend.call_args.args[1]
        self.assertIn("Already recommended (do not repeat):\n- Dune", prompt)

    @patch("app.providers.gemini.recommend")
    def test_extend_skips_finished_pool(self, mock_recommend):
        """Stop after the maximum number of extensions."""
        data = cache.get(self.key)
        cache.set(
            self.key,
            {**data, "extensions": ai_recommendations.MAX_EXTENSIONS},
        )

        self.assertFalse(
            ai_recommendations.extend_global_recommendations(
                self.user,
                MediaTypes.MOVIE.value,
            ),
        )
        mock_recommend.assert_not_called()

    @patch("app.providers.gemini.recommend")
    def test_extend_skips_missing_pool(self, mock_recommend):
        """Don't extend a pool that doesn't exist anymore."""
        cache.delete(self.key)

        self.assertFalse(
            ai_recommendations.extend_global_recommendations(
                self.user,
                MediaTypes.MOVIE.value,
            ),
        )
        mock_recommend.assert_not_called()

    @patch("app.providers.gemini.recommend")
    def test_extend_skips_while_generating(self, mock_recommend):
        """Don't extend while another generation holds the lock."""
        cache.add(f"ai_lock_{self.key}", "other")

        self.assertFalse(
            ai_recommendations.extend_global_recommendations(
                self.user,
                MediaTypes.MOVIE.value,
            ),
        )
        mock_recommend.assert_not_called()

    @patch("app.providers.services.search")
    @patch("app.providers.gemini.recommend")
    def test_extend_discarded_after_invalidation(self, mock_recommend, mock_search):
        """Don't restore a pool invalidated during generation."""

        def invalidate(*_args):
            ai_recommendations.invalidate_media_type(
                self.user,
                MediaTypes.MOVIE.value,
            )
            return [{"title": "Arrival", "year": 2016}]

        mock_recommend.side_effect = invalidate
        mock_search.return_value = {"results": [search_result(2, "Arrival", 2016)]}

        self.assertFalse(
            ai_recommendations.extend_global_recommendations(
                self.user,
                MediaTypes.MOVIE.value,
            ),
        )
        self.assertIsNone(cache.get(self.key))


class GlobalPoolTests(AIRecommendationsTestCase):
    """Test serving and generating the global recommendation pool."""

    def setUp(self):
        """Prepare the pool cache key."""
        super().setUp()
        self.key = ai_recommendations.global_cache_key(
            self.user.id,
            MediaTypes.MOVIE.value,
        )

    def test_cached_pool_served(self):
        """Return the cached pool without queueing a generation."""
        cache.set(self.key, {"generated_at": timezone.now(), "items": []})
        queue_generation = MagicMock()

        data = ai_recommendations.global_recommendations(
            self.user,
            MediaTypes.MOVIE.value,
            queue_generation,
        )

        self.assertEqual(data["items"], [])
        queue_generation.assert_not_called()

    def test_missing_pool_queues_generation(self):
        """Queue a generation and report it in progress."""
        queue_generation = MagicMock()

        with self.assertRaises(ai_recommendations.RecommendationsInProgressError):
            ai_recommendations.global_recommendations(
                self.user,
                MediaTypes.MOVIE.value,
                queue_generation,
            )

        queue_generation.assert_called_once_with(self.user.id, MediaTypes.MOVIE.value)

    def test_force_drops_cached_pool(self):
        """Drop the cached pool and queue a new generation."""
        cache.set(self.key, {"generated_at": timezone.now(), "items": []})

        with self.assertRaises(ai_recommendations.RecommendationsInProgressError):
            ai_recommendations.global_recommendations(
                self.user,
                MediaTypes.MOVIE.value,
                MagicMock(),
                force=True,
            )

        self.assertIsNone(cache.get(self.key))

    @patch("app.providers.services.search")
    @patch("app.providers.gemini.recommend")
    def test_pool_cached_item_by_item(self, mock_recommend, mock_search):
        """Cache a partial pool while recommendations are still resolving."""
        mock_recommend.return_value = [
            {"title": "First", "year": None},
            {"title": "Second", "year": None},
        ]
        cached_titles = []

        def search(_media_type, query, _page, _source=None):
            if query == "Second":
                # Searches run concurrently, so wait for the first item
                for _ in range(100):
                    data = cache.get(self.key)
                    if data["items"]:
                        break
                    time.sleep(0.01)
                cached_titles.append(
                    ([item["title"] for item in data["items"]], data["partial"]),
                )
            return {"results": [search_result(query, query)]}

        mock_search.side_effect = search

        data = ai_recommendations.generate_global_recommendations(
            self.user,
            MediaTypes.MOVIE.value,
        )

        self.assertEqual(cached_titles, [(["First"], True)])
        self.assertEqual([item["title"] for item in data["items"]], ["First", "Second"])
        self.assertFalse(cache.get(self.key)["partial"])

    @patch("app.providers.gemini.recommend")
    def test_generation_error_cached(self, mock_recommend):
        """Cache the error briefly so the page shows it instead of polling."""
        mock_recommend.side_effect = services.ProviderAPIError(
            "gemini",
            Exception("boom"),
        )

        with self.assertRaises(services.ProviderAPIError):
            ai_recommendations.generate_global_recommendations(
                self.user,
                MediaTypes.MOVIE.value,
            )

        data = cache.get(self.key)
        self.assertIn("Gemini", data["error"])
        self.assertFalse(data["partial"])
        self.assertFalse(ai_recommendations.needs_extension(data))

    @patch("app.providers.services.search")
    @patch("app.providers.gemini.recommend")
    def test_generation_stops_after_invalidation(self, mock_recommend, mock_search):
        """Don't restore a pool invalidated during generation."""
        mock_recommend.return_value = [{"title": "First", "year": None}]

        def search(*_args):
            ai_recommendations.invalidate_media_type(
                self.user,
                MediaTypes.MOVIE.value,
            )
            return {"results": [search_result("First", "First")]}

        mock_search.side_effect = search

        self.assertIsNone(
            ai_recommendations.generate_global_recommendations(
                self.user,
                MediaTypes.MOVIE.value,
            ),
        )
        self.assertIsNone(cache.get(self.key))

    def test_generation_in_progress(self):
        """Raise when another generation holds the lock."""
        cache.add(f"ai_lock_{self.key}", "other")

        with self.assertRaises(ai_recommendations.RecommendationsInProgressError):
            ai_recommendations.generate_global_recommendations(
                self.user,
                MediaTypes.MOVIE.value,
            )

    def test_partial_pool_not_extended(self):
        """Don't extend a pool that's still being generated."""
        self.assertFalse(
            ai_recommendations.needs_extension({"items": [], "partial": True}),
        )

    @patch("app.providers.services.search")
    @patch("app.providers.gemini.recommend")
    def test_partial_pool_expires_with_lock(self, mock_recommend, mock_search):
        """Cache partial pools only as long as the lock, final pools for a day."""
        mock_recommend.return_value = [{"title": "First", "year": None}]
        mock_search.return_value = {"results": [search_result("First", "First")]}

        with patch.object(
            ai_recommendations.cache,
            "set",
            wraps=ai_recommendations.cache.set,
        ) as mock_set:
            ai_recommendations.generate_global_recommendations(
                self.user,
                MediaTypes.MOVIE.value,
            )

        timeouts = [call.args[2] for call in mock_set.call_args_list]
        self.assertEqual(timeouts[-1], ai_recommendations.CACHE_TIMEOUT)
        self.assertTrue(
            all(
                timeout == ai_recommendations.lock_timeout()
                for timeout in timeouts[:-1]
            ),
        )

    @patch("app.providers.services.search")
    @patch("app.providers.gemini.recommend")
    def test_unresolved_pool_cached_as_error(self, mock_recommend, mock_search):
        """Show an error instead of caching an empty pool for a day."""
        mock_recommend.return_value = [{"title": "Unknown", "year": None}]
        mock_search.return_value = {"results": []}

        with self.assertRaises(ai_recommendations.NoRecommendationsError):
            ai_recommendations.generate_global_recommendations(
                self.user,
                MediaTypes.MOVIE.value,
            )

        data = cache.get(self.key)
        self.assertIn("None of the recommended titles", data["error"])
        self.assertFalse(ai_recommendations.needs_extension(data))

    @patch("app.providers.gemini.recommend")
    def test_non_progressive_failure_keeps_pool(self, mock_recommend):
        """Keep the current pool when a background refresh fails."""
        pool = {"generated_at": timezone.now(), "items": [search_result(1, "Dune")]}
        cache.set(self.key, pool)
        mock_recommend.side_effect = services.ProviderAPIError(
            "gemini",
            Exception("boom"),
        )

        with self.assertRaises(services.ProviderAPIError):
            ai_recommendations.generate_global_recommendations(
                self.user,
                MediaTypes.MOVIE.value,
                progressive=False,
            )

        self.assertEqual(cache.get(self.key), pool)

    @patch("app.providers.services.search")
    @patch("app.providers.gemini.recommend")
    def test_non_progressive_replaces_pool(self, mock_recommend, mock_search):
        """Swap in the refreshed pool only once it's complete."""
        cache.set(
            self.key,
            {"generated_at": timezone.now(), "items": [search_result(1, "Dune")]},
        )
        mock_recommend.return_value = [{"title": "Arrival", "year": None}]
        mock_search.return_value = {"results": [search_result(2, "Arrival")]}

        data = ai_recommendations.generate_global_recommendations(
            self.user,
            MediaTypes.MOVIE.value,
            progressive=False,
        )

        self.assertEqual(cache.get(self.key), data)
        self.assertEqual([item["title"] for item in data["items"]], ["Arrival"])

    @patch("app.providers.services.search")
    def test_iter_resolved_stops_without_draining_stream(self, mock_search):
        """Return right away when the caller stops before the stream ends."""
        release = threading.Event()
        self.addCleanup(release.set)
        mock_search.return_value = {"results": [search_result("First", "First")]}

        def stream():
            yield {"title": "First", "year": None}
            # Stands in for an LLM that is still generating
            release.wait(timeout=5)
            yield {"title": "Second", "year": None}

        resolved = ai_recommendations.iter_resolved(stream(), MediaTypes.MOVIE.value)
        start = time.perf_counter()
        next(resolved)
        resolved.close()

        self.assertLess(time.perf_counter() - start, 1)
