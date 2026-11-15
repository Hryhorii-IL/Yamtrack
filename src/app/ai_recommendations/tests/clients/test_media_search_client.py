import threading
import time

from django.test import TestCase

from app.ai_recommendations.bl.clients.media_search_client import MediaSearchClient
from app.ai_recommendations.bl.types.recommendation import RecommendedTitle
from app.ai_recommendations.tests.fakes import resolved
from app.providers import services

SOURCE = "tmdb"


def result(media_id, title, year=None):
    """Return a TMDB movie search result."""
    return resolved(media_id, title, year).to_card()


class MediaSearchClientTests(TestCase):
    """Test finding recommended titles on their default source."""

    def test_resolves_in_order_without_duplicates(self):
        """Skip failed, unmatched and duplicate titles, keep the LLM order."""

        def search(_media_type, query, _page, _source):
            if query == "Broken":
                raise services.ProviderAPIError(SOURCE, Exception(SOURCE))
            return {
                "Dune": {"results": [result(1, "Dune", 2021)]},
                "Dune again": {"results": [result(1, "Dune again")]},
                "Arrival": {"results": [result(2, "Arrival")]},
                "Nothing": {"results": []},
            }[query]

        titles = [
            RecommendedTitle(title=title)
            for title in ("Dune", "Broken", "Nothing", "Arrival", "Dune again")
        ]

        found = list(MediaSearchClient(search=search).resolve(titles, "movie"))

        # "Dune again" resolves to the media "Dune" already found
        self.assertEqual([title.title for title in found], ["Dune", "Arrival"])
        self.assertEqual(found[0].media_id, "1")

    def test_searches_while_streaming(self):
        """Start searching before the stream ends."""
        first_searched = threading.Event()

        def search(_media_type, query, _page, _source):
            if query == "First":
                first_searched.set()
                # Finish after the second search to check the result order
                time.sleep(0.05)
            return {"results": [result(query, query)]}

        def stream():
            yield RecommendedTitle(title="First")
            self.assertTrue(first_searched.wait(timeout=5))
            yield RecommendedTitle(title="Second")

        found = list(MediaSearchClient(search=search).resolve(stream(), "movie"))

        self.assertEqual([title.title for title in found], ["First", "Second"])

    def test_stops_without_draining_stream(self):
        """Return right away when the caller stops before the stream ends."""
        release = threading.Event()
        self.addCleanup(release.set)

        def stream():
            yield RecommendedTitle(title="First")
            # Stands in for an LLM that is still generating
            release.wait(timeout=5)
            yield RecommendedTitle(title="Second")

        client = MediaSearchClient(
            search=lambda *_args: {"results": [result("First", "First")]},
        )
        found = client.resolve(stream(), "movie")
        start = time.perf_counter()
        next(found)
        found.close()

        self.assertLess(time.perf_counter() - start, 1)
