from __future__ import annotations

import logging
import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

from django.db import connections

from app import config
from app.ai_recommendations.bl.calculations.matching import best_match
from app.ai_recommendations.bl.types.constants import RESOLVE_WORKERS
from app.ai_recommendations.bl.types.recommendation import ResolvedTitle
from app.providers import services

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable

    from app.ai_recommendations.bl.types.recommendation import RecommendedTitle

logger = logging.getLogger(__name__)

type Search = Callable[[str, str, int, str], dict]
type SearchEntry = tuple[RecommendedTitle, Future[list[ResolvedTitle] | None]]


class MediaSearchClient:
    """Finds recommended titles on the default source of their media type."""

    def __init__(
        self,
        *,
        search: Search = services.search,
        workers: int = RESOLVE_WORKERS,
    ) -> None:
        """Keep the provider search function and the number of parallel searches."""
        self.search = search
        self.workers = workers

    def resolve(
        self,
        recommendations: Iterable[RecommendedTitle],
        media_type: str,
    ) -> Generator[ResolvedTitle]:
        """Yield the found recommendations in their order, without duplicates.

        Each search starts as soon as its recommendation arrives, so streamed
        recommendations are resolved while the LLM is still generating.
        """
        source = config.get_default_source_name(media_type).value
        searches: queue.Queue[SearchEntry | None] = queue.Queue()
        stopped = threading.Event()
        # One extra worker consumes the (possibly streamed) recommendations
        executor = ThreadPoolExecutor(max_workers=self.workers + 1)

        def submit_searches() -> None:
            try:
                for recommendation in recommendations:
                    if stopped.is_set():
                        break
                    search = executor.submit(
                        self._search,
                        recommendation,
                        media_type,
                        source,
                    )
                    searches.put((recommendation, search))
            finally:
                searches.put(None)

        producer = executor.submit(submit_searches)
        try:
            seen = set()
            while (entry := searches.get()) is not None:
                recommendation, search = entry
                results = search.result()
                if results is None:
                    continue

                match = best_match(results, recommendation)
                if match is None or match.key in seen:
                    continue

                seen.add(match.key)
                yield match

            # Raise LLM errors once the resolved matches were yielded
            producer.result()
        finally:
            # Don't wait for the rest of the LLM stream when the caller stops early
            stopped.set()
            executor.shutdown(wait=False, cancel_futures=True)

    def _search(
        self,
        recommendation: RecommendedTitle,
        media_type: str,
        source: str,
    ) -> list[ResolvedTitle] | None:
        """Return the search results of a recommendation, or None on errors."""
        try:
            results = self.search(media_type, recommendation.title, 1, source)
        except services.ProviderAPIError:
            logger.warning("Could not resolve AI recommendation %s", recommendation)
            return None
        finally:
            # Worker threads don't go through Django's request cleanup
            connections.close_all()
        return [to_resolved_title(result) for result in results["results"]]


def to_resolved_title(result: dict) -> ResolvedTitle:
    """Return a provider search result as a resolved title."""
    return ResolvedTitle(
        media_id=str(result["media_id"]),
        source=result["source"],
        media_type=result["media_type"],
        title=result["title"],
        image=result["image"],
        release_year=result.get("release_year"),
    )
