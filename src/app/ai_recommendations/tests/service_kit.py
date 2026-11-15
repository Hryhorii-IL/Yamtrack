"""Builds services over the Django stores, with fake clients and a fixed clock."""

from app.ai_recommendations.bl.services.media_recommendations_service import (
    MediaRecommendationsService,
)
from app.ai_recommendations.bl.services.pool_extension_service import (
    PoolExtensionService,
)
from app.ai_recommendations.bl.services.pool_generation_service import (
    PoolGenerationService,
)
from app.ai_recommendations.bl.stores.django_library_store import DjangoLibraryStore
from app.ai_recommendations.tests.fakes import (
    NOW,
    FakeLLMClient,
    FakeSearchClient,
    FixedClock,
)


def generation_service(user_id, pool_store, **kwargs):
    """Return a movie pool generation service."""
    return PoolGenerationService(
        user_id,
        "movie",
        progressive=kwargs.pop("progressive", True),
        refresh=kwargs.pop("refresh", False),
        last_attempt=kwargs.pop("last_attempt", True),
        thinking=kwargs.pop("thinking", False),
        llm_client=kwargs.pop("llm_client", FakeLLMClient()),
        search_client=kwargs.pop("search_client", FakeSearchClient()),
        pool_store=pool_store,
        library_store=DjangoLibraryStore(),
        clock=FixedClock(NOW),
    )


def extension_service(user_id, pool_store, **kwargs):
    """Return a movie pool extension service."""
    return PoolExtensionService(
        user_id,
        "movie",
        thinking=kwargs.pop("thinking", False),
        llm_client=kwargs.pop("llm_client", FakeLLMClient()),
        search_client=kwargs.pop("search_client", FakeSearchClient()),
        pool_store=pool_store,
        library_store=DjangoLibraryStore(),
    )


def media_service(user_id, pool_store, **kwargs):
    """Return a service for recommendations similar to TMDB movie 1."""
    return MediaRecommendationsService(
        user_id,
        "movie",
        "tmdb",
        "1",
        thinking=kwargs.pop("thinking", False),
        llm_client=kwargs.pop("llm_client", FakeLLMClient()),
        search_client=kwargs.pop("search_client", FakeSearchClient()),
        pool_store=pool_store,
        library_store=DjangoLibraryStore(),
        clock=FixedClock(NOW),
        fetch_metadata=kwargs.pop("fetch_metadata", lambda *_args: {"title": "Viewed"}),
    )
