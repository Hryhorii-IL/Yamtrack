"""In-memory stores, scripted clients and a fixed clock for the tests."""

from app.ai_recommendations.tests.fakes.builders import NOW, resolved, taste
from app.ai_recommendations.tests.fakes.clock import FixedClock
from app.ai_recommendations.tests.fakes.in_memory_library_store import (
    FakeLibrary,
    InMemoryLibraryStore,
)
from app.ai_recommendations.tests.fakes.in_memory_pool_store import InMemoryPoolStore
from app.ai_recommendations.tests.fakes.llm_client import FakeLLMClient
from app.ai_recommendations.tests.fakes.search_client import FakeSearchClient

__all__ = [
    "NOW",
    "FakeLLMClient",
    "FakeLibrary",
    "FakeSearchClient",
    "FixedClock",
    "InMemoryLibraryStore",
    "InMemoryPoolStore",
    "resolved",
    "taste",
]
