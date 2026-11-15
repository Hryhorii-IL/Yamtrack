from __future__ import annotations

from typing import TYPE_CHECKING, Final

from django.conf import settings

from app.ai_recommendations.bl.clients.claude_client import ClaudeClient
from app.ai_recommendations.bl.clients.gemini_client import GeminiClient
from app.ai_recommendations.bl.clients.media_search_client import MediaSearchClient

if TYPE_CHECKING:
    from app.ai_recommendations.bl.clients.llm_client import LLMClient

LLM_CLIENTS: Final[dict[str, type[LLMClient]]] = {
    GeminiClient.provider: GeminiClient,
    ClaudeClient.provider: ClaudeClient,
}


def make_llm_client() -> LLMClient:
    """Return the client of the AI provider selected by AI_PROVIDER."""
    client_class = ClaudeClient if llm_client_class() is ClaudeClient else GeminiClient
    return client_class(
        api_key=getattr(settings, client_class.api_key_setting),
        models=getattr(settings, client_class.models_setting),
        timeout=getattr(settings, client_class.timeout_setting),
    )


def make_media_search_client() -> MediaSearchClient:
    """Return the client finding recommended titles on their providers."""
    return MediaSearchClient()


def llm_client_class() -> type[LLMClient]:
    """Return the client class of the selected AI provider."""
    return LLM_CLIENTS[settings.AI_PROVIDER]


def llm_max_duration() -> int:
    """Return how long the selected provider may take, without building a client."""
    client_class = llm_client_class()
    return client_class.max_duration_of(
        getattr(settings, client_class.models_setting),
        getattr(settings, client_class.timeout_setting),
    )
