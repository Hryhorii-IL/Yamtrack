from __future__ import annotations

from django.conf import settings

from app.ai_recommendations.bl.clients import llm_client_class
from app.ai_recommendations.bl.types.provider import ProviderInfo


def get_provider_info() -> ProviderInfo:
    """Return the selected AI provider and whether it has an API key."""
    client_class = llm_client_class()
    return ProviderInfo(
        label=client_class.label,
        api_key_setting=client_class.api_key_setting,
        configured=bool(getattr(settings, client_class.api_key_setting)),
    )
