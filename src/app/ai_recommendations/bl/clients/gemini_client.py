from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
from google import genai
from google.genai import errors, types

from app.ai_recommendations.bl.calculations.stream_parsing import ItemStreamParser
from app.ai_recommendations.bl.clients.llm_client import LLMClient
from app.ai_recommendations.bl.clients.schema import RecommendationListSchema

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from django.core.cache.backends.base import BaseCache

    from app.ai_recommendations.bl.types.recommendation import RecommendedTitle


class GeminiClient(LLMClient):
    """Streams recommendations from Gemini models."""

    provider = "gemini"
    label = "Gemini"
    api_key_setting = "GEMINI_API_KEY"
    models_setting = "GEMINI_MODELS"
    timeout_setting = "GEMINI_TIMEOUT"

    def __init__(
        self,
        *,
        api_key: str,
        models: Sequence[str],
        timeout: int,
        sdk: genai.Client | None = None,
        cache: BaseCache | None = None,
    ) -> None:
        """Create the Gemini SDK client unless a test passes one."""
        super().__init__(models=models, timeout=timeout, cache=cache)
        self.sdk = sdk or genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=timeout * 1000),
        )

    def generate(
        self,
        model: str,
        system_instruction: str,
        prompt: str,
    ) -> Iterator[RecommendedTitle]:
        """Yield recommendations as a single Gemini model streams them."""
        parser = ItemStreamParser()
        try:
            for chunk in self.sdk.models.generate_content_stream(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=RecommendationListSchema,
                ),
            ):
                yield from parser.feed(chunk.text or "")
        except errors.APIError as error:
            raise self.error(error.code, f"{model}: {error.message}") from error
        except httpx.HTTPError as error:
            details = f"{model}: {error or type(error).__name__}"
            raise self.error(None, details) from error
