import importlib
import json
import logging
import re

import pydantic
import requests
from django.conf import settings
from django.core.cache import cache

from app.providers import services

logger = logging.getLogger(__name__)

AUTH_ERROR_CODES = (401, 403)
# How long a failed model is tried after the others
UNAVAILABLE_TIMEOUT = 60 * 10

ITEMS_START = re.compile(r'"items"\s*:\s*\[')

PROVIDERS = {
    "gemini": {"label": "Gemini", "api_key_setting": "GEMINI_API_KEY"},
    "claude": {"label": "Claude", "api_key_setting": "ANTHROPIC_API_KEY"},
}


class Recommendation(pydantic.BaseModel):
    """A single title recommended by an LLM."""

    title: str
    year: int | None = None


class RecommendationList(pydantic.BaseModel):
    """Structured response returned by an LLM."""

    items: list[Recommendation]


class ItemStreamParser:
    """Extract complete recommendations from a streamed RecommendationList JSON."""

    def __init__(self):
        """Start with an empty buffer."""
        self.buffer = ""
        self.position = None
        self.decoder = json.JSONDecoder()

    def feed(self, text):
        """Add a chunk of JSON text and yield the newly completed items."""
        self.buffer += text
        if self.position is None:
            match = ITEMS_START.search(self.buffer)
            if match is None:
                return
            self.position = match.end()

        while True:
            position = self.position
            while position < len(self.buffer) and self.buffer[position] in " \t\r\n,":
                position += 1
            self.position = position
            if position >= len(self.buffer) or self.buffer[position] == "]":
                return
            try:
                item, self.position = self.decoder.raw_decode(self.buffer, position)
            except json.JSONDecodeError:
                # The item is incomplete, wait for the next chunk
                return
            try:
                yield Recommendation.model_validate(item).model_dump()
            except pydantic.ValidationError:
                logger.warning("Skipping invalid streamed recommendation %s", item)


def provider_module():
    """Return the module of the configured AI provider."""
    return importlib.import_module(f"app.providers.{settings.AI_PROVIDER}")


def provider_label():
    """Return the display name of the configured AI provider."""
    return PROVIDERS[settings.AI_PROVIDER]["label"]


def api_key_setting():
    """Return the setting name holding the configured AI provider's API key."""
    return PROVIDERS[settings.AI_PROVIDER]["api_key_setting"]


def is_configured():
    """Return whether the configured AI provider has an API key."""
    return bool(getattr(settings, api_key_setting()))


def recommend(provider, models, generate, system_instruction, prompt):
    """Yield recommendation dicts from the first configured model that works.

    ``generate(model, system_instruction, prompt)`` yields a single model's
    recommendation dicts and raises a ProviderAPIError when it fails. Falls
    back to the next model only while nothing has been yielded yet.
    """
    label = PROVIDERS[provider]["label"]
    if not models:
        raise_api_error(provider, None, f"No {label} models configured")

    *fallback_models, last_model = get_models(provider, models)
    for model in fallback_models:
        try:
            yield from generate_or_skip(
                provider,
                generate,
                model,
                system_instruction,
                prompt,
            )
        except services.ProviderAPIError as error:
            # The API key is shared by all models, so retrying won't help
            if error.status_code in AUTH_ERROR_CODES:
                raise
            logger.warning(
                "%s model %s failed, trying the next model",
                label,
                model,
            )
        else:
            return

    yield from generate_or_skip(
        provider,
        generate,
        last_model,
        system_instruction,
        prompt,
    )


def get_models(provider, models):
    """Return the configured models with recently failed ones moved last."""
    unavailable = cache.get_many(
        [unavailable_cache_key(provider, model) for model in models],
    )
    return sorted(
        models,
        key=lambda model: unavailable_cache_key(provider, model) in unavailable,
    )


def generate_or_skip(provider, generate, model, system_instruction, prompt):
    """Yield a model's recommendations and skip the model for a while if it fails.

    A failure after some recommendations were yielded keeps them instead of
    raising, since the caller may already be using them.
    """
    yielded = False
    try:
        for recommendation in generate(model, system_instruction, prompt):
            yielded = True
            yield recommendation
        if not yielded:
            raise_api_error(provider, None, f"{model}: invalid response")
    except services.ProviderAPIError as error:
        if error.status_code not in AUTH_ERROR_CODES:
            cache.set(unavailable_cache_key(provider, model), 1, UNAVAILABLE_TIMEOUT)
        if not yielded:
            raise
        logger.warning(
            "%s model %s failed mid-stream, keeping partial recommendations",
            PROVIDERS[provider]["label"],
            model,
        )


def unavailable_cache_key(provider, model):
    """Return the cache key marking a model as recently failed."""
    return f"{provider}_unavailable_{model}"


def raise_api_error(provider, status_code, message):
    """Raise a ProviderAPIError with the LLM status code and message."""
    mock_response = type(
        "obj",
        (object,),
        {"status_code": status_code, "text": message},
    )()
    mock_error = requests.exceptions.HTTPError(response=mock_response)
    raise services.ProviderAPIError(provider, mock_error, message)
