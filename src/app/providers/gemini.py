import functools

import httpx
from django.conf import settings
from google import genai
from google.genai import errors, types

from app.providers import llm

PROVIDER = "gemini"


def recommend(system_instruction, prompt):
    """Yield recommendation dicts from the first configured model that works."""
    client = genai.Client(
        api_key=settings.GEMINI_API_KEY,
        http_options=types.HttpOptions(timeout=settings.GEMINI_TIMEOUT * 1000),
    )
    return llm.recommend(
        PROVIDER,
        settings.GEMINI_MODELS,
        functools.partial(generate, client),
        system_instruction,
        prompt,
    )


def max_duration():
    """Return how long a request can take when every model times out."""
    return len(settings.GEMINI_MODELS) * settings.GEMINI_TIMEOUT


def generate(client, model, system_instruction, prompt):
    """Yield recommendation dicts as a single Gemini model streams them."""
    parser = llm.ItemStreamParser()
    try:
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=llm.RecommendationList,
            ),
        ):
            yield from parser.feed(chunk.text or "")
    except errors.APIError as error:
        llm.raise_api_error(PROVIDER, error.code, f"{model}: {error.message}")
    except httpx.HTTPError as error:
        llm.raise_api_error(PROVIDER, None, f"{model}: {error or type(error).__name__}")
