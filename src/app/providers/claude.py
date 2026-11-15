import functools
import logging

import anthropic
import pydantic
from django.conf import settings

from app.providers import llm

logger = logging.getLogger(__name__)

PROVIDER = "claude"
MAX_TOKENS = 16000
# Fewer recommendations without thinking are retried with thinking
MIN_FAST_ITEMS = 12


def recommend(system_instruction, prompt):
    """Yield recommendation dicts from the first configured model that works."""
    client = anthropic.Anthropic(
        api_key=settings.ANTHROPIC_API_KEY,
        timeout=settings.CLAUDE_TIMEOUT,
        # Fall back to the next model instead of retrying a failing one
        max_retries=0,
    )
    return llm.recommend(
        PROVIDER,
        settings.CLAUDE_MODELS,
        functools.partial(generate, client),
        system_instruction,
        prompt,
    )


def max_duration():
    """Return how long a request can take when every model times out."""
    # Each model may run a second request with thinking
    return 2 * len(settings.CLAUDE_MODELS) * settings.CLAUDE_TIMEOUT


def generate(client, model, system_instruction, prompt):
    """Yield recommendation dicts as a single Claude model streams them.

    Thinking is disabled since it delays the first recommendation by several
    seconds; a short or empty list is retried once with thinking, and the
    repeated titles are dropped when resolving.
    """
    count = 0
    for thinking in (False, True):
        if thinking:
            if count >= MIN_FAST_ITEMS:
                return
            logger.warning(
                "Claude model %s returned %s recommendations, retrying with thinking",
                model,
                count,
            )
        for recommendation in stream(
            client,
            model,
            system_instruction,
            prompt,
            thinking=thinking,
        ):
            count += 1
            yield recommendation


def stream(client, model, system_instruction, prompt, *, thinking):
    """Yield the recommendations of a single streamed Claude request."""
    parser = llm.ItemStreamParser()
    try:
        with client.messages.stream(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system_instruction,
            messages=[{"role": "user", "content": prompt}],
            # Picking titles doesn't need deep reasoning, keep it fast and cheap
            output_config={"effort": "low"},
            output_format=llm.RecommendationList,
            thinking={"type": "adaptive" if thinking else "disabled"},
        ) as response:
            for text in response.text_stream:
                yield from parser.feed(text)
    except anthropic.APIStatusError as error:
        llm.raise_api_error(PROVIDER, error.status_code, f"{model}: {error.message}")
    except anthropic.APIConnectionError as error:
        llm.raise_api_error(PROVIDER, None, f"{model}: {error or type(error).__name__}")
    # Raised for truncated or malformed JSON
    except pydantic.ValidationError:
        llm.raise_api_error(PROVIDER, None, f"{model}: invalid response")
