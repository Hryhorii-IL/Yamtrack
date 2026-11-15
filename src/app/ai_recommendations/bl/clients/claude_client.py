from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

import anthropic
import pydantic

from app.ai_recommendations.bl.calculations.stream_parsing import ItemStreamParser
from app.ai_recommendations.bl.clients.llm_client import LLMClient
from app.ai_recommendations.bl.clients.schema import RecommendationListSchema

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from anthropic.types import ThinkingConfigParam
    from django.core.cache.backends.base import BaseCache

    from app.ai_recommendations.bl.types.recommendation import RecommendedTitle

logger = logging.getLogger(__name__)

MAX_TOKENS: Final = 16000
# Fewer recommendations without thinking are retried with thinking
MIN_FAST_ITEMS: Final = 12


class ClaudeClient(LLMClient):
    """Streams recommendations from Claude models."""

    provider = "claude"
    label = "Claude"
    api_key_setting = "ANTHROPIC_API_KEY"
    models_setting = "CLAUDE_MODELS"
    timeout_setting = "CLAUDE_TIMEOUT"
    # Each model may run a second request with thinking
    requests_per_model = 2

    def __init__(
        self,
        *,
        api_key: str,
        models: Sequence[str],
        timeout: int,
        sdk: anthropic.Anthropic | None = None,
        cache: BaseCache | None = None,
    ) -> None:
        """Create the Anthropic SDK client unless a test passes one."""
        super().__init__(models=models, timeout=timeout, cache=cache)
        self.sdk = sdk or anthropic.Anthropic(
            api_key=api_key,
            timeout=timeout,
            # Fall back to the next model instead of retrying a failing one
            max_retries=0,
        )

    def generate(
        self,
        model: str,
        system_instruction: str,
        prompt: str,
        thinking: bool,  # noqa: FBT001 - forwarded positionally by the fallback
    ) -> Iterator[RecommendedTitle]:
        """Yield recommendations as a single Claude model streams them.

        Without ``thinking`` the first recommendation comes seconds sooner; a
        short or empty list is then retried once with thinking, and the repeated
        titles are dropped when resolving.
        """
        if thinking:
            yield from self._stream(model, system_instruction, prompt, thinking=True)
            return

        count = 0
        for thinking_retry in (False, True):
            if thinking_retry:
                if count >= MIN_FAST_ITEMS:
                    return
                logger.warning(
                    "Claude model %s returned %s recommendations, retrying with "
                    "thinking",
                    model,
                    count,
                )
            for recommendation in self._stream(
                model,
                system_instruction,
                prompt,
                thinking=thinking_retry,
            ):
                count += 1
                yield recommendation

    def _stream(
        self,
        model: str,
        system_instruction: str,
        prompt: str,
        *,
        thinking: bool,
    ) -> Iterator[RecommendedTitle]:
        """Yield the recommendations of a single streamed request."""
        parser = ItemStreamParser()
        thinking_config: ThinkingConfigParam = (
            {"type": "adaptive"} if thinking else {"type": "disabled"}
        )
        try:
            with self.sdk.messages.stream(
                model=model,
                max_tokens=MAX_TOKENS,
                system=system_instruction,
                messages=[{"role": "user", "content": prompt}],
                # Picking titles doesn't need deep reasoning, keep it fast and cheap
                output_config={"effort": "low"},
                output_format=RecommendationListSchema,
                thinking=thinking_config,
            ) as response:
                for text in response.text_stream:
                    yield from parser.feed(text)
        except anthropic.APIStatusError as error:
            raise self.error(error.status_code, f"{model}: {error.message}") from error
        except anthropic.APIConnectionError as error:
            details = f"{model}: {error or type(error).__name__}"
            raise self.error(None, details) from error
        # Raised for truncated or malformed JSON
        except pydantic.ValidationError as error:
            msg = f"{model}: invalid response"
            raise self.invalid(msg) from error
