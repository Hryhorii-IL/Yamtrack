from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Final

from django.core.cache import caches

from app.ai_recommendations.bl.errors import (
    LLMClientAuthError,
    LLMClientError,
    llm_error_class,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from django.core.cache.backends.base import BaseCache

    from app.ai_recommendations.bl.types.recommendation import RecommendedTitle

logger = logging.getLogger(__name__)

# How long a failed model is tried after the others
UNAVAILABLE_TIMEOUT: Final = 60 * 10


class LLMClient(ABC):
    """Streams recommendations from the first configured model that works."""

    provider: ClassVar[str]
    label: ClassVar[str]
    api_key_setting: ClassVar[str]
    models_setting: ClassVar[str]
    timeout_setting: ClassVar[str]
    # Requests a model may need for one recommendation list
    requests_per_model: ClassVar[int] = 1

    def __init__(
        self,
        *,
        models: Sequence[str],
        timeout: int,
        cache: BaseCache | None = None,
    ) -> None:
        """Keep the models in fallback order and the per-model timeout."""
        self.models = tuple(models)
        self.timeout = timeout
        self.cache = cache if cache is not None else caches["default"]

    def max_duration(self) -> int:
        """Return how long a request can take when every model times out."""
        return self.max_duration_of(self.models, self.timeout)

    @classmethod
    def max_duration_of(cls, models: Sequence[str], timeout: int) -> int:
        """Return how long a request can take when every model times out."""
        return cls.requests_per_model * len(models) * timeout

    def recommend(
        self,
        system_instruction: str,
        prompt: str,
        *,
        thinking: bool = False,
    ) -> Iterator[RecommendedTitle]:
        """Yield recommendations, falling back to the next model on failures.

        Falls back only while nothing has been yielded, since the caller may
        already be using the first model's recommendations.
        """
        request = (system_instruction, prompt, thinking)
        if not self.models:
            msg = f"No {self.label} models configured"
            raise self.invalid(msg)

        *fallback_models, last_model = self._models_in_order()
        for model in fallback_models:
            try:
                yield from self._generate_or_skip(model, *request)
            except LLMClientAuthError:
                # The API key is shared by all models, so retrying won't help
                raise
            except LLMClientError:
                logger.warning("%s model %s failed, trying the next", self.label, model)
            else:
                return

        yield from self._generate_or_skip(last_model, *request)

    @abstractmethod
    def generate(
        self,
        model: str,
        system_instruction: str,
        prompt: str,
        thinking: bool,  # noqa: FBT001 - forwarded positionally by the fallback
    ) -> Iterator[RecommendedTitle]:
        """Yield a single model's recommendations as they stream in."""

    def error(self, status_code: int | None, details: str) -> LLMClientError:
        """Return the error to raise for a failed request or connection."""
        logger.error("%s error: %s", self.label, details)
        return llm_error_class(status_code)(self.label, status_code, details)

    def invalid(self, details: str) -> LLMClientError:
        """Return the error of an unusable response, which retrying won't fix."""
        logger.error("%s error: %s", self.label, details)
        return LLMClientError(self.label, None, details)

    def unavailable_cache_key(self, model: str) -> str:
        """Return the cache key marking a model as recently failed."""
        return f"{self.provider}_unavailable_{model}"

    def _models_in_order(self) -> list[str]:
        """Return the models with recently failed ones moved last."""
        unavailable = self.cache.get_many(
            [self.unavailable_cache_key(model) for model in self.models],
        )
        return sorted(
            self.models,
            key=lambda model: self.unavailable_cache_key(model) in unavailable,
        )

    def _generate_or_skip(
        self,
        model: str,
        system_instruction: str,
        prompt: str,
        thinking: bool,  # noqa: FBT001 - part of the forwarded request
    ) -> Iterator[RecommendedTitle]:
        """Yield a model's recommendations and skip the model for a while on failure.

        A failure after some recommendations were yielded keeps them instead of
        raising, since the caller may already be using them.
        """
        yielded = False
        try:
            for recommendation in self.generate(
                model,
                system_instruction,
                prompt,
                thinking,
            ):
                yielded = True
                yield recommendation
            if not yielded:
                msg = f"{model}: invalid response"
                raise self.invalid(msg)
        except LLMClientError as error:
            if not isinstance(error, LLMClientAuthError):
                self.cache.set(
                    self.unavailable_cache_key(model),
                    1,
                    UNAVAILABLE_TIMEOUT,
                )
            if not yielded:
                raise
            logger.warning(
                "%s model %s failed mid-stream, keeping partial recommendations",
                self.label,
                model,
            )
