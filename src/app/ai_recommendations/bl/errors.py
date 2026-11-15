from __future__ import annotations

from typing import ClassVar, Final

from app.ai_recommendations.bl.types.result import ErrorCode

TRANSIENT_STATUS_CODES: Final = (408, 429)


class PoolGenerationServiceError(Exception):
    """Base error of PoolGenerationService."""

    code: ClassVar[ErrorCode] = ErrorCode.INVALID


class PoolGenerationServiceWrongStatusForOperationError(PoolGenerationServiceError):
    """A PoolGenerationService method was called out of order: a bug."""


class PoolGenerationServiceNoResultsError(PoolGenerationServiceError):
    """None of the recommended titles could be found."""


class PoolGenerationServiceTransientError(PoolGenerationServiceError):
    """The LLM is temporarily unavailable; the job should retry."""

    code = ErrorCode.UNAVAILABLE


class PoolExtensionServiceError(Exception):
    """Base error of PoolExtensionService."""

    code: ClassVar[ErrorCode] = ErrorCode.INVALID


class PoolExtensionServiceWrongStatusForOperationError(PoolExtensionServiceError):
    """A PoolExtensionService method was called out of order: a bug."""


class PoolExtensionServiceTransientError(PoolExtensionServiceError):
    """The LLM is temporarily unavailable; the job should retry."""

    code = ErrorCode.UNAVAILABLE


class MediaRecommendationsServiceError(Exception):
    """Base error of MediaRecommendationsService."""

    code: ClassVar[ErrorCode] = ErrorCode.INVALID


class MediaRecommendationsServiceWrongStatusForOperationError(
    MediaRecommendationsServiceError,
):
    """A MediaRecommendationsService method was called out of order: a bug."""


class MediaRecommendationsServiceNotFoundError(MediaRecommendationsServiceError):
    """The viewed media couldn't be loaded from its provider."""

    code = ErrorCode.NOT_FOUND


class MediaRecommendationsServiceTransientError(MediaRecommendationsServiceError):
    """The LLM is temporarily unavailable."""

    code = ErrorCode.UNAVAILABLE


class PoolStoreError(Exception):
    """Base error of the pool store."""

    code: ClassVar[ErrorCode] = ErrorCode.INVALID


class PoolStoreLockError(PoolStoreError):
    """Another request or task holds the generation lock."""

    code = ErrorCode.CONFLICT


class LibraryStoreError(Exception):
    """Base error of the library store."""

    code: ClassVar[ErrorCode] = ErrorCode.INVALID


class LibraryStoreNotFoundError(LibraryStoreError):
    """The user doesn't exist anymore."""

    code = ErrorCode.NOT_FOUND


class LLMClientError(Exception):
    """An LLM request failed; the message is safe to show to the user."""

    def __init__(self, label: str, status_code: int | None, details: str) -> None:
        """Build the same message provider errors show elsewhere in the app."""
        self.status_code = status_code
        message = f"There was an error contacting the {label} API"
        message += (
            " (network error)" if status_code is None else f" (HTTP {status_code})"
        )
        message += f": {details}. Check the logs for more details."
        super().__init__(message)


class LLMClientTransientError(LLMClientError):
    """Network errors, rate limits and server errors, worth retrying later."""


class LLMClientAuthError(LLMClientError):
    """The API key was rejected, so other models won't work either."""


def llm_error_class(status_code: int | None) -> type[LLMClientError]:
    """Return the LLM error class of a failed request's status code."""
    if status_code in (401, 403):
        return LLMClientAuthError
    if (
        status_code is None
        or status_code in TRANSIENT_STATUS_CODES
        or status_code >= 500  # noqa: PLR2004 - server errors
    ):
        return LLMClientTransientError
    return LLMClientError
