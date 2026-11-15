from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING

from app import config
from app.ai_recommendations.bl.calculations.pool import without_media
from app.ai_recommendations.bl.calculations.prompts import (
    media_prompt,
    system_instruction,
)
from app.ai_recommendations.bl.errors import (
    LLMClientError,
    LLMClientTransientError,
    MediaRecommendationsServiceError,
    MediaRecommendationsServiceNotFoundError,
    MediaRecommendationsServiceTransientError,
    MediaRecommendationsServiceWrongStatusForOperationError,
)
from app.ai_recommendations.bl.services.step import step
from app.ai_recommendations.bl.types.pool import MediaRecommendations
from app.ai_recommendations.bl.types.recommendation import MediaDetails
from app.providers import services

if TYPE_CHECKING:
    from collections.abc import Callable

    from app.ai_recommendations.bl.clients.protocols import (
        RecommendationSource,
        TitleResolver,
    )
    from app.ai_recommendations.bl.stores.library_store import LibraryStore
    from app.ai_recommendations.bl.stores.pool_store import PoolStore
    from app.ai_recommendations.bl.types.clock import Clock

type FetchMetadata = Callable[[str, str, str], dict]


class MediaRecommendationsStatus(Enum):
    """Steps of MediaRecommendationsService, in the order they must run."""

    LOADED = auto()
    PLANNED = auto()
    COMMITTED = auto()
    FAILED = auto()


class MediaRecommendationsService:
    """Generates recommendations similar to a single media."""

    def __init__(  # noqa: PLR0913 - the media identity plus injected collaborators
        self,
        user_id: int,
        media_type: str,
        source: str,
        media_id: str,
        *,
        thinking: bool,
        llm_client: RecommendationSource,
        search_client: TitleResolver,
        pool_store: PoolStore,
        library_store: LibraryStore,
        clock: Clock,
        fetch_metadata: FetchMetadata | None = None,
    ) -> None:
        """Load the viewed media and the user's taste."""
        self.status = MediaRecommendationsStatus.LOADED
        self.user_id = user_id
        self.media_type = media_type
        self.thinking = thinking
        self.llm_client = llm_client
        self.search_client = search_client
        self.pool_store = pool_store
        self.now = clock.now()
        fetch_metadata = fetch_metadata or services.get_media_metadata
        try:
            metadata = fetch_metadata(media_type, media_id, source)
        except services.ProviderAPIError as error:
            raise MediaRecommendationsServiceNotFoundError(str(error)) from error
        self.details = MediaDetails(
            media_id=media_id,
            source=source,
            title=metadata["title"],
            genres=tuple(metadata.get("genres") or ()),
            synopsis=metadata.get("synopsis") or "",
            user_score=library_store.load_user_score(
                user_id,
                media_type,
                source,
                media_id,
            ),
        )
        self.taste = library_store.load_taste(user_id, media_type)
        self.provider_name = config.get_default_source_name(media_type).label
        self.system_instruction = ""
        self.prompt = ""

    def _check_status(self, expected: MediaRecommendationsStatus) -> None:
        if self.status is not expected:
            msg = f"{expected.name} expected, {self.status.name} found"
            raise MediaRecommendationsServiceWrongStatusForOperationError(msg)

    @step(
        MediaRecommendationsStatus.LOADED,
        MediaRecommendationsStatus.PLANNED,
        MediaRecommendationsStatus.FAILED,
    )
    def plan(self) -> None:
        """Build the prompt describing the viewed media and the user's taste."""
        self.system_instruction = system_instruction(self.provider_name)
        self.prompt = media_prompt(self.media_type, self.details, self.taste)

    @step(
        MediaRecommendationsStatus.PLANNED,
        MediaRecommendationsStatus.COMMITTED,
        MediaRecommendationsStatus.FAILED,
    )
    def commit(self) -> MediaRecommendations:
        """Generate the recommendations and save them."""
        recommendations = self.llm_client.recommend(
            self.system_instruction,
            self.prompt,
            thinking=self.thinking,
        )
        try:
            items = without_media(
                self.search_client.resolve(recommendations, self.media_type),
                self.details.source,
                self.details.media_id,
            )
        except LLMClientTransientError as error:
            raise MediaRecommendationsServiceTransientError(str(error)) from error
        except LLMClientError as error:
            raise MediaRecommendationsServiceError(str(error)) from error

        result = MediaRecommendations(generated_at=self.now, items=items)
        self.pool_store.save_media(
            self.user_id,
            self.media_type,
            self.details.source,
            self.details.media_id,
            result,
        )
        return result
