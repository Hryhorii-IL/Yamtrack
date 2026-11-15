from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING

from app import config
from app.ai_recommendations.bl.calculations.pool import needs_extension, with_batch
from app.ai_recommendations.bl.calculations.prompts import (
    global_prompt,
    system_instruction,
)
from app.ai_recommendations.bl.errors import (
    LLMClientError,
    LLMClientTransientError,
    PoolExtensionServiceError,
    PoolExtensionServiceTransientError,
    PoolExtensionServiceWrongStatusForOperationError,
)
from app.ai_recommendations.bl.services.step import step

if TYPE_CHECKING:
    from app.ai_recommendations.bl.clients.protocols import (
        RecommendationSource,
        TitleResolver,
    )
    from app.ai_recommendations.bl.stores.library_store import LibraryStore
    from app.ai_recommendations.bl.stores.pool_store import PoolStore


class PoolExtensionStatus(Enum):
    """Steps of PoolExtensionService, in the order they must run."""

    LOADED = auto()
    PLANNED = auto()
    COMMITTED = auto()
    FAILED = auto()


class PoolExtensionService:
    """Adds a batch of new recommendations to a saved pool."""

    def __init__(
        self,
        user_id: int,
        media_type: str,
        *,
        llm_client: RecommendationSource,
        search_client: TitleResolver,
        pool_store: PoolStore,
        library_store: LibraryStore,
    ) -> None:
        """Load the pool and the user's taste."""
        self.status = PoolExtensionStatus.LOADED
        self.user_id = user_id
        self.media_type = media_type
        self.llm_client = llm_client
        self.search_client = search_client
        self.pool_store = pool_store
        self.pool = pool_store.load_pool(user_id, media_type)
        self.taste = library_store.load_taste(user_id, media_type)
        self.provider_name = config.get_default_source_name(media_type).label
        self.system_instruction = ""
        self.prompt: str | None = None

    def _check_status(self, expected: PoolExtensionStatus) -> None:
        if self.status is not expected:
            msg = f"{expected.name} expected, {self.status.name} found"
            raise PoolExtensionServiceWrongStatusForOperationError(msg)

    @step(
        PoolExtensionStatus.LOADED,
        PoolExtensionStatus.PLANNED,
        PoolExtensionStatus.FAILED,
    )
    def plan(self) -> None:
        """Build the prompt of the next batch, unless the pool is done."""
        if self.pool is None or not needs_extension(self.pool):
            return
        self.system_instruction = system_instruction(self.provider_name)
        self.prompt = global_prompt(
            self.media_type,
            self.taste,
            recommended=[item.title for item in self.pool.items],
        )

    @step(
        PoolExtensionStatus.PLANNED,
        PoolExtensionStatus.COMMITTED,
        PoolExtensionStatus.FAILED,
    )
    def commit(self) -> bool:
        """Generate the batch and save the extended pool.

        Return whether the pool was extended.
        """
        if self.pool is None or self.prompt is None:
            return False

        recommendations = self.llm_client.recommend(
            self.system_instruction,
            self.prompt,
        )
        try:
            items = list(self.search_client.resolve(recommendations, self.media_type))
        except LLMClientTransientError as error:
            raise PoolExtensionServiceTransientError(str(error)) from error
        except LLMClientError as error:
            raise PoolExtensionServiceError(str(error)) from error

        # The pool may have been invalidated while generating
        if not self.pool_store.is_current(self.user_id, self.media_type, self.pool):
            return False
        self.pool_store.save_pool(
            self.user_id,
            self.media_type,
            with_batch(self.pool, items),
        )
        return True
