from __future__ import annotations

import dataclasses
from enum import Enum, auto
from typing import TYPE_CHECKING

from app import config
from app.ai_recommendations.bl.calculations.pool import with_item
from app.ai_recommendations.bl.calculations.prompts import (
    global_prompt,
    system_instruction,
)
from app.ai_recommendations.bl.errors import (
    LLMClientError,
    LLMClientTransientError,
    PoolGenerationServiceError,
    PoolGenerationServiceNoResultsError,
    PoolGenerationServiceTransientError,
    PoolGenerationServiceWrongStatusForOperationError,
)
from app.ai_recommendations.bl.services.step import step
from app.ai_recommendations.bl.types.pool import RecommendationPool

if TYPE_CHECKING:
    from app.ai_recommendations.bl.clients.protocols import (
        RecommendationSource,
        TitleResolver,
    )
    from app.ai_recommendations.bl.stores.library_store import LibraryStore
    from app.ai_recommendations.bl.stores.pool_store import PoolStore
    from app.ai_recommendations.bl.types.clock import Clock


class PoolGenerationStatus(Enum):
    """Steps of PoolGenerationService, in the order they must run."""

    LOADED = auto()
    PLANNED = auto()
    COMMITTED = auto()
    FAILED = auto()


class PoolGenerationService:
    """Generates the recommendation pool of a media type.

    With ``progressive`` the pool is saved item by item while the LLM streams,
    so pages show cards before the generation ends. A transient LLM failure
    leaves the partial pool for the retried job to continue; a permanent one,
    or the job's last attempt, replaces it with the error. Without
    ``progressive`` the current pool is kept until the new one is complete,
    which suits background refreshes.
    """

    def __init__(  # noqa: PLR0913 - the pool identity plus injected collaborators
        self,
        user_id: int,
        media_type: str,
        *,
        progressive: bool,
        refresh: bool,
        last_attempt: bool,
        llm_client: RecommendationSource,
        search_client: TitleResolver,
        pool_store: PoolStore,
        library_store: LibraryStore,
        clock: Clock,
    ) -> None:
        """Load the user's taste and the pool the generation continues or replaces."""
        self.status = PoolGenerationStatus.LOADED
        self.user_id = user_id
        self.media_type = media_type
        self.progressive = progressive
        self.refresh = refresh
        self.last_attempt = last_attempt
        self.llm_client = llm_client
        self.search_client = search_client
        self.pool_store = pool_store
        self.now = clock.now()
        self.taste = library_store.load_taste(user_id, media_type)
        self.provider_name = config.get_default_source_name(media_type).label
        self.current = pool_store.load_pool(user_id, media_type)
        self.pool: RecommendationPool | None = None
        self.system_instruction = ""
        self.prompt = ""

    def _check_status(self, expected: PoolGenerationStatus) -> None:
        if self.status is not expected:
            msg = f"{expected.name} expected, {self.status.name} found"
            raise PoolGenerationServiceWrongStatusForOperationError(msg)

    @step(
        PoolGenerationStatus.LOADED,
        PoolGenerationStatus.PLANNED,
        PoolGenerationStatus.FAILED,
    )
    def plan(self) -> None:
        """Decide whether to start, continue or skip the pool, and build the prompt."""
        current = self.current
        if current is not None and not current.partial and not self.refresh:
            # Already generated, e.g. a duplicate job: nothing to do
            return
        resuming = self.progressive and current is not None and current.partial
        self.pool = (
            current
            if resuming
            else RecommendationPool(generated_at=self.now, partial=self.progressive)
        )
        self.system_instruction = system_instruction(self.provider_name)
        self.prompt = global_prompt(
            self.media_type,
            self.taste,
            recommended=[item.title for item in self.pool.items],
        )

    @step(
        PoolGenerationStatus.PLANNED,
        PoolGenerationStatus.COMMITTED,
        PoolGenerationStatus.FAILED,
    )
    def commit(self) -> RecommendationPool | None:
        """Generate the pool and save it.

        Return the pool, or None when it was invalidated meanwhile.
        """
        if self.pool is None:
            return self.current
        start = self.pool
        try:
            if not self._fill():
                return None
        except LLMClientTransientError as error:
            if not self.last_attempt:
                if self.progressive and self._is_current(start):
                    # Keep the partial pool alive for the retry to continue
                    self._save(partial=True)
                raise PoolGenerationServiceTransientError(str(error)) from error
            self._fail(start, str(error))
            raise PoolGenerationServiceError(str(error)) from error
        except LLMClientError as error:
            self._fail(start, str(error))
            raise PoolGenerationServiceError(str(error)) from error
        except PoolGenerationServiceError as error:
            self._fail(start, str(error))
            raise
        except Exception:
            self._fail(start, "Failed to generate AI recommendations.")
            raise

        expected = start if self.progressive else self.current
        if not self._is_current(expected):
            return None
        self.pool = dataclasses.replace(self.pool, partial=False)
        self._save(partial=False)
        return self.pool

    def _fill(self) -> bool:
        """Add the resolved recommendations, saving them as they arrive.

        Return False when a progressive pool was invalidated meanwhile.
        """
        assert self.pool is not None  # noqa: S101 - set by plan()
        if self.progressive:
            self._save(partial=True)
        known = {item.key for item in self.pool.items}
        recommendations = self.llm_client.recommend(
            self.system_instruction,
            self.prompt,
        )
        for item in self.search_client.resolve(recommendations, self.media_type):
            if self.progressive and not self._is_current(self.pool):
                return False
            if item.key in known:
                continue
            known.add(item.key)
            self.pool = with_item(self.pool, item)
            if self.progressive:
                self._save(partial=True)
        if not self.pool.items:
            msg = "None of the recommended titles could be found."
            raise PoolGenerationServiceNoResultsError(msg)
        return True

    def _save(self, *, partial: bool) -> None:
        assert self.pool is not None  # noqa: S101 - set by plan()
        if partial:
            self.pool_store.save_partial_pool(self.user_id, self.media_type, self.pool)
        else:
            self.pool_store.save_pool(self.user_id, self.media_type, self.pool)

    def _is_current(self, pool: RecommendationPool | None) -> bool:
        return self.pool_store.is_current(self.user_id, self.media_type, pool)

    def _fail(self, start: RecommendationPool, message: str) -> None:
        """Show the error on pages, unless the pool was invalidated meanwhile."""
        if not self.progressive or not self._is_current(start):
            return
        assert self.pool is not None  # noqa: S101 - set by plan()
        self.pool_store.save_failed_pool(
            self.user_id,
            self.media_type,
            dataclasses.replace(self.pool, partial=False, error=message),
        )
