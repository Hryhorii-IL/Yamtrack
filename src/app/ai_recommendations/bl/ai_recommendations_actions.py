from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.ai_recommendations.bl.calculations.pool import needs_extension
from app.ai_recommendations.bl.calculations.thinking import uses_thinking
from app.ai_recommendations.bl.clients import (
    make_llm_client,
    make_media_search_client,
    thinking_mode,
)
from app.ai_recommendations.bl.errors import (
    LibraryStoreError,
    MediaRecommendationsServiceError,
    MediaRecommendationsServiceTransientError,
    PoolExtensionServiceError,
    PoolExtensionServiceTransientError,
    PoolGenerationServiceError,
    PoolGenerationServiceTransientError,
    PoolStoreError,
    PoolStoreLockError,
)
from app.ai_recommendations.bl.services.media_recommendations_service import (
    MediaRecommendationsService,
)
from app.ai_recommendations.bl.services.pool_extension_service import (
    PoolExtensionService,
)
from app.ai_recommendations.bl.services.pool_generation_service import (
    PoolGenerationService,
)
from app.ai_recommendations.bl.stores import make_library_store, make_pool_store
from app.ai_recommendations.bl.types.clock import SystemClock
from app.ai_recommendations.bl.types.result import ErrorCode, Result

if TYPE_CHECKING:
    from collections.abc import Callable

    from app.ai_recommendations.bl.clients.protocols import (
        RecommendationSource,
        TitleResolver,
    )
    from app.ai_recommendations.bl.stores.library_store import LibraryStore
    from app.ai_recommendations.bl.stores.pool_store import PoolStore
    from app.ai_recommendations.bl.types.clock import Clock
    from app.ai_recommendations.bl.types.pool import (
        MediaRecommendations,
        QueueKind,
        RecommendationPool,
    )

    type QueueTask = Callable[[int, str], None]

logger = logging.getLogger(__name__)


class AIRecommendationsActions:
    """The only entry point of AI recommendation writes for views and tasks.

    Collaborators default to the real ones; tests pass fakes.
    """

    @staticmethod
    def pool(
        user_id: int,
        media_type: str,
        *,
        force: bool,
        queue_generation: QueueTask,
        queue_extension: QueueTask,
        pool_store: PoolStore | None = None,
    ) -> Result[RecommendationPool]:
        """Return the pool of a media type, queueing its generation if missing.

        The value is None while the pool is generated. The queue callables
        start Celery tasks, which this layer can't import.
        """
        store = pool_store or make_pool_store()
        with store.transaction():
            if force:
                store.delete_pool(user_id, media_type)
            pool = store.load_pool(user_id, media_type)
            if pool is None:
                store.on_commit(lambda: queue_generation(user_id, media_type))
            elif needs_extension(pool):
                store.on_commit(lambda: queue_extension(user_id, media_type))
        # Eager Celery generates inline once the transaction is over
        return Result(value=pool or store.load_pool(user_id, media_type))

    @staticmethod
    def generate_pool(  # noqa: PLR0913 - the pool identity plus collaborators
        user_id: int,
        media_type: str,
        *,
        progressive: bool = True,
        refresh: bool = False,
        last_attempt: bool = True,
        llm_client: RecommendationSource | None = None,
        search_client: TitleResolver | None = None,
        pool_store: PoolStore | None = None,
        library_store: LibraryStore | None = None,
        clock: Clock | None = None,
    ) -> Result[RecommendationPool]:
        """Generate the pool of a media type, continuing a partial one.

        Raises PoolGenerationServiceTransientError so the job retries, unless
        it's the ``last_attempt``. The value is None when the pool was
        invalidated meanwhile.
        """
        store = pool_store or make_pool_store()
        try:
            with store.transaction():
                store.lock_pool(user_id, media_type)
                service = PoolGenerationService(
                    user_id,
                    media_type,
                    progressive=progressive,
                    refresh=refresh,
                    last_attempt=last_attempt,
                    # Progressive pools are watched card by card, refreshes aren't
                    thinking=uses_thinking(thinking_mode(), background=not progressive),
                    llm_client=llm_client or make_llm_client(),
                    search_client=search_client or make_media_search_client(),
                    pool_store=store,
                    library_store=library_store or make_library_store(),
                    clock=clock or SystemClock(),
                )
                service.plan()
                pool = service.commit()
        except PoolGenerationServiceTransientError:
            raise
        except PoolStoreLockError as error:
            logger.info(
                "%s AI pool of user %s is generating already", media_type, user_id
            )
            return Result(error_code=error.code)
        except (PoolGenerationServiceError, PoolStoreError, LibraryStoreError) as error:
            logger.exception(
                "%s AI pool of user %s is not generated.", media_type, user_id
            )
            return Result(error_code=error.code)
        return Result(value=pool)

    @staticmethod
    def extend_pool(
        user_id: int,
        media_type: str,
        *,
        llm_client: RecommendationSource | None = None,
        search_client: TitleResolver | None = None,
        pool_store: PoolStore | None = None,
        library_store: LibraryStore | None = None,
    ) -> Result[bool]:
        """Add a batch to the pool; the value tells whether it was extended.

        Raises PoolExtensionServiceTransientError so the job retries.
        """
        store = pool_store or make_pool_store()
        try:
            with store.transaction():
                store.lock_pool(user_id, media_type)
                service = PoolExtensionService(
                    user_id,
                    media_type,
                    thinking=uses_thinking(thinking_mode(), background=True),
                    llm_client=llm_client or make_llm_client(),
                    search_client=search_client or make_media_search_client(),
                    pool_store=store,
                    library_store=library_store or make_library_store(),
                )
                service.plan()
                extended = service.commit()
        except PoolExtensionServiceTransientError:
            raise
        except PoolStoreLockError as error:
            logger.info(
                "%s AI pool of user %s is generating already", media_type, user_id
            )
            return Result(error_code=error.code)
        except (PoolExtensionServiceError, PoolStoreError, LibraryStoreError) as error:
            logger.exception(
                "%s AI pool of user %s is not extended.", media_type, user_id
            )
            return Result(error_code=error.code)
        return Result(value=extended)

    @staticmethod
    def media_recommendations(  # noqa: PLR0913 - the media identity plus collaborators
        user_id: int,
        media_type: str,
        source: str,
        media_id: str,
        *,
        force: bool,
        llm_client: RecommendationSource | None = None,
        search_client: TitleResolver | None = None,
        pool_store: PoolStore | None = None,
        library_store: LibraryStore | None = None,
        clock: Clock | None = None,
    ) -> Result[MediaRecommendations]:
        """Return cached or newly generated recommendations similar to a media.

        Runs inside an HTTP request, where nothing retries, so transient
        errors come back as UNAVAILABLE instead of being raised.
        """
        store = pool_store or make_pool_store()
        if not force:
            cached = store.load_media(user_id, media_type, source, media_id)
            if cached is not None:
                return Result(value=cached)

        try:
            with store.transaction():
                store.lock_media(user_id, media_type, source, media_id)
                service = MediaRecommendationsService(
                    user_id,
                    media_type,
                    source,
                    media_id,
                    thinking=uses_thinking(thinking_mode(), background=False),
                    llm_client=llm_client or make_llm_client(),
                    search_client=search_client or make_media_search_client(),
                    pool_store=store,
                    library_store=library_store or make_library_store(),
                    clock=clock or SystemClock(),
                )
                service.plan()
                recommendations = service.commit()
        except PoolStoreLockError:
            # Another request is generating them; serve the old ones meanwhile
            cached = (
                None
                if force
                else store.load_media(user_id, media_type, source, media_id)
            )
            if cached is not None:
                return Result(value=cached)
            return Result(error_code=ErrorCode.CONFLICT)
        except MediaRecommendationsServiceTransientError as error:
            logger.warning("AI recommendations for %s %s: %s", source, media_id, error)
            return Result(error_code=error.code)
        except (MediaRecommendationsServiceError, LibraryStoreError) as error:
            logger.exception(
                "AI recommendations for %s %s of user %s are not generated.",
                source,
                media_id,
                user_id,
            )
            return Result(error_code=error.code)
        return Result(value=recommendations)

    @staticmethod
    def mark_queued(
        kind: QueueKind,
        user_id: int,
        media_type: str,
        *,
        pool_store: PoolStore | None = None,
    ) -> bool:
        """Mark a pool task as queued, returning False if it already was."""
        return (pool_store or make_pool_store()).mark_queued(kind, user_id, media_type)

    @staticmethod
    def keep_queued(
        kind: QueueKind,
        user_id: int,
        media_type: str,
        *,
        pool_store: PoolStore | None = None,
    ) -> None:
        """Keep a pool task marked as queued while it's retried or chained."""
        (pool_store or make_pool_store()).keep_queued(kind, user_id, media_type)

    @staticmethod
    def clear_queued(
        kind: QueueKind,
        user_id: int,
        media_type: str,
        *,
        pool_store: PoolStore | None = None,
    ) -> None:
        """Mark a pool task as no longer queued."""
        (pool_store or make_pool_store()).clear_queued(kind, user_id, media_type)

    @staticmethod
    def invalidate_user(user_id: int, *, pool_store: PoolStore | None = None) -> None:
        """Drop every cached recommendation of the user."""
        (pool_store or make_pool_store()).invalidate_user(user_id)

    @staticmethod
    def invalidate_media_type(
        user_id: int,
        media_type: str,
        *,
        pool_store: PoolStore | None = None,
    ) -> None:
        """Drop the user's cached recommendations of a media type."""
        (pool_store or make_pool_store()).invalidate_media_type(user_id, media_type)
