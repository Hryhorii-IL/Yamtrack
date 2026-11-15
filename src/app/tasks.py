import logging
from datetime import timedelta

from celery import shared_task
from django.conf import settings
from django.utils import timezone

from app.ai_recommendations.bl.ai_recommendations_actions import (
    AIRecommendationsActions,
)
from app.ai_recommendations.bl.errors import (
    PoolExtensionServiceTransientError,
    PoolGenerationServiceTransientError,
)
from app.ai_recommendations.bl.queries.media_types import get_media_types
from app.ai_recommendations.bl.queries.pool import pool_needs_extension
from app.ai_recommendations.bl.queries.provider import get_provider_info
from app.ai_recommendations.bl.types.pool import QueueKind
from app.models import UserMessage
from users.models import User

logger = logging.getLogger(__name__)

# Seconds between queued refreshes, to stay under free tier rate limits
AI_RECOMMENDATIONS_DELAY = 5
# Retries of AI tasks hitting transient LLM errors, waiting 10, 20 then 40s
AI_MAX_RETRIES = 3
AI_RETRY_DELAY = 10


@shared_task(name="Cleanup user messages")
def cleanup_user_messages():
    """Delete shown user messages older than the configured retention window."""
    cutoff = timezone.now() - timedelta(days=settings.USER_MESSAGE_RETENTION_DAYS)
    deleted_count, _ = UserMessage.objects.filter(
        shown_at__isnull=False,
        shown_at__lt=cutoff,
    ).delete()

    logger.info("Deleted %s old shown user messages.", deleted_count)

    return deleted_count


def queue_ai_generation(user_id, media_type, *, refresh=False, countdown=0):
    """Queue a background generation of a recommendation pool once."""
    if AIRecommendationsActions.mark_queued(QueueKind.GENERATION, user_id, media_type):
        generate_ai_recommendations.apply_async(
            (user_id, media_type),
            {"refresh": refresh},
            countdown=countdown,
        )


def queue_ai_extension(user_id, media_type):
    """Queue a background extension of a recommendation pool once."""
    if AIRecommendationsActions.mark_queued(QueueKind.EXTENSION, user_id, media_type):
        extend_ai_recommendations.delay(user_id, media_type)


@shared_task(
    name="Generate AI recommendations",
    bind=True,
    autoretry_for=(PoolGenerationServiceTransientError,),
    max_retries=AI_MAX_RETRIES,
    retry_backoff=AI_RETRY_DELAY,
    retry_jitter=False,
)
def generate_ai_recommendations(self, user_id, media_type, *, refresh=False):
    """Generate a user's AI recommendation pool and queue its extension.

    Refreshes keep the current pool until the new one is complete; other
    generations show it card by card.
    """
    keep_queued = False
    try:
        result = AIRecommendationsActions.generate_pool(
            user_id,
            media_type,
            progressive=not refresh,
            refresh=refresh,
            last_attempt=self.request.retries >= self.max_retries,
        )
    except PoolGenerationServiceTransientError:
        # Pages keep polling the partial pool while the generation is retried
        keep_queued = True
        AIRecommendationsActions.keep_queued(QueueKind.GENERATION, user_id, media_type)
        raise
    else:
        # Queued before the flag is cleared so pages keep polling
        if result.value is not None and pool_needs_extension(user_id, media_type):
            queue_ai_extension(user_id, media_type)
        return result.ok
    finally:
        if not keep_queued:
            AIRecommendationsActions.clear_queued(
                QueueKind.GENERATION,
                user_id,
                media_type,
            )


@shared_task(
    name="Extend AI recommendations",
    bind=True,
    autoretry_for=(PoolExtensionServiceTransientError,),
    max_retries=AI_MAX_RETRIES,
    retry_backoff=AI_RETRY_DELAY,
    retry_jitter=False,
)
def extend_ai_recommendations(self, user_id, media_type):
    """Add a batch to a user's AI recommendation pool and queue the next one."""
    keep_queued = False
    try:
        result = AIRecommendationsActions.extend_pool(user_id, media_type)
        if result.value and pool_needs_extension(user_id, media_type):
            # Keep the queued flag so pages keep polling for the next batch
            keep_queued = True
            AIRecommendationsActions.keep_queued(
                QueueKind.EXTENSION,
                user_id,
                media_type,
            )
            extend_ai_recommendations.delay(user_id, media_type)
        return bool(result.value)
    except PoolExtensionServiceTransientError:
        keep_queued = self.request.retries < self.max_retries
        if keep_queued:
            AIRecommendationsActions.keep_queued(
                QueueKind.EXTENSION,
                user_id,
                media_type,
            )
        raise
    finally:
        if not keep_queued:
            AIRecommendationsActions.clear_queued(
                QueueKind.EXTENSION,
                user_id,
                media_type,
            )


@shared_task(name="Refresh AI recommendations")
def refresh_ai_recommendations():
    """Queue a refresh of the AI recommendations of every opted-in user."""
    provider = get_provider_info()
    if not provider.configured:
        logger.info(
            "Skipping AI recommendations refresh, no %s API key set.",
            provider.label,
        )
        return 0

    queued_count = 0
    for user in User.objects.filter(ai_recommendations_enabled=True):
        for media_type in get_media_types(user.id):
            # Spread the LLM requests to stay under free tier rate limits
            queue_ai_generation(
                user.id,
                media_type,
                refresh=True,
                countdown=queued_count * AI_RECOMMENDATIONS_DELAY,
            )
            queued_count += 1

    logger.info("Queued %s AI recommendation refreshes.", queued_count)

    return queued_count
