import logging
import time
from datetime import timedelta

from celery import shared_task
from django.conf import settings
from django.core.cache import cache
from django.utils import timezone

from app import ai_recommendations
from app.models import UserMessage
from app.providers import llm
from users.models import User

logger = logging.getLogger(__name__)

# Pause between LLM requests to stay under free tier rate limits
AI_RECOMMENDATIONS_DELAY = 5


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


def queue_ai_generation(user_id, media_type):
    """Queue a background generation of a recommendation pool once."""
    _queue_once(
        ai_recommendations.generation_queued_key(user_id, media_type),
        generate_ai_recommendations,
        user_id,
        media_type,
    )


def queue_ai_extension(user_id, media_type):
    """Queue a background extension of a recommendation pool once."""
    _queue_once(
        ai_recommendations.extension_queued_key(user_id, media_type),
        extend_ai_recommendations,
        user_id,
        media_type,
    )


def queue_ai_extension_if_needed(user_id, media_type, data):
    """Queue a pool extension when a generated pool is below its target size."""
    if data is not None and ai_recommendations.needs_extension(data):
        queue_ai_extension(user_id, media_type)


def _queue_once(queued_key, task, user_id, media_type):
    """Queue a pool task unless one is already queued or running."""
    if cache.add(queued_key, 1, ai_recommendations.lock_timeout()):
        task.delay(user_id, media_type)


@shared_task(name="Generate AI recommendations")
def generate_ai_recommendations(user_id, media_type):
    """Generate a user's AI recommendation pool and queue its extension."""
    try:
        user = User.objects.get(id=user_id)
        data = ai_recommendations.generate_global_recommendations(user, media_type)
    except ai_recommendations.RecommendationsInProgressError:
        return False
    # Failures are cached for the page, the next visit retries
    except Exception:
        logger.exception(
            "Failed to generate %s AI recommendations for user %s",
            media_type,
            user_id,
        )
        return False
    else:
        # Queued before the flag is cleared so pages keep polling
        queue_ai_extension_if_needed(user_id, media_type, data)
        return data is not None
    finally:
        cache.delete(ai_recommendations.generation_queued_key(user_id, media_type))


@shared_task(name="Extend AI recommendations")
def extend_ai_recommendations(user_id, media_type):
    """Add a batch to a user's AI recommendation pool and queue the next one."""
    queued_key = ai_recommendations.extension_queued_key(user_id, media_type)
    try:
        user = User.objects.get(id=user_id)
        extended = ai_recommendations.extend_global_recommendations(user, media_type)
    # Missing users or failed batches leave the pool for the next request
    except Exception:
        logger.exception(
            "Failed to extend %s AI recommendations for user %s",
            media_type,
            user_id,
        )
        extended = False

    data = cache.get(ai_recommendations.global_cache_key(user_id, media_type))
    if extended and data is not None and ai_recommendations.needs_extension(data):
        # Keep the queued flag so pages keep polling for the next batch
        cache.set(queued_key, 1, ai_recommendations.lock_timeout())
        extend_ai_recommendations.delay(user_id, media_type)
    else:
        cache.delete(queued_key)
    return extended


@shared_task(name="Refresh AI recommendations")
def refresh_ai_recommendations():
    """Regenerate the cached AI recommendations of every opted-in user."""
    if not llm.is_configured():
        logger.info(
            "Skipping AI recommendations refresh, no %s API key set.",
            llm.provider_label(),
        )
        return 0

    refreshed_count = 0
    is_first_request = True
    for user in User.objects.filter(ai_recommendations_enabled=True):
        for media_type in ai_recommendations.get_media_types(user):
            if not is_first_request:
                time.sleep(AI_RECOMMENDATIONS_DELAY)
            is_first_request = False
            try:
                # Keep the current pool until the new one is complete
                data = ai_recommendations.generate_global_recommendations(
                    user,
                    media_type,
                    progressive=False,
                )
            except ai_recommendations.RecommendationsInProgressError:
                logger.info(
                    "Skipped %s AI recommendations for %s, already generating",
                    media_type,
                    user.username,
                )
                continue
            # One failure shouldn't stop the refresh for the remaining users
            except Exception:
                logger.exception(
                    "Failed to refresh %s AI recommendations for %s",
                    media_type,
                    user.username,
                )
                continue
            queue_ai_extension_if_needed(user.id, media_type, data)
            refreshed_count += 1

    logger.info("Refreshed %s AI recommendation sets.", refreshed_count)

    return refreshed_count
