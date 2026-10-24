import logging
import time
from datetime import timedelta

from celery import shared_task
from django.conf import settings
from django.utils import timezone

from app import ai_recommendations
from app.models import UserMessage
from users.models import User

logger = logging.getLogger(__name__)

# Pause between Gemini requests to stay under free tier rate limits
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


@shared_task(name="Refresh AI recommendations")
def refresh_ai_recommendations():
    """Regenerate the cached AI recommendations of every opted-in user."""
    if not settings.GEMINI_API_KEY:
        logger.info("Skipping AI recommendations refresh, no Gemini API key set.")
        return 0

    refreshed_count = 0
    is_first_request = True
    for user in User.objects.filter(ai_recommendations_enabled=True):
        for media_type in ai_recommendations.get_media_types(user):
            if not is_first_request:
                time.sleep(AI_RECOMMENDATIONS_DELAY)
            is_first_request = False
            try:
                ai_recommendations.global_recommendations(user, media_type, force=True)
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
            refreshed_count += 1

    logger.info("Refreshed %s AI recommendation sets.", refreshed_count)

    return refreshed_count
