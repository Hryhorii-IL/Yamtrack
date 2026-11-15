"""Database builders for tests that run against the Django stores."""

from django.conf import settings
from django.contrib.auth import get_user_model

from app.ai_recommendations.bl.stores.django_cache_pool_store import (
    DjangoCachePoolStore,
)
from app.models import Item, MediaTypes, Movie, Sources, Status

LOCK_TIMEOUT = 60


def create_user(**kwargs):
    """Return a user with AI recommendations enabled."""
    return get_user_model().objects.create_user(
        username=kwargs.pop("username", "test"),
        ai_recommendations_enabled=True,
        **kwargs,
    )


def create_movie(user, media_id, title, score=None):
    """Return a tracked movie of the user."""
    # Django adds managers at runtime, ty runs without django-stubs
    item = Item.objects.create(  # ty: ignore[missing-attribute]
        media_id=media_id,
        source=Sources.TMDB.value,
        media_type=MediaTypes.MOVIE.value,
        title=title,
        image=settings.IMG_NONE,
    )
    return Movie.objects.create(  # ty: ignore[missing-attribute]
        item=item,
        user=user,
        score=score,
        status=Status.PLANNING.value,
    )


def pool_store():
    """Return the Django cache pool store with a short lock timeout."""
    return DjangoCachePoolStore(lock_timeout=LOCK_TIMEOUT)
