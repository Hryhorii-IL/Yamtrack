from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from django.apps import apps
from django.contrib.auth import get_user_model

from app.ai_recommendations.bl.errors import LibraryStoreNotFoundError
from app.ai_recommendations.bl.types.constants import LIBRARY_LIMIT, MEDIA_TYPES
from app.ai_recommendations.bl.types.taste import RatedTitle, TasteInput, TasteSettings

if TYPE_CHECKING:
    from users.models import User


class DjangoLibraryStore:
    """Reads the user's library and AI preferences through the Django ORM."""

    def load_enabled_media_types(self, user_id: int) -> tuple[str, ...]:
        """Return the AI media types enabled in the user's preferences."""
        enabled = self._user(user_id).get_enabled_media_types()
        return tuple(media_type for media_type in MEDIA_TYPES if media_type in enabled)

    def has_media(self, user_id: int, media_type: str) -> bool:
        """Return whether the user tracks anything of a media type."""
        return _model(media_type).objects.filter(user_id=user_id).exists()

    def load_taste(self, user_id: int, media_type: str) -> TasteInput:
        """Return the ratings, library and wishes describing the user's taste."""
        user = self._user(user_id)
        settings = TasteSettings(
            include_liked=cast("bool", user.ai_include_liked),
            liked_min_score=cast("int", user.ai_liked_min_score),
            include_disliked=cast("bool", user.ai_include_disliked),
            disliked_max_score=cast("int", user.ai_disliked_max_score),
        )
        rated: tuple[RatedTitle, ...] = ()
        # Ratings of every media type are signals of overall taste
        if settings.include_liked or settings.include_disliked:
            rated = tuple(
                RatedTitle(
                    item_id=media.item_id,
                    title=media.item.title,
                    media_type=media.item.media_type,
                    score=float(media.score),
                    created_at=media.created_at,
                )
                for rated_type in self.load_enabled_media_types(user_id)
                for media in _model(rated_type)
                .objects.filter(user_id=user_id, score__isnull=False)
                .select_related("item")
                .only("item__title", "item__media_type", "score", "created_at")
            )

        library_titles = (
            _model(media_type)
            .objects.filter(user_id=user_id)
            .order_by("-created_at")
            .values_list("item__title", flat=True)
        )
        return TasteInput(
            settings=settings,
            rated=rated,
            library=tuple(dict.fromkeys(library_titles))[:LIBRARY_LIMIT],
            positive=user.get_ai_prompt(media_type, "positive"),
            negative=user.get_ai_prompt(media_type, "negative"),
        )

    def load_user_score(
        self,
        user_id: int,
        media_type: str,
        source: str,
        media_id: str,
    ) -> float | None:
        """Return the user's latest score of a media, if rated."""
        media = (
            _model(media_type)
            .objects.filter(
                user_id=user_id,
                item__media_id=media_id,
                item__source=source,
                score__isnull=False,
            )
            .order_by("-created_at")
            .first()
        )
        return None if media is None else float(media.score)

    def _user(self, user_id: int) -> User:
        """Return the user, or raise LibraryStoreNotFoundError."""
        user = cast("Any", get_user_model()).objects.filter(id=user_id).first()
        if user is None:
            raise LibraryStoreNotFoundError(user_id)
        return user


def _model(media_type: str) -> Any:
    """Return the tracking model of a media type."""
    return apps.get_model(app_label="app", model_name=media_type)
