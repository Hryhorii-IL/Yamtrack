import secrets

from django.conf import settings
from django.contrib.auth.models import AbstractUser
from django.core.exceptions import ValidationError
from django.db import models
from django_celery_beat.models import PeriodicTask
from django_celery_results.models import TaskResult

from app.models import Item, MediaTypes, Status
from users import helpers

EXCLUDED_SEARCH_TYPES = [MediaTypes.SEASON.value, MediaTypes.EPISODE.value]

VALID_SEARCH_TYPES = [
    value for value in MediaTypes.values if value not in EXCLUDED_SEARCH_TYPES
]

# Sentinel for an unconfigured watch provider region.
WATCH_PROVIDER_REGION_UNSET = "UNSET"


AI_PROMPT_KINDS = ("positive", "negative")
AI_PROMPT_MAX_LENGTH = 500


def validate_ai_prompts(value):
    """Validate {media_type: {"positive": str, "negative": str}} AI prompts."""
    if not isinstance(value, dict):
        msg = "AI prompts must be an object."
        raise ValidationError(msg)

    for media_type, prompts in value.items():
        if media_type not in VALID_SEARCH_TYPES:
            msg = f"Invalid media type for AI prompts: {media_type}."
            raise ValidationError(msg)
        if not isinstance(prompts, dict) or not set(prompts) <= set(AI_PROMPT_KINDS):
            msg = f"AI prompts for {media_type} must only have positive and negative."
            raise ValidationError(msg)
        for text in prompts.values():
            if not isinstance(text, str) or len(text) > AI_PROMPT_MAX_LENGTH:
                msg = (
                    f"AI prompts for {media_type} must be text of at most "
                    f"{AI_PROMPT_MAX_LENGTH} characters."
                )
                raise ValidationError(msg)


def generate_token():
    """Generate a user token."""
    return secrets.token_urlsafe(24)


class HomeSortChoices(models.TextChoices):
    """Choices for home page sort options."""

    UPCOMING = "upcoming", "Upcoming"
    RECENT = "recent", "Recent"
    COMPLETION = "completion", "Completion"
    EPISODES_LEFT = "episodes_left", "Episodes Left"
    TITLE = "title", "Title"


class MediaSortChoices(models.TextChoices):
    """Choices for media list sort options."""

    SCORE = "score", "Rating"
    TITLE = "title", "Title"
    PROGRESS = "progress", "Progress"
    START_DATE = "start_date", "Start Date"
    END_DATE = "end_date", "End Date"


class MediaStatusChoices(models.TextChoices):
    """Choices for media list status options."""

    ALL = "All", "All"
    COMPLETED = Status.COMPLETED.value, Status.COMPLETED.label
    IN_PROGRESS = Status.IN_PROGRESS.value, Status.IN_PROGRESS.label
    PLANNING = Status.PLANNING.value, Status.PLANNING.label
    PAUSED = Status.PAUSED.value, Status.PAUSED.label
    DROPPED = Status.DROPPED.value, Status.DROPPED.label


class LayoutChoices(models.TextChoices):
    """Choices for media list layout options."""

    GRID = "grid", "Grid"
    TABLE = "table", "Table"


class CalendarLayoutChoices(models.TextChoices):
    """Choices for calendar layout options."""

    GRID = "grid", "Grid"
    LIST = "list", "List"


class ListSortChoices(models.TextChoices):
    """Choices for list sort options."""

    LAST_ITEM_ADDED = "last_item_added", "Last Item Added"
    NAME = "name", "Name"
    ITEMS_COUNT = "items_count", "Items Count"
    NEWEST_FIRST = "newest_first", "Newest First"


class ListDetailSortChoices(models.TextChoices):
    """Choices for list detail sort options."""

    DATE_ADDED = "date_added", "Date Added"
    TITLE = "title", "Title"
    MEDIA_TYPE = "media_type", "Media Type"


class QuickWatchDateChoices(models.TextChoices):
    """Choices for quick watch date behavior when bulk-marking media as completed."""

    CURRENT_DATE = "current_date", "Current Date"
    RELEASE_DATE = "release_date", "Release Date"
    NO_DATE = "no_date", "No Date"


class DateFormatChoices(models.TextChoices):
    """Choices for date format display."""

    ISO = "Y-m-d", "2026-01-18 (ISO)"
    EUROPEAN = "d/m/Y", "18/01/2026 (EU)"
    US = "m/d/Y", "01/18/2026 (US)"
    LONG = "M j, Y", "Jan 18, 2026"
    LONG_EU = "j M, Y", "18 Jan, 2026"


class TimeFormatChoices(models.TextChoices):
    """Choices for time format display."""

    HOUR_24 = "H:i", "14:30 (24-hour)"
    HOUR_12 = "g:i A", "2:30 PM (12-hour)"


class WeekStartDayChoices(models.TextChoices):
    """Choices for week start day."""

    MONDAY = "monday", "Monday"
    SUNDAY = "sunday", "Sunday"


class User(AbstractUser):
    """Custom user model."""

    is_demo = models.BooleanField(default=False)

    profile_private = models.BooleanField(
        default=True, help_text="Toggle profile visibility to anonymous users"
    )

    last_search_type = models.CharField(
        max_length=10,
        default=MediaTypes.TV.value,
        choices=MediaTypes.choices,
    )

    home_sort = models.CharField(
        max_length=20,
        default=HomeSortChoices.UPCOMING,
        choices=HomeSortChoices,
    )
    home_hide_unreleased = models.BooleanField(
        default=False,
        help_text="Hide unreleased media from the home page",
    )

    # Media type preferences: TV Shows
    tv_enabled = models.BooleanField(default=True)
    tv_layout = models.CharField(
        max_length=20,
        default=LayoutChoices.GRID,
        choices=LayoutChoices,
    )
    tv_sort = models.CharField(
        max_length=20,
        default=MediaSortChoices.SCORE,
        choices=MediaSortChoices,
    )
    tv_status = models.CharField(
        max_length=20,
        default=MediaStatusChoices.ALL,
        choices=MediaStatusChoices,
    )

    # Media type preferences: TV Seasons
    season_enabled = models.BooleanField(default=True)
    season_layout = models.CharField(
        max_length=20,
        default=LayoutChoices.GRID,
        choices=LayoutChoices,
    )
    season_sort = models.CharField(
        max_length=20,
        default=MediaSortChoices.SCORE,
        choices=MediaSortChoices,
    )
    season_status = models.CharField(
        max_length=20,
        default=MediaStatusChoices.ALL,
        choices=MediaStatusChoices,
    )

    # Media type preferences: Movies
    movie_enabled = models.BooleanField(default=True)
    movie_layout = models.CharField(
        max_length=20,
        default=LayoutChoices.GRID,
        choices=LayoutChoices,
    )
    movie_sort = models.CharField(
        max_length=20,
        default=MediaSortChoices.SCORE,
        choices=MediaSortChoices,
    )
    movie_status = models.CharField(
        max_length=20,
        default=MediaStatusChoices.ALL,
        choices=MediaStatusChoices,
    )

    # Media type preferences: Anime
    anime_enabled = models.BooleanField(default=True)
    anime_layout = models.CharField(
        max_length=20,
        default=LayoutChoices.TABLE,
        choices=LayoutChoices,
    )
    anime_sort = models.CharField(
        max_length=20,
        default=MediaSortChoices.SCORE,
        choices=MediaSortChoices,
    )
    anime_status = models.CharField(
        max_length=20,
        default=MediaStatusChoices.ALL,
        choices=MediaStatusChoices,
    )

    # Media type preferences: Manga
    manga_enabled = models.BooleanField(default=True)
    manga_layout = models.CharField(
        max_length=20,
        default=LayoutChoices.TABLE,
        choices=LayoutChoices,
    )
    manga_sort = models.CharField(
        max_length=20,
        default=MediaSortChoices.SCORE,
        choices=MediaSortChoices,
    )
    manga_status = models.CharField(
        max_length=20,
        default=MediaStatusChoices.ALL,
        choices=MediaStatusChoices,
    )

    # Media type preferences: Games
    game_enabled = models.BooleanField(default=True)
    game_layout = models.CharField(
        max_length=20,
        default=LayoutChoices.GRID,
        choices=LayoutChoices,
    )
    game_sort = models.CharField(
        max_length=20,
        default=MediaSortChoices.SCORE,
        choices=MediaSortChoices,
    )
    game_status = models.CharField(
        max_length=20,
        default=MediaStatusChoices.ALL,
        choices=MediaStatusChoices,
    )

    # Media type preferences: Books
    book_enabled = models.BooleanField(default=True)
    book_layout = models.CharField(
        max_length=20,
        default=LayoutChoices.GRID,
        choices=LayoutChoices,
    )
    book_sort = models.CharField(
        max_length=20,
        default=MediaSortChoices.SCORE,
        choices=MediaSortChoices,
    )
    book_status = models.CharField(
        max_length=20,
        default=MediaStatusChoices.ALL,
        choices=MediaStatusChoices,
    )

    # Media type preferences: Comics
    comic_enabled = models.BooleanField(default=True)
    comic_layout = models.CharField(
        max_length=20,
        default=LayoutChoices.GRID,
        choices=LayoutChoices,
    )
    comic_sort = models.CharField(
        max_length=20,
        default=MediaSortChoices.SCORE,
        choices=MediaSortChoices,
    )
    comic_status = models.CharField(
        max_length=20,
        default=MediaStatusChoices.ALL,
        choices=MediaStatusChoices,
    )

    # Media type preferences: Board Games
    boardgame_enabled = models.BooleanField(default=True)
    boardgame_layout = models.CharField(
        max_length=20,
        default=LayoutChoices.GRID,
        choices=LayoutChoices,
    )
    boardgame_sort = models.CharField(
        max_length=20,
        default=MediaSortChoices.SCORE,
        choices=MediaSortChoices,
    )
    boardgame_status = models.CharField(
        max_length=20,
        default=MediaStatusChoices.ALL,
        choices=MediaStatusChoices,
    )

    # UI preferences
    clickable_media_cards = models.BooleanField(
        default=False,
        help_text="Hide hover overlay on touch devices",
    )

    obfuscate_unseen_episodes = models.BooleanField(
        default=False,
        help_text="Blur unseen episode images and descriptions",
    )

    # Tracking settings
    quick_watch_date = models.CharField(
        max_length=20,
        default=QuickWatchDateChoices.CURRENT_DATE,
        choices=QuickWatchDateChoices,
        help_text="Date to use when bulk-marking media as completed",
    )

    date_format = models.CharField(
        max_length=20,
        default=DateFormatChoices.ISO,
        choices=DateFormatChoices,
        help_text="Preferred date display format",
    )
    time_format = models.CharField(
        max_length=20,
        default=TimeFormatChoices.HOUR_24,
        choices=TimeFormatChoices,
        help_text="Preferred time display format",
    )

    week_start_day = models.CharField(
        max_length=10,
        default=WeekStartDayChoices.MONDAY,
        choices=WeekStartDayChoices,
        help_text="First day of the week",
    )

    # Progress bar
    progress_bar = models.BooleanField(
        default=True,
        help_text="Show progress bar",
    )

    # Hide completed recommendations
    hide_completed_recommendations = models.BooleanField(
        default=False,
        help_text="Hide completed media in recommendations",
    )

    # Hide zero ratings
    hide_zero_rating = models.BooleanField(
        default=False,
        help_text="Hide zero ratings from media cards",
    )

    # AI recommendations
    ai_recommendations_enabled = models.BooleanField(
        default=False,
        help_text="Show the AI recommendations page",
    )
    ai_media_recommendations_enabled = models.BooleanField(
        default=False,
        help_text="Show an AI recommendations tab on media details pages",
    )
    ai_include_liked = models.BooleanField(
        default=True,
        help_text="Include liked media in AI recommendation prompts",
    )
    ai_liked_min_score = models.PositiveSmallIntegerField(
        default=8,
        help_text="Minimum score for media to count as liked",
    )
    ai_include_disliked = models.BooleanField(
        default=True,
        help_text="Include disliked media in AI recommendation prompts",
    )
    ai_disliked_max_score = models.PositiveSmallIntegerField(
        default=4,
        help_text="Maximum score for media to count as disliked",
    )
    ai_prompts = models.JSONField(
        default=dict,
        blank=True,
        validators=[validate_ai_prompts],
        help_text="Wishes added to AI recommendation prompts per media type",
    )

    # Watch provider region
    watch_provider_region = models.CharField(
        max_length=5,
        default=WATCH_PROVIDER_REGION_UNSET,
        help_text="Region to show watch providers for",
    )

    # Calendar preferences
    calendar_layout = models.CharField(
        max_length=20,
        default=CalendarLayoutChoices.GRID,
        choices=CalendarLayoutChoices,
    )

    # Lists preferences
    lists_sort = models.CharField(
        max_length=20,
        default=ListSortChoices.LAST_ITEM_ADDED,
        choices=ListSortChoices,
    )
    list_detail_sort = models.CharField(
        max_length=20,
        default=ListDetailSortChoices.DATE_ADDED,
        choices=ListDetailSortChoices,
    )
    list_detail_status = models.CharField(
        max_length=20,
        default=MediaStatusChoices.ALL,
        choices=MediaStatusChoices,
    )

    # Notification settings
    notification_urls = models.TextField(
        blank=True,
        help_text="Apprise URLs for notifications",
    )
    notification_excluded_items = models.ManyToManyField(
        Item,
        related_name="excluded_by_users",
        blank=True,
        help_text="Items excluded from notifications",
    )
    release_notifications_enabled = models.BooleanField(
        default=True,
        help_text="Receive notifications for recently released media",
    )
    daily_digest_enabled = models.BooleanField(
        default=True,
        help_text="Receive a daily digest of upcoming releases",
    )

    # Integration settings
    token = models.CharField(
        max_length=32,
        unique=True,
        default=generate_token,
        help_text="Token for external integrations",
    )
    plex_usernames = models.TextField(
        blank=True,
        help_text="Comma-separated list of Plex usernames for webhook matching",
    )
    jellyfin_mark_played_enabled = models.BooleanField(
        default=False,
        help_text="Process Jellyfin MarkPlayed webhook events",
    )
    jellyfin_mark_unplayed_enabled = models.BooleanField(
        default=False,
        help_text="Process Jellyfin MarkUnplayed webhook events",
    )

    class Meta:
        """Meta options for the model."""

        ordering = ["username"]
        constraints = [
            models.CheckConstraint(
                name="last_search_type_valid",
                condition=models.Q(last_search_type__in=VALID_SEARCH_TYPES),
            ),
            models.CheckConstraint(
                name="home_sort_valid",
                condition=models.Q(home_sort__in=HomeSortChoices.values),
            ),
            models.CheckConstraint(
                name="tv_layout_valid",
                condition=models.Q(tv_layout__in=LayoutChoices.values),
            ),
            models.CheckConstraint(
                name="season_layout_valid",
                condition=models.Q(season_layout__in=LayoutChoices.values),
            ),
            models.CheckConstraint(
                name="movie_layout_valid",
                condition=models.Q(movie_layout__in=LayoutChoices.values),
            ),
            models.CheckConstraint(
                name="anime_layout_valid",
                condition=models.Q(anime_layout__in=LayoutChoices.values),
            ),
            models.CheckConstraint(
                name="manga_layout_valid",
                condition=models.Q(manga_layout__in=LayoutChoices.values),
            ),
            models.CheckConstraint(
                name="game_layout_valid",
                condition=models.Q(game_layout__in=LayoutChoices.values),
            ),
            models.CheckConstraint(
                name="book_layout_valid",
                condition=models.Q(book_layout__in=LayoutChoices.values),
            ),
            models.CheckConstraint(
                name="tv_sort_valid",
                condition=models.Q(tv_sort__in=MediaSortChoices.values),
            ),
            models.CheckConstraint(
                name="season_sort_valid",
                condition=models.Q(season_sort__in=MediaSortChoices.values),
            ),
            models.CheckConstraint(
                name="movie_sort_valid",
                condition=models.Q(movie_sort__in=MediaSortChoices.values),
            ),
            models.CheckConstraint(
                name="anime_sort_valid",
                condition=models.Q(anime_sort__in=MediaSortChoices.values),
            ),
            models.CheckConstraint(
                name="manga_sort_valid",
                condition=models.Q(manga_sort__in=MediaSortChoices.values),
            ),
            models.CheckConstraint(
                name="game_sort_valid",
                condition=models.Q(game_sort__in=MediaSortChoices.values),
            ),
            models.CheckConstraint(
                name="book_sort_valid",
                condition=models.Q(book_sort__in=MediaSortChoices.values),
            ),
            models.CheckConstraint(
                name="calendar_layout_valid",
                condition=models.Q(calendar_layout__in=CalendarLayoutChoices.values),
            ),
            models.CheckConstraint(
                name="lists_sort_valid",
                condition=models.Q(lists_sort__in=ListSortChoices.values),
            ),
            models.CheckConstraint(
                name="list_detail_sort_valid",
                condition=models.Q(list_detail_sort__in=ListDetailSortChoices.values),
            ),
            models.CheckConstraint(
                name="list_detail_status_valid",
                condition=models.Q(list_detail_status__in=MediaStatusChoices.values),
            ),
            models.CheckConstraint(
                name="tv_status_valid",
                condition=models.Q(tv_status__in=MediaStatusChoices.values),
            ),
            models.CheckConstraint(
                name="season_status_valid",
                condition=models.Q(season_status__in=MediaStatusChoices.values),
            ),
            models.CheckConstraint(
                name="movie_status_valid",
                condition=models.Q(movie_status__in=MediaStatusChoices.values),
            ),
            models.CheckConstraint(
                name="anime_status_valid",
                condition=models.Q(anime_status__in=MediaStatusChoices.values),
            ),
            models.CheckConstraint(
                name="manga_status_valid",
                condition=models.Q(manga_status__in=MediaStatusChoices.values),
            ),
            models.CheckConstraint(
                name="game_status_valid",
                condition=models.Q(game_status__in=MediaStatusChoices.values),
            ),
            models.CheckConstraint(
                name="book_status_valid",
                condition=models.Q(book_status__in=MediaStatusChoices.values),
            ),
            models.CheckConstraint(
                name="quick_watch_date_valid",
                condition=models.Q(quick_watch_date__in=QuickWatchDateChoices.values),
            ),
            models.CheckConstraint(
                name="week_start_day_valid",
                condition=models.Q(week_start_day__in=WeekStartDayChoices.values),
            ),
            models.CheckConstraint(
                name="ai_liked_min_score_valid",
                condition=models.Q(ai_liked_min_score__lte=10),
            ),
            models.CheckConstraint(
                name="ai_disliked_max_score_valid",
                condition=models.Q(ai_disliked_max_score__lte=10),
            ),
            models.CheckConstraint(
                name="ai_liked_above_disliked_score",
                condition=models.Q(
                    ai_liked_min_score__gt=models.F("ai_disliked_max_score"),
                ),
            ),
        ]

    def update_preference(self, field_name, new_value):
        """
        Update user preference if the new value is valid and different from current.

        Args:
            field_name: The name of the field to update
            new_value: The new value to set

        Returns:
            The value that was set (or the original value if invalid)
        """
        current_value = getattr(self, field_name)
        preference_value = self.get_valid_preference(field_name, new_value)

        if preference_value != current_value:
            setattr(self, field_name, preference_value)
            self.save(update_fields=[field_name])

        return preference_value

    def get_valid_preference(self, field_name, new_value):
        """
        Return a valid preference value without saving it.

        Args:
            field_name: The name of the field to validate against
            new_value: The new value to check

        Returns:
            The new value if valid, otherwise the current field value.
        """
        # If no new value provided, return current value
        if new_value is None:
            return getattr(self, field_name)

        # Special case for last_search_type
        if field_name == "last_search_type" and new_value not in VALID_SEARCH_TYPES:
            return getattr(self, field_name)

        field = self._meta.get_field(field_name)
        # Check if the field has choices
        if hasattr(field, "choices") and field.choices:
            # Get valid values from field choices
            valid_values = [choice[0] for choice in field.choices]

            # If the new value is not valid, return current value
            if new_value not in valid_values:
                return getattr(self, field_name)

        return new_value

    def resolve_watch_date(self, now, release_date):
        """
        Resolve the appropriate watch date based on user preference.

        Args:
            now: Pre-calculated current datetime
            release_date: The release/air date for the specific media item

        Returns:
            datetime or None based on user preference
        """
        if self.quick_watch_date == QuickWatchDateChoices.NO_DATE:
            return None

        if self.quick_watch_date == QuickWatchDateChoices.RELEASE_DATE:
            return release_date  # Will be None if not available in metadata

        # CURRENT_DATE is the default
        return now

    @property
    def ai_recommendations_available(self):
        """Return whether the AI recommendations page is usable."""
        return self.ai_recommendations_enabled and bool(settings.GEMINI_API_KEY)

    @property
    def ai_media_recommendations_available(self):
        """Return whether the AI tab on media details pages is usable."""
        return self.ai_media_recommendations_enabled and bool(settings.GEMINI_API_KEY)

    def get_ai_prompt(self, media_type, kind):
        """Return the user's AI prompt text of a kind for a media type."""
        return self.ai_prompts.get(media_type, {}).get(kind, "")

    def get_enabled_media_types(self):
        """Return a list of enabled media type values based on user preferences."""
        enabled_types = []

        for media_type in MediaTypes.values:
            if media_type == MediaTypes.EPISODE.value:
                continue

            enabled_field = f"{media_type}_enabled"
            if getattr(self, enabled_field, False):
                enabled_types.append(media_type)

        return enabled_types

    def get_active_media_types(self):
        """Return a list of active media type values based on user preferences."""
        enabled_types = self.get_enabled_media_types()

        # Add season if TV is enabled (and season isn't already in the list)
        if (
            MediaTypes.TV.value in enabled_types
            and MediaTypes.SEASON.value not in enabled_types
        ):
            enabled_types.insert(0, MediaTypes.SEASON.value)

        return enabled_types

    def get_import_tasks(self):
        """Return import tasks history and schedules for the user."""
        import_tasks = {
            "trakt": "Import from Trakt",
            "simkl": "Import from SIMKL",
            "myanimelist": "Import from MyAnimeList",
            "anilist": "Import from AniList",
            "kitsu": "Import from Kitsu",
            "yamtrack": "Import from Yamtrack",
            "hltb": "Import from HowLongToBeat",
            "steam": "Import from Steam",
            "imdb": "Import from IMDB",
            "goodreads": "Import from GoodReads",
        }

        # Reverse mapping to get source from task name
        task_to_source = {v: k for k, v in import_tasks.items()}

        task_result_filter_text = f"'user_id': {self.id},"

        # Get all task results for this user
        task_results = TaskResult.objects.filter(
            task_kwargs__contains=task_result_filter_text,
            task_name__in=import_tasks.values(),
        ).order_by(
            "-date_done",
        )  # Most recent first

        # Build results list
        results = []
        for task in task_results:
            source = task_to_source[task.task_name]
            processed_task = helpers.process_task_result(task)
            results.append(
                {
                    "task": processed_task,
                    "source": source,
                    "date": task.date_done,
                    "status": task.status,
                    "summary": processed_task.summary,
                    "errors": processed_task.errors,
                },
            )

        # Get periodic tasks with their crontab schedules
        periodic_tasks_filter_text = f'"user_id": {self.id},'
        periodic_tasks = PeriodicTask.objects.filter(
            task__in=import_tasks.values(),
            kwargs__contains=periodic_tasks_filter_text,
            enabled=True,
        ).select_related("crontab")

        # Build schedules list
        schedules = []
        for periodic_task in periodic_tasks:
            source = task_to_source.get(periodic_task.task, "unknown")

            # Extract username from task name if available
            username = ""
            if " for " in periodic_task.name:
                username = periodic_task.name.split(" for ")[1].split(" at ")[0]

            schedule_info = helpers.get_next_run_info(periodic_task)
            if schedule_info:
                schedules.append(
                    {
                        "task": periodic_task,
                        "source": source,
                        "username": username,
                        "last_run": periodic_task.last_run_at,
                        "next_run": schedule_info["next_run"],
                        "schedule": schedule_info["frequency"],
                        "mode": schedule_info["mode"],
                    },
                )

        return {
            "results": results,
            "schedules": schedules,
        }

    def regenerate_token(self):
        """Regenerate the user's token."""
        self.token = generate_token()
        self.save(update_fields=["token"])
