import logging

import apprise
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import update_session_auth_hash
from django.core.cache import cache
from django.core.exceptions import ValidationError
from django.db import IntegrityError
from django.db.models import Q
from django.shortcuts import get_object_or_404, redirect, render
from django.template.defaultfilters import pluralize
from django.views.decorators.http import require_GET, require_http_methods, require_POST
from django_celery_beat.models import PeriodicTask

from app import ai_recommendations
from app.models import Item, MediaTypes
from app.providers import tmdb
from users.forms import NotificationSettingsForm, PasswordChangeForm, UserUpdateForm
from users.models import (
    AI_PROMPT_KINDS,
    AI_PROMPT_MAX_LENGTH,
    VALID_SEARCH_TYPES,
    WATCH_PROVIDER_REGION_UNSET,
    DateFormatChoices,
    QuickWatchDateChoices,
    TimeFormatChoices,
    WeekStartDayChoices,
    validate_ai_prompts,
)

logger = logging.getLogger(__name__)


@require_http_methods(["GET", "POST"])
def account(request):
    """Update the user's account and import/export data."""
    user_form = UserUpdateForm(instance=request.user)
    password_form = PasswordChangeForm(user=request.user)

    if request.method == "POST":
        # Handle username update
        if "username" in request.POST:
            user_form = UserUpdateForm(request.POST, instance=request.user)

            if user_form.is_valid():
                user_form.save()
                messages.success(request, "Your profile has been updated!")
                logger.info(
                    "Successful profile change for user: %s",
                    request.user.username,
                )
                return redirect("account")
            logger.warning(
                "Failed profile change for user: %s - %s",
                request.user.username,
                list(user_form.errors.keys()),
            )

        # Handle password update
        elif any(
            key in request.POST
            for key in ["old_password", "new_password1", "new_password2"]
        ):
            password_form = PasswordChangeForm(user=request.user, data=request.POST)

            if password_form.is_valid():
                user = password_form.save()
                update_session_auth_hash(
                    request,
                    user,
                )
                messages.success(request, "Your password has been updated!")
                logger.info(
                    "Successful password change for user: %s",
                    request.user.username,
                )
                return redirect("account")
            logger.warning(
                "Failed password change for user: %s - %s",
                request.user.username,
                list(password_form.errors.keys()),
            )

    context = {
        "user_form": user_form,
        "password_form": password_form,
    }

    return render(request, "users/account.html", context)


@require_http_methods(["GET", "POST"])
def notifications(request):
    """Render the notifications settings page."""
    if request.method == "POST":
        form = NotificationSettingsForm(request.POST, instance=request.user)
        if form.is_valid():
            form.save()
            messages.success(request, "Notification settings updated successfully!")
        else:
            for errors in form.errors.values():
                for error in errors:
                    messages.error(request, f"{error}")

        return redirect("notifications")

    form = NotificationSettingsForm(instance=request.user)

    return render(
        request,
        "users/notifications.html",
        {
            "form": form,
        },
    )


@require_GET
def search_items(request):
    """Search for items to exclude from notifications."""
    query = request.GET.get("q", "").strip()

    if not query or len(query) <= 1:
        return render(
            request,
            "users/components/search_results.html",
        )

    # Search for items that match the query
    items = (
        Item.objects.filter(
            Q(title__icontains=query),
        )
        .exclude(
            id__in=request.user.notification_excluded_items.values_list(
                "id",
                flat=True,
            ),
        )
        .distinct()[:10]
    )

    return render(
        request,
        "users/components/search_results.html",
        {"items": items, "query": query},
    )


@require_POST
def exclude_item(request):
    """Exclude an item from notifications."""
    item_id = request.POST["item_id"]
    item = get_object_or_404(Item, id=item_id)
    request.user.notification_excluded_items.add(item)

    # Return the updated excluded items list
    excluded_items = request.user.notification_excluded_items.all()

    return render(
        request,
        "users/components/excluded_items.html",
        {"excluded_items": excluded_items},
    )


@require_POST
def include_item(request):
    """Remove an item from the exclusion list."""
    item_id = request.POST["item_id"]
    item = get_object_or_404(Item, id=item_id)
    request.user.notification_excluded_items.remove(item)

    # Return the updated excluded items list
    excluded_items = request.user.notification_excluded_items.all()

    return render(
        request,
        "users/components/excluded_items.html",
        {"excluded_items": excluded_items},
    )


@require_GET
def test_notification(request):
    """Send a test notification to the user."""
    try:
        # Create Apprise instance
        apobj = apprise.Apprise()

        # Add all notification URLs
        notification_urls = [
            url.strip()
            for url in request.user.notification_urls.splitlines()
            if url.strip()
        ]
        if not notification_urls:
            messages.error(request, "No notification URLs configured.")
            return redirect("notifications")

        for url in notification_urls:
            apobj.add(url)

        # Send test notification
        result = apobj.notify(
            title="YamTrack Test Notification",
            body=(
                "This is a test notification from YamTrack. "
                "If you're seeing this, your notifications are working correctly!"
            ),
        )

        if result:
            messages.success(request, "Test notification sent successfully!")
        else:
            messages.error(request, "Failed to send test notification.")
    except Exception:
        logger.exception("Error sending notification")

    return redirect("notifications")


AI_PROMPT_FIELDS = [
    "ai_include_liked",
    "ai_liked_min_score",
    "ai_include_disliked",
    "ai_disliked_max_score",
]


def parse_ai_recommendation_preferences(user, data):
    """Return the submitted AI recommendation preferences by field name."""
    return {
        "ai_recommendations_enabled": "ai_recommendations_enabled" in data,
        "ai_media_recommendations_enabled": "ai_media_recommendations_enabled" in data,
        "ai_include_liked": "ai_include_liked" in data,
        "ai_include_disliked": "ai_include_disliked" in data,
        "ai_liked_min_score": parse_score(
            data.get("ai_liked_min_score"),
            user.ai_liked_min_score,
        ),
        "ai_disliked_max_score": parse_score(
            data.get("ai_disliked_max_score"),
            user.ai_disliked_max_score,
        ),
        "ai_prompts": parse_ai_prompts(data),
    }


def validate_ai_recommendation_preferences(preferences):
    """Return an error message if the AI recommendation preferences are invalid."""
    if preferences["ai_liked_min_score"] <= preferences["ai_disliked_max_score"]:
        return "The minimum liked score must be higher than the maximum disliked score."
    try:
        validate_ai_prompts(preferences["ai_prompts"])
    except ValidationError as error:
        return " ".join(error.messages)
    return None


def get_ai_prompt_state(user, media_types):
    """Return the settings AI recommendations are generated from."""
    fields = [
        *AI_PROMPT_FIELDS,
        *(f"{media_type}_enabled" for media_type in media_types),
    ]
    return [getattr(user, field) for field in fields], user.ai_prompts


def invalidate_ai_recommendations(user, previous_state, media_types):
    """Delete cached AI recommendations built from outdated settings."""
    previous_fields, previous_prompts = previous_state
    current_fields, current_prompts = get_ai_prompt_state(user, media_types)
    if current_fields != previous_fields:
        ai_recommendations.invalidate_user(user)
        return

    for media_type in VALID_SEARCH_TYPES:
        if previous_prompts.get(media_type) != current_prompts.get(media_type):
            ai_recommendations.invalidate_media_type(user, media_type)


def parse_ai_prompts(data):
    """Return the non-empty AI prompts submitted for each media type."""
    prompts = {}
    for media_type in VALID_SEARCH_TYPES:
        media_prompts = {}
        for kind in AI_PROMPT_KINDS:
            # Browsers count a line break as one character towards maxlength
            text = (
                data.get(f"ai_prompt_{kind}_{media_type}", "")
                .replace("\r\n", "\n")
                .strip()
            )
            if text:
                media_prompts[kind] = text
        if media_prompts:
            prompts[media_type] = media_prompts
    return prompts


def parse_score(value, default):
    """Return ``value`` as a score clamped to 0-10, or ``default`` if invalid."""
    try:
        score = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, min(10, score))


@require_http_methods(["GET", "POST"])
def preferences(request):
    """Render the preferences settings page."""
    media_types = MediaTypes.values
    media_types.remove(MediaTypes.EPISODE.value)
    watch_provider_regions = tmdb.watch_provider_regions()

    if request.method == "GET":
        return render(
            request,
            "users/preferences.html",
            {
                "media_types": media_types,
                "quick_watch_date_choices": QuickWatchDateChoices.choices,
                "date_format_choices": DateFormatChoices.choices,
                "time_format_choices": TimeFormatChoices.choices,
                "week_start_day_choices": WeekStartDayChoices.choices,
                "watch_provider_choices": watch_provider_regions,
                "gemini_configured": bool(settings.GEMINI_API_KEY),
                "ai_prompt_max_length": AI_PROMPT_MAX_LENGTH,
                "ai_prompts": [
                    {
                        "media_type": media_type,
                        "positive": request.user.get_ai_prompt(media_type, "positive"),
                        "negative": request.user.get_ai_prompt(media_type, "negative"),
                    }
                    for media_type in VALID_SEARCH_TYPES
                ],
            },
        )

    # Prevent demo users from updating preferences
    if request.user.is_demo:
        messages.error(request, "This section is view-only for demo accounts.")
        return redirect("preferences")

    # Process form submission
    request.user.clickable_media_cards = "clickable_media_cards" in request.POST
    request.user.obfuscate_unseen_episodes = "obfuscate_unseen_episodes" in request.POST
    request.user.quick_watch_date = request.POST.get(
        "quick_watch_date",
        QuickWatchDateChoices.CURRENT_DATE,
    )
    request.user.progress_bar = "progress_bar" in request.POST
    request.user.hide_completed_recommendations = (
        "hide_completed_recommendations" in request.POST
    )
    request.user.hide_zero_rating = "hide_zero_rating" in request.POST
    request.user.date_format = request.POST.get(
        "date_format",
        DateFormatChoices.ISO,
    )
    request.user.time_format = request.POST.get(
        "time_format",
        TimeFormatChoices.HOUR_24,
    )
    week_start_day = request.POST.get("week_start_day")
    if week_start_day in WeekStartDayChoices.values:
        request.user.week_start_day = week_start_day
    media_types_checked = request.POST.getlist("media_types_checkboxes")

    provider_region = request.POST.get("watch_provider_region", "")
    if provider_region in [region[0] for region in watch_provider_regions]:
        request.user.watch_provider_region = provider_region
    else:
        request.user.watch_provider_region = WATCH_PROVIDER_REGION_UNSET

    previous_ai_state = get_ai_prompt_state(request.user, media_types)

    # Update user preferences for each media type
    for media_type in media_types:
        setattr(
            request.user,
            f"{media_type}_enabled",
            media_type in media_types_checked,
        )

    # Invalid AI preferences are skipped so the other preferences still save
    ai_preferences = parse_ai_recommendation_preferences(request.user, request.POST)
    error = validate_ai_recommendation_preferences(ai_preferences)
    if error:
        messages.error(request, f"AI recommendation settings weren't saved. {error}")
    else:
        for field, value in ai_preferences.items():
            setattr(request.user, field, value)

    # Save changes and redirect
    request.user.save()

    # Cached recommendations were built from the previous prompt settings
    invalidate_ai_recommendations(request.user, previous_ai_state, media_types)
    messages.success(request, "Settings updated.")

    return redirect("preferences")


@require_GET
def integrations(request):
    """Render the integrations settings page."""
    return render(request, "users/integrations.html")


@require_GET
def import_data(request):
    """Render the import data settings page."""
    import_tasks = request.user.get_import_tasks()
    return render(request, "users/import_data.html", {"import_tasks": import_tasks})


@require_GET
def export_data(request):
    """Render the export data settings page."""
    return render(request, "users/export_data.html")


@require_GET
def advanced(request):
    """Render the advanced settings page."""
    return render(request, "users/advanced.html")


@require_GET
def about(request):
    """Render the about page."""
    return render(request, "users/about.html", {"version": settings.VERSION})


@require_POST
def delete_import_schedule(request):
    """Delete an import schedule."""
    task_name = request.POST.get("task_name")
    try:
        task = PeriodicTask.objects.get(
            name=task_name,
            kwargs__contains=f'"user_id": {request.user.id}',
        )
        task.delete()
        messages.success(request, "Import schedule deleted.")
    except PeriodicTask.DoesNotExist:
        messages.error(request, "Import schedule not found.")
    return redirect("import_data")


@require_POST
def regenerate_token(request):
    """Regenerate the token for the user."""
    while True:
        try:
            request.user.regenerate_token()
            messages.success(request, "Token regenerated successfully.")
            break
        except IntegrityError:
            continue
    return redirect("integrations")


@require_POST
def update_plex_usernames(request):
    """Update the Plex usernames for the user."""
    usernames = request.POST.get("plex_usernames", "")

    username_list = [u.strip() for u in usernames.split(",") if u.strip()]

    seen = set()
    deduplicated_usernames = [
        u for u in username_list if not (u in seen or seen.add(u))
    ]

    # Reconstruct with comma-space separation
    cleaned_usernames = ", ".join(deduplicated_usernames)

    if cleaned_usernames != request.user.plex_usernames:
        request.user.plex_usernames = cleaned_usernames
        request.user.save(update_fields=["plex_usernames"])
        messages.success(request, "Plex usernames updated successfully")

    return redirect("integrations")


@require_POST
def update_jellyfin_webhook_events(request):
    """Update optional Jellyfin webhook event handling for the user."""
    request.user.jellyfin_mark_played_enabled = (
        "jellyfin_mark_played_enabled" in request.POST
    )
    request.user.jellyfin_mark_unplayed_enabled = (
        "jellyfin_mark_unplayed_enabled" in request.POST
    )
    request.user.save(
        update_fields=[
            "jellyfin_mark_played_enabled",
            "jellyfin_mark_unplayed_enabled",
        ],
    )
    messages.success(request, "Jellyfin webhook settings updated successfully")

    return redirect("integrations")


@require_POST
def clear_search_cache(request):
    """Clear all cached search entries."""
    deleted = cache.delete_pattern("search_*")

    messages.success(
        request,
        f"Successfully cleared {deleted} search entr{pluralize(deleted, 'y,ies')}",
    )
    logger.info(
        "Successfully cleared %s search entries",
        deleted,
    )

    return redirect("advanced")
