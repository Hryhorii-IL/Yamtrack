from __future__ import annotations

from typing import Final

CACHE_TIMEOUT: Final = 60 * 60 * 24  # 1 day
# Failed generations are shown briefly, then retried on the next visit
ERROR_TIMEOUT: Final = 60
# Extra lock time for the provider searches after the LLM responds
LOCK_EXTRA_TIMEOUT: Final = 60 * 2

GLOBAL_COUNT: Final = 16
# Global recommendations shown on the page
POOL_SHOWN: Final = 32
# Spare pool items replace the ones the user tracks later
POOL_SPARE: Final = 8
# Global recommendations are extended in the background up to this pool size
POOL_SIZE: Final = POOL_SHOWN + POOL_SPARE
# Background batches per pool, so duplicate-heavy batches can't loop forever
MAX_EXTENSIONS: Final = 5
MEDIA_COUNT: Final = 12
# Ask for extra titles since some can't be resolved or are already tracked
EXTRA_REQUESTED: Final = 8

LIKED_LIMIT: Final = 50
DISLIKED_LIMIT: Final = 25
LIBRARY_LIMIT: Final = 200
SYNOPSIS_LIMIT: Final = 600

SECTION_NAME: Final = "ai_recommendations"
# Concurrent provider searches, still bounded by the providers' rate limiters
RESOLVE_WORKERS: Final = 4

# Values of app.models.MediaTypes, kept as plain strings for pure modules
MEDIA_TYPES: Final = (
    "tv",
    "movie",
    "anime",
    "manga",
    "game",
    "book",
    "comic",
    "boardgame",
)

TYPE_LABELS: Final = {
    "tv": ("TV show", "TV shows"),
    "movie": ("movie", "movies"),
    "anime": ("anime", "anime"),
    "manga": ("manga", "manga"),
    "game": ("video game", "video games"),
    "book": ("book", "books"),
    "comic": ("comic", "comics"),
    "boardgame": ("board game", "board games"),
}
