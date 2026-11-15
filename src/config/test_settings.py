from fakeredis import FakeRedisConnection

from .settings import *  # noqa: F403

CACHES = {
    "default": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": REDIS_URL,  # noqa: F405
        "TIMEOUT": 18000,  # 5 hours
        "OPTIONS": {
            "CONNECTION_POOL_KWARGS": {"connection_class": FakeRedisConnection},
        },
    },
}

CELERY_TASK_ALWAYS_EAGER = True

TESTING = True

# Steam API key for testing
STEAM_API_KEY = "test_steam_api_key"

# AI provider settings for testing, independent of the local .env
AI_PROVIDER = "gemini"
GEMINI_API_KEY = "test_gemini_api_key"
GEMINI_MODELS = ["gemini-test"]
ANTHROPIC_API_KEY = "test_anthropic_api_key"
CLAUDE_MODELS = ["claude-test"]
