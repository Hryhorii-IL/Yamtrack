import json
from unittest.mock import MagicMock

import anthropic
import httpx2
from django.core.cache import cache
from django.test import TestCase

from app.ai_recommendations.bl.clients.claude_client import MIN_FAST_ITEMS, ClaudeClient
from app.ai_recommendations.bl.clients.schema import RecommendationListSchema
from app.ai_recommendations.bl.errors import LLMClientAuthError, LLMClientError
from app.ai_recommendations.bl.types.recommendation import RecommendedTitle

REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def api_error(status_code):
    """Return an Anthropic API status error."""
    response = httpx2.Response(status_code, request=REQUEST)
    return anthropic.APIStatusError("failed", response=response, body=None)


def titles(count):
    """Return ``count`` distinct recommended titles."""
    return [RecommendedTitle(title=f"Title {index}") for index in range(count)]


def stream(recommendations):
    """Return a stream context manager emitting the JSON in chunks."""
    text = json.dumps(
        {"items": [{"title": title.title} for title in recommendations]},
    )
    response = MagicMock()
    response.text_stream = iter(
        text[index : index + 7] for index in range(0, len(text), 7)
    )
    context = MagicMock()
    context.__enter__.return_value = response
    return context


def client(*responses):
    """Return a Claude client whose SDK answers the requests in turn."""
    sdk = MagicMock()
    sdk.messages.stream.side_effect = list(responses)
    return ClaudeClient(api_key="key", models=["first", "second"], timeout=10, sdk=sdk)


def requests(claude):
    """Return the kwargs of every streamed request."""
    return [call.kwargs for call in claude.sdk.messages.stream.call_args_list]


class ClaudeClientTests(TestCase):
    """Test streaming recommendations from Claude."""

    def setUp(self):
        """Clear models marked as unavailable."""
        cache.clear()

    def test_streams_without_thinking(self):
        """Stream the titles with thinking disabled and the output schema."""
        claude = client(stream(titles(MIN_FAST_ITEMS)))

        self.assertEqual(
            list(claude.recommend("system", "prompt")),
            titles(MIN_FAST_ITEMS),
        )
        [kwargs] = requests(claude)
        self.assertEqual(kwargs["model"], "first")
        self.assertEqual(kwargs["system"], "system")
        self.assertEqual(kwargs["messages"], [{"role": "user", "content": "prompt"}])
        self.assertIs(kwargs["output_format"], RecommendationListSchema)
        self.assertEqual(kwargs["thinking"], {"type": "disabled"})

    def test_short_list_retried_with_thinking(self):
        """Retry on the same model with thinking when too few titles came."""
        claude = client(stream(titles(1)), stream(titles(3)))

        self.assertEqual(
            list(claude.recommend("system", "prompt")),
            titles(1) + titles(3),
        )
        fast, retry = requests(claude)
        self.assertEqual(fast["thinking"], {"type": "disabled"})
        self.assertEqual(retry["thinking"], {"type": "adaptive"})
        self.assertEqual(retry["model"], "first")

    def test_empty_responses_fall_back(self):
        """Try the next model when both requests return nothing."""
        claude = client(stream([]), stream([]), stream(titles(MIN_FAST_ITEMS)))

        self.assertEqual(
            len(list(claude.recommend("system", "prompt"))), MIN_FAST_ITEMS
        )
        self.assertEqual(
            [kwargs["model"] for kwargs in requests(claude)],
            ["first", "first", "second"],
        )

    def test_api_error(self):
        """Wrap Claude API errors, keeping the status code."""
        claude = client(api_error(529), api_error(529))

        with self.assertRaises(LLMClientError) as context:
            list(claude.recommend("system", "prompt"))

        self.assertEqual(context.exception.status_code, 529)
        self.assertIn("Claude", str(context.exception))

    def test_auth_error(self):
        """Don't try other models when the API key is rejected."""
        claude = client(api_error(401))

        with self.assertRaises(LLMClientAuthError):
            list(claude.recommend("system", "prompt"))

        self.assertEqual(len(requests(claude)), 1)

    def test_network_error(self):
        """Wrap timeouts as network errors."""
        timeout = anthropic.APITimeoutError(request=REQUEST)
        claude = client(timeout, timeout)

        with self.assertRaises(LLMClientError) as context:
            list(claude.recommend("system", "prompt"))

        self.assertIsNone(context.exception.status_code)

    def test_max_duration(self):
        """Cover each model's retry with thinking."""
        self.assertEqual(client().max_duration(), 40)
