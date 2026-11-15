import json
from unittest.mock import MagicMock, patch

import anthropic
import httpx2
from django.conf import settings
from django.core.cache import cache
from django.test import TestCase, override_settings

from app.providers import claude, llm, services

REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def api_error(status_code, message):
    """Return an Anthropic API status error with the given status code."""
    response = httpx2.Response(status_code, request=REQUEST)
    return anthropic.APIStatusError(message, response=response, body=None)


def items(count):
    """Return ``count`` distinct recommendation dicts."""
    return [{"title": f"Title {index}", "year": 2000 + index} for index in range(count)]


def stream_response(recommendations, chunk_size=7):
    """Return a mocked stream context manager emitting the JSON in chunks."""
    text = json.dumps({"items": recommendations})
    response = MagicMock()
    response.text_stream = iter(
        text[index : index + chunk_size] for index in range(0, len(text), chunk_size)
    )
    context = MagicMock()
    context.__enter__.return_value = response
    return context


@override_settings(CLAUDE_MODELS=["first", "second"])
class ClaudeTests(TestCase):
    """Test the streaming Claude client wrapper."""

    def setUp(self):
        """Clear unavailable models."""
        cache.clear()

    def stream_calls(self, mock_client):
        """Return the kwargs of every streamed request."""
        return [
            call.kwargs
            for call in mock_client.return_value.messages.stream.call_args_list
        ]

    @patch("app.providers.claude.anthropic.Anthropic")
    def test_recommend(self, mock_client):
        """Yield the streamed recommendations without thinking."""
        mock_client.return_value.messages.stream.return_value = stream_response(
            items(claude.MIN_FAST_ITEMS),
        )

        result = list(claude.recommend("system", "prompt"))

        self.assertEqual(result, items(claude.MIN_FAST_ITEMS))
        [kwargs] = self.stream_calls(mock_client)
        self.assertEqual(kwargs["model"], settings.CLAUDE_MODELS[0])
        self.assertEqual(kwargs["system"], "system")
        self.assertEqual(kwargs["messages"], [{"role": "user", "content": "prompt"}])
        self.assertIs(kwargs["output_format"], llm.RecommendationList)
        self.assertEqual(kwargs["thinking"], {"type": "disabled"})

    @patch("app.providers.claude.anthropic.Anthropic")
    def test_short_list_retried_with_thinking(self, mock_client):
        """Retry with thinking when the fast request returns too few items."""
        mock_client.return_value.messages.stream.side_effect = [
            stream_response(items(1)),
            stream_response(items(3)),
        ]

        result = list(claude.recommend("system", "prompt"))

        self.assertEqual(result, items(1) + items(3))
        first, second = self.stream_calls(mock_client)
        self.assertEqual(first["thinking"], {"type": "disabled"})
        self.assertEqual(second["thinking"], {"type": "adaptive"})
        self.assertEqual(second["model"], "first")

    @patch("app.providers.claude.anthropic.Anthropic")
    def test_empty_responses_fall_back_to_next_model(self, mock_client):
        """Try the next model when both requests return nothing."""
        mock_client.return_value.messages.stream.side_effect = [
            stream_response([]),
            stream_response([]),
            stream_response(items(claude.MIN_FAST_ITEMS)),
        ]

        result = list(claude.recommend("system", "prompt"))

        self.assertEqual(len(result), claude.MIN_FAST_ITEMS)
        self.assertEqual(
            [kwargs["model"] for kwargs in self.stream_calls(mock_client)],
            ["first", "first", "second"],
        )

    @patch("app.providers.claude.anthropic.Anthropic")
    def test_recommend_api_error(self, mock_client):
        """Wrap Claude API errors in ProviderAPIError."""
        mock_client.return_value.messages.stream.side_effect = api_error(
            429,
            "rate limited",
        )

        with self.assertRaises(services.ProviderAPIError) as context:
            list(claude.recommend("system", "prompt"))

        self.assertEqual(context.exception.status_code, 429)
        self.assertIn("Claude", str(context.exception))

    @patch("app.providers.claude.anthropic.Anthropic")
    def test_recommend_network_error(self, mock_client):
        """Wrap network errors and timeouts in ProviderAPIError."""
        mock_client.return_value.messages.stream.side_effect = (
            anthropic.APITimeoutError(request=REQUEST)
        )

        with self.assertRaises(services.ProviderAPIError) as context:
            list(claude.recommend("system", "prompt"))

        self.assertIsNone(context.exception.status_code)

    @patch("app.providers.claude.anthropic.Anthropic")
    def test_fallback_to_next_model(self, mock_client):
        """Try the next model when one fails and skip it next time."""
        mock_client.return_value.messages.stream.side_effect = [
            api_error(529, "overloaded"),
            stream_response(items(claude.MIN_FAST_ITEMS)),
        ]

        result = list(claude.recommend("system", "prompt"))

        self.assertEqual(len(result), claude.MIN_FAST_ITEMS)
        self.assertIsNotNone(cache.get(llm.unavailable_cache_key("claude", "first")))

    @patch("app.providers.claude.anthropic.Anthropic")
    def test_auth_error_stops_fallback(self, mock_client):
        """Don't try other models when the API key is rejected."""
        mock_client.return_value.messages.stream.side_effect = api_error(
            401,
            "invalid x-api-key",
        )

        with self.assertRaises(services.ProviderAPIError):
            list(claude.recommend("system", "prompt"))

        mock_client.return_value.messages.stream.assert_called_once()

    @override_settings(CLAUDE_MODELS=["a", "b"], CLAUDE_TIMEOUT=20)
    def test_max_duration(self):
        """Cover every model timing out, including the retry with thinking."""
        self.assertEqual(claude.max_duration(), 80)
