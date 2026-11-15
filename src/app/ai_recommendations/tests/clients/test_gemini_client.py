import json
from unittest.mock import MagicMock

import httpx
from django.core.cache import cache
from django.test import TestCase
from google.genai import errors

from app.ai_recommendations.bl.clients.gemini_client import GeminiClient
from app.ai_recommendations.bl.clients.schema import RecommendationListSchema
from app.ai_recommendations.bl.errors import LLMClientError
from app.ai_recommendations.bl.types.recommendation import RecommendedTitle

DUNE = RecommendedTitle(title="Dune", year=2021)


def chunks(*texts):
    """Return streamed response chunks with the given texts."""
    return iter(MagicMock(text=text) for text in texts)


def failing(error, *texts):
    """Return a stream that yields the given texts and then raises."""
    yield from chunks(*texts)
    raise error


def client(*streams):
    """Return a Gemini client whose SDK streams the given responses in turn."""
    sdk = MagicMock()
    sdk.models.generate_content_stream.side_effect = list(streams)
    return GeminiClient(api_key="key", models=["first", "second"], timeout=10, sdk=sdk)


class GeminiClientTests(TestCase):
    """Test streaming recommendations from Gemini."""

    def setUp(self):
        """Clear models marked as unavailable."""
        cache.clear()

    def test_streams_recommendations(self):
        """Parse the streamed JSON and send the structured output schema."""
        text = json.dumps({"items": [{"title": "Dune", "year": 2021}]})
        gemini = client(chunks(text[:10], text[10:]))

        self.assertEqual(list(gemini.recommend("system", "prompt")), [DUNE])
        kwargs = gemini.sdk.models.generate_content_stream.call_args.kwargs
        self.assertEqual(kwargs["model"], "first")
        self.assertEqual(kwargs["contents"], "prompt")
        self.assertEqual(kwargs["config"].system_instruction, "system")
        self.assertIs(kwargs["config"].response_schema, RecommendationListSchema)

    def test_api_error(self):
        """Wrap Gemini API errors, keeping the status code."""
        error = errors.APIError(429, {"error": {"message": "quota exceeded"}})
        gemini = client(error, error)

        with self.assertRaises(LLMClientError) as context:
            list(gemini.recommend("system", "prompt"))

        self.assertEqual(context.exception.status_code, 429)
        self.assertIn("Gemini", str(context.exception))

    def test_network_error_falls_back(self):
        """Try the next model after a timeout before any title."""
        text = json.dumps({"items": [{"title": "Dune", "year": 2021}]})
        gemini = client(
            failing(httpx.ReadTimeout("timed out"), '{"items": ['),
            chunks(text),
        )

        self.assertEqual(list(gemini.recommend("system", "prompt")), [DUNE])

    def test_invalid_response(self):
        """Fail when no title can be parsed."""
        gemini = client(chunks("not json"), chunks("not json"))

        with self.assertRaises(LLMClientError):
            list(gemini.recommend("system", "prompt"))
